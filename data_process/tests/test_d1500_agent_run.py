import sys
from pathlib import Path

import pytest


@pytest.fixture
def controller(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / 'scripts'))
    import d1500_agent_run
    return d1500_agent_run


def test_inventory_prefers_cached_source_and_excludes_prior_urls(controller):
    host = 'https://video.adsoftheworld.com'
    rows = [{'source_video_id': sid, 'source_url': url} for sid, url in [
        ('a', host + '/parent'),
        ('b', host + '/same'),
        ('c', host + ':443/same'),
        ('d', host + '/rejected'),
        ('e', 'https://example.org/video'),
    ]]
    result = controller.eligible_inventory(rows, {'d'}, [host + '/parent'], {'c': ['cached.mp4']})
    assert result == [{**rows[2], 'existing_paths': ['cached.mp4']}]


def test_shared_loop_isolates_source_failures_and_keeps_d1500_count(controller, tmp_path, monkeypatch):
    agent = controller.agent
    work = tmp_path / 'work'
    out = tmp_path / 'out'
    work.mkdir()
    raw = work / 'duplicate.mp4'
    raw.write_bytes(b'parent video')
    corrupt = work / 'corrupt.mp4'
    corrupt.write_bytes(b'undecodable frames')
    canonical = tmp_path / 'parent.jsonl'
    agent.jsonl(canonical, [{'qid': 'parent-q', 'vid': 'lumae_ads_parent'}])
    config = {
        'base_count': 1, 'target_new': 500, 'parent': str(canonical), 'parent_hashes': [agent.digest(raw)],
        'protected': {str(canonical): agent.digest(canonical)},
    }
    agent.atomic_json(work / 'config.json', config)
    agent.atomic_json(work / 'inventory.json', [
        {'source_video_id': 'corrupt', 'existing_paths': [str(corrupt)]},
        {'source_video_id': 'duplicate', 'existing_paths': [str(raw)]},
    ])
    monkeypatch.setattr(agent, 'probe_video_file', lambda path: {'is_model_compatible': True, 'duration': 30})

    def fail_decode(*args, **kwargs):
        raise ValueError('Frame decode failed')

    monkeypatch.setattr(agent, 'build_visual_index', fail_decode)
    monkeypatch.setattr(controller, 'BASE', work)
    monkeypatch.setattr(controller, 'OUT', out)
    monkeypatch.setattr(controller, 'init', lambda: config)
    with controller.bound_agent():
        result = agent.run(prepare=True)
        assert result['count'] == 0
        summary = agent.progress()
        assert summary['total'] == 1 and summary['remaining'] == 500 and summary['duplicate_rejections'] == 1
        assert summary['technical_rejections'] == 1
        assert agent.read_json(work / 'states/corrupt.json')['reason'] == 'Frame decode failed'
        assert agent.load_jsonl(canonical) == [{'qid': 'parent-q', 'vid': 'lumae_ads_parent'}]
        agent.jsonl(canonical, [{'qid': 'changed-q', 'vid': 'lumae_ads_parent'}])
        with pytest.raises(AssertionError, match='Protected state changed'):
            agent.progress()
