"""Extend the frozen D1000 corpus using the D500-compatible visual review loop."""

import argparse
from collections import Counter
from contextlib import contextmanager
import json
from pathlib import Path
import sys
from urllib.parse import urlsplit

sys.dont_write_bytecode = True
import d1000_agent_run as agent
from data_process.lumae_scale.acquisition import PermanentSource, allowed_url, run_lock

RUN = 'd1500_agent_20261106_01'
BASE = Path('E:/LUMAE/work/d1500') / RUN
OUT = agent.ROOT / 'research_outputs' / RUN
D1000_WORK = agent.BASE
D1000_OUT = agent.OUT
D1000 = D1000_OUT / 'lumae_ads_d1000'
TARGET_NEW = 500


def url_identity(url):
    parsed = urlsplit(allowed_url(url))
    return parsed.path or '/', parsed.query


def eligible_inventory(rows, excluded_ids, source_urls, raw):
    """Reconcile prior decisions, one source URL per candidate, and reusable files."""
    seen = {url_identity(url) for url in source_urls}
    eligible = []
    for row in sorted(rows, key=lambda row: (row['source_video_id'] not in raw, row['source_video_id'])):
        sid = row['source_video_id']
        if sid in excluded_ids:
            continue
        try:
            key = url_identity(row['source_url'])
        except PermanentSource:
            continue
        if key in seen:
            continue
        seen.add(key)
        eligible.append({**row, 'existing_paths': raw.get(sid, [])})
    return eligible


def init():
    manifest = agent.read_json(D1000 / 'dataset_manifest.json')
    canonical = D1000 / 'lumae_ads_d1000_all.jsonl'
    parent = agent.load_jsonl(canonical)
    source_hashes = manifest['source_video_sha256']
    assert manifest['acquisition_complete'] is True
    assert len(parent) == len({r['qid'] for r in parent}) == len(source_hashes) == len(set(source_hashes.values())) == 1000
    assert {r['vid'] for r in parent} == set(source_hashes)

    pin_path = D1000 / 'checksums.sha256'
    pins = {str(D1000 / name): checksum for checksum, name in (
        line.split('  ', 1) for line in pin_path.read_text(encoding='utf-8').splitlines())}
    assert pins[str(canonical)] == manifest['dataset_sha256']
    validation = agent.read_json(D1000_OUT / 'feature_validation.json')
    feature_manifest = D1000_OUT / 'unified_feature_manifest.json'
    features = agent.read_json(feature_manifest)
    assert features['canonical_sha256'] == manifest['dataset_sha256']
    assert features['counts'] == {'video': 1000, 'text': 1000} and validation['single_root_preflight_pass']
    assert set(features['video']) == set(source_hashes) and set(features['text']) == {r['qid'] for r in parent}
    pins.update({
        str(D1000_OUT / 'lumae_d1000_colab.zip'): validation['package_sha256'],
        str(feature_manifest): validation['unified_manifest_sha256'],
    })
    pin = BASE / 'config.json'
    saved = agent.read_json(pin) if pin.exists() else None
    for path in (pin_path, D1000_OUT / 'feature_checksums.json', D1000_OUT / 'feature_validation.json',
                 D1000_OUT / 'cached_runner_config.json', D1000_WORK / 'inventory.json'):
        pins[str(path)] = saved['protected'][str(path)] if saved else agent.digest(path)
    agent.verify_pins(pins)
    config = {
        'parent': str(canonical), 'parent_sha256': manifest['dataset_sha256'],
        'parent_hashes': sorted(source_hashes.values()), 'protected': pins,
        'base_count': 1000, 'target_new': TARGET_NEW, 'qid_prefix': 'd1500_compat_',
        'feature_binding': {'manifest': str(feature_manifest), 'sha256': pins[str(feature_manifest)],
                            'video': 1000, 'text': 1000},
        'd1000_campaign_split_frozen': manifest['campaign_disjoint_certified'],
        'policy': agent.POLICY, 'wrapper': agent.WRAPPER_VERSION,
    }
    assert config['d1000_campaign_split_frozen'] is False, 'Reconcile against the frozen campaign split before continuing'
    if saved:
        assert saved == config, 'D1500 resume inputs changed'
    else:
        agent.atomic_json(pin, config)

    roots = agent.read_json(agent.P21 / 'candidate_ranking/next_pool_cache_audit.json')['registered_roots']
    roots = [*roots, *map(str, sorted((D1000_WORK / 'network').glob('batch_*/downloads')))]
    agent.NETWORK_OPTIONS = {
        'registered_roots': roots, 'parent_hashes': config['parent_hashes'], 'scope': 'D1500_D500_COMPAT_INTERNAL_BATCH',
        'source_permission': {
            'scope': 'D1500 agent-driven collection, existing allowlist and unchanged P1 limits',
            'authorization_attachment': '902c00f9-c4a3-40d1-9fe4-7c41d90ed475/Pasted text.txt sections 3 and 7',
        },
    }
    inventory = BASE / 'inventory.json'
    if not inventory.exists():
        states = {p.stem for p in (D1000_WORK / 'states').glob('*.json')}
        attempted = {r['source_video_id'] for p in (D1000_WORK / 'network').glob('batch_*/source_selection.json')
                     for r in agent.read_json(p)['candidates']}
        parent_ids = {r['vid'].removeprefix('lumae_ads_') for r in parent}
        parent_ids.update(Path(r['metadata']['video_filename']).stem for r in parent)
        pool = agent.read_json(agent.ROOT / 'local_data/raw/adsqa/source_manifest.json')['videos']
        excluded = parent_ids | states | attempted
        prior_urls = [r['source_url'] for r in pool if r['source_video_id'] in excluded]
        agent.atomic_json(inventory, eligible_inventory(pool, excluded, prior_urls, agent.inventory_mp4(roots)))
    return config


def progress():
    result = agent_progress()
    batches = []
    for batch in (BASE / 'network').glob('batch_*'):
        complete = batch / 'bulk_complete.json'
        if complete.is_file():
            batches.append(agent.read_json(complete))
        else:
            checkpoint = batch / 'manifests/pilot_download_manifest.json'
            if checkpoint.is_file():
                batches.append(agent.read_json(checkpoint)['metrics'])
    counts = Counter(result['current_run_outcomes'])
    bindings = BASE / 'training_features/bindings'
    result.update(
        sources_attempted=sum(b['attempted_candidates'] for b in batches),
        downloads=sum(b['successful_downloads'] for b in batches),
        technical_rejections=counts['REJECTED_TECHNICAL'], duplicate_rejections=counts['REJECTED_DUPLICATE'],
        annotation_rejections=counts['REJECTED_ANNOTATION'], feature_pairs_ready=sum(1 for _ in bindings.glob('*.json')),
        split_counts={'TRAIN_ELIGIBLE': 0, 'SPLIT_RESTRICTED': 0, 'CAMPAIGN_UNKNOWN': result['new_accepted']},
    )
    agent.atomic_json(OUT / 'progress.json', result)
    return result


def resume_batch_15():
    init()
    if progress()['cached_remaining']:
        raise ValueError('Finish the available local review queue before reopening acquisition')
    agent.network_batch(resume_connection_batch=BASE / 'network/batch_0015')
    return agent.run(prepare=True)


def freeze():
    config = init()
    summary = progress()
    if not summary['complete']:
        raise ValueError('D1500 freeze requires 500 accepted additions')
    frozen = OUT / 'lumae_ads_d1500'
    rows = agent.load_jsonl(OUT / 'accepted_annotations.jsonl')
    canonical = frozen / 'lumae_ads_d1500_all.jsonl'
    agent.jsonl(canonical, agent.load_jsonl(OUT / 'provisional_dataset.jsonl'))
    additions = frozen / 'accepted_annotations.jsonl'
    agent.jsonl(additions, rows)
    manifest = {
        'dataset_version': 'lumae_ads_d1500', 'run_id': RUN, 'acquisition_complete': True,
        'dataset_sha256': agent.digest(canonical), 'd1000_canonical_sha256': config['parent_sha256'],
        'd1000_manifest_sha256': config['protected'][str(D1000 / 'dataset_manifest.json')],
        'counts': {'frozen_d1000': 1000, 'additional_sources': TARGET_NEW, 'total': 1500},
        'additions_sha256': agent.digest(additions),
        'source_video_sha256': {**agent.read_json(D1000 / 'dataset_manifest.json')['source_video_sha256'],
                                **{'lumae_ads_' + r['source_video_id']: r['source_video_sha256'] for r in rows}},
        'additional_human_verified': 0, 'split_counts': summary['split_counts'],
        'd1000_campaign_split_frozen': config['d1000_campaign_split_frozen'], 'evaluation_ready': False,
        'additional_features': {'required_pairs': TARGET_NEW, 'manifest': 'new_feature_manifest.json',
                                'validation': 'feature_validation.json'},
    }
    manifest_path = frozen / 'dataset_manifest.json'
    agent.atomic_json(manifest_path, manifest)
    checksums = frozen / 'checksums.sha256'
    temporary = checksums.with_suffix('.tmp')
    temporary.write_bytes(''.join(f'{agent.digest(path)}  {path.name}\n'
                                  for path in (additions, manifest_path, canonical)).encode('ascii'))
    temporary.replace(checksums)
    return manifest['counts']


def handoff():
    from zipfile import ZIP_DEFLATED, ZipFile
    from experiments.M1_lumae_ads import cached_runner

    config = init()
    progress()
    folder = OUT / 'lumae_ads_d1500'
    dataset = folder / 'dataset_manifest.json'
    freeze_manifest = agent.read_json(dataset)
    assert freeze_manifest['acquisition_complete']
    additions = folder / 'accepted_annotations.jsonl'
    assert agent.digest(additions) == freeze_manifest['additions_sha256']
    assert agent.digest(folder / 'lumae_ads_d1500_all.jsonl') == freeze_manifest['dataset_sha256']
    index_path = OUT / 'new_feature_manifest.json'
    index = agent.read_json(index_path)
    parent = agent.read_json(D1000_OUT / 'unified_feature_manifest.json')
    assert index['feature_contract'] == parent['feature_contract']
    assert index['historical_binding'] == config['feature_binding']
    rows = agent.load_jsonl(additions)
    canon = OUT / 'additions_loader.jsonl'
    agent.jsonl(canon, [agent.canonical_row(r, r['qid']) for r in rows])
    assert set(index['video']) == {'lumae_ads_' + r['source_video_id'] for r in rows}
    assert set(index['text']) == {r['qid'] for r in rows}
    root = BASE / 'training_features'
    cls = cached_runner.dataset_type(agent.ROOT / 'research_outputs/p0_20261103_01/upstream')
    loader = cls(canon, root, index, cached_runner.read(cached_runner.CONFIG))
    assert len(loader) == TARGET_NEW
    for sample in loader:
        assert sample['model_inputs']['video_feat'].shape[1] == 514
        assert sample['model_inputs']['query_feat'].shape[1] == 512
    assert len(index['checksums']) == 2 * TARGET_NEW
    for relative, checksum in index['checksums'].items():
        assert agent.digest(cached_runner.feature_path(root, relative)) == checksum
    guide = OUT / 'COLAB_HANDOFF.md'
    guide.write_text(
        '# LUMAE D1500 incremental features\n\n'
        'Extract this ZIP into a separate D1500 directory in Colab. Keep the existing D1000 ZIP and its '
        'extracted directory separate. Verify the SHA-256 of its unified_feature_manifest.json against '
        'historical_binding.sha256 in the new feature manifest.\n\n'
        'Use additions_loader.jsonl with the existing CachedDataset loader, new_feature_manifest.json '
        'as the feature index, and the extracted features directory as its feature root. Verify the '
        'files against the checksums in that index. The existing loader adds TEF when reading video features.\n\n'
        'Reconcile the campaign groups with a frozen D1000 evaluation split before generating a D1500 '
        'training split. The current additions file covers acquisition, with provisional eligibility.\n',
        encoding='utf-8',
    )
    archive = OUT / 'lumae_d1500_incremental_colab.zip'
    temp = archive.with_suffix('.zip.tmp')
    with ZipFile(temp, 'w', compression=ZIP_DEFLATED) as z:
        for p in (dataset, canon, guide, index_path):
            z.write(p, p.name)
        for relative in sorted(index['checksums']):
            z.write(root / relative, 'features/' + relative)
    with ZipFile(temp) as z:
        assert z.testzip() is None
    temp.replace(archive)
    result = {'cached_dataset_pass': True, 'feature_pairs': len(loader),
              'dataset_manifest_sha256': agent.digest(dataset), 'feature_manifest_sha256': agent.digest(index_path),
              'd1000_feature_manifest_sha256': config['feature_binding']['sha256'],
              'package': str(archive), 'package_sha256': agent.digest(archive)}
    agent.atomic_json(OUT / 'feature_validation.json', result)
    return result


agent_progress = agent.progress


@contextmanager
def bound_agent():
    """Bind the existing review functions only within this D1500 invocation."""
    values = {'BASE': BASE, 'OUT': OUT, 'init': init, 'progress': progress, 'NETWORK_OPTIONS': {}}
    old = {name: getattr(agent, name) for name in values}
    try:
        for name, value in values.items():
            setattr(agent, name, value)
        yield
    finally:
        for name, value in old.items():
            setattr(agent, name, value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['init', 'run', 'status', 'prepare', 'observe', 'propose', 'verdict',
                                            'features', 'freeze', 'handoff', 'resume-batch-15'])
    parser.add_argument('rows', nargs='?', type=Path)
    args = parser.parse_args()
    with bound_agent(), run_lock(BASE / ('feature_extraction.lock' if args.command == 'features' else 'controller.lock')):
        if args.command in ('observe', 'verdict'):
            if not args.rows:
                parser.error('An observations or verdicts JSON file is required')
            getattr(agent, args.command)(agent.read_json(args.rows))
            return
        actions = {'init': lambda: (init(), progress())[1], 'status': progress,
                   'features': agent.features, 'freeze': freeze, 'handoff': handoff,
                   'resume-batch-15': resume_batch_15}
        action = actions.get(args.command)
        result = action() if action else agent.run(args.command == 'prepare', args.command == 'propose')
        print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == '__main__':
    main()
