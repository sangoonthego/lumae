import importlib.util,json
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('backend_probe',ROOT/'scripts/d1000_auto_backend_check.py')
probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)

@pytest.mark.parametrize('answer',['Yes, a person is visible.','{"events": [{"query":"a person"}]}','{"events": "YES"}','{"events":[{"query":"A woman lifts a red cup from a table.","visible_subject":"woman","visible_action":"lifts","visible_object":"cup","evidence_timestamps":[2],"coarse_start":4,"coarse_end":2}]}'])
def test_unstructured_or_invalid_interval_is_not_model_event(answer):
 assert not probe.structured(answer)

def test_real_inference_evidence_is_not_reported_as_full_pipeline_pass():
 r=json.loads((probe.OUT/'auto_annotation_smoke_results.json').read_text())
 assert len(r['sources'])==len({x['source_video_id'] for x in r['sources']})==10
 assert r['accepted']==0 and r['full_pipeline_smoke_pass'] is False
 assert all(x['actual_model_generate_executed'] and not x['manual_observations_used'] and not x['manual_verdicts_used'] for x in r['sources'])
 assert all(x['state']=='ABSTAIN_BACKEND_NOT_VALIDATED' and not x['localization_executed'] for x in r['sources'])
 for x in r['sources']:
  assert probe.digest(Path(x['source_path']))==x['source_sha256']
  assert probe.digest(Path(x['image']))==x['image_sha256']

def test_diagnostic_resume_does_not_generate_again_or_change_records(capsys):
 files=list((probe.WORK/'diagnostics').glob('*.json'))
 before={str(p):probe.digest(p) for p in files}
 probe.main();captured=capsys.readouterr().out
 assert 'RESUMED_IDENTICAL_DIAGNOSTICS' in captured and '"model_calls": 0' in captured
 assert before=={str(p):probe.digest(p) for p in files}
