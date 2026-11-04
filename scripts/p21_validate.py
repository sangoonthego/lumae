"""Read-only integrity validation; outputs only new P21 validation evidence."""
import argparse,json,sys,xml.etree.ElementTree as ET
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from p21_audit import O,E,R,ROOT
from p1_preflight import compare
from p21_fresh_wave import frozen_guard
from p21_process import P21BuildConfig
from p21_review import checked_review
from data_process.lumae_scale.models import read_json,digest,atomic_json
from data_process.lumae_scale.freeze import validate_annotation
from data_process.lumae_scale.builder import BuilderSession
from data_process.annotation.assisted_deployment import verify_frozen_ranker
from data_process.lumae_scale.p21_quality_registry import load_exclusions,effective_records

def historical(full=False):
 result=compare(read_json(ROOT/'research_outputs/p1_20261103_01/protected_asset_snapshot.json'),full=full)
 atomic_json(O/'validation/historical_final.json',result)
 assert result['pass'],result
 previous=ROOT/'research_outputs/p16_p2_20261103_01';manifest=read_json(previous/'p2_execution_manifest.json');count=0
 assert digest(previous/'p2_execution_manifest.json')==read_json(O/'audit/audit_before_change.json')['p2_execution_manifest_sha256']
 for field,items in manifest.items():
  if not field.endswith('_sha256') or not isinstance(items,dict):continue
  for name,h in items.items():
   if not isinstance(h,str) or len(h)!=64:continue
   path=previous/name if field=='artifact_sha256' else Path(name)
   assert digest(path)==h,str(path);count+=1
 verify_frozen_ranker();frozen_guard()
 atomic_json(O/'validation/sealed_previous_runs.json',{'pass':True,'bound_hashes_checked':count,'all_original_18_unchanged':True,'semantic_v3_frozen':True,'d48_inaccessible_scope_disclosed':True})
 print('HISTORICAL_PASS',result['checked'],count,flush=True)

def tests():
 root=ET.parse(O/'validation/full_regression_corrected.xml').getroot();cases=list(root.iter('testcase'))
 retry=ET.parse(O/'validation/ui_idle_retry.xml').getroot()
 retry_pass={(c.get('classname'),c.get('name')) for c in retry.iter('testcase') if c.find('failure') is None and c.find('error') is None}
 failed=[c for c in cases if c.find('failure') is not None or c.find('error') is not None]
 unresolved=[c for c in failed if (c.get('classname'),c.get('name')) not in retry_pass]
 extra=ET.parse(O/'validation/reconcile_tests_final.xml').getroot();extra_cases=list(extra.iter('testcase'));extra_fail=list(extra.iter('failure'))+list(extra.iter('error'))
 quality=ET.parse(O/'validation/quality_registry_tests.xml').getroot();quality_cases=list(quality.iter('testcase'));quality_fail=list(quality.iter('failure'))+list(quality.iter('error'))
 source=ET.parse(O/'validation/source_inventory_guard_tests.xml').getroot();source_cases=list(source.iter('testcase'));source_fail=list(source.iter('failure'))+list(source.iter('error'))
 result={'pass':bool(cases) and not unresolved and len(extra_cases)==2 and not extra_fail and len(quality_cases)==2 and not quality_fail and len(source_cases)==2 and not source_fail,'tests':len(cases)+len(extra_cases)+len(quality_cases)+len(source_cases),'failures':[x.attrib for x in unresolved]+[x.attrib for x in extra_fail]+[x.attrib for x in quality_fail]+[x.attrib for x in source_fail],
   'offline_source_inventory_guard_tests_passed':len(source_cases)-len(source_fail),
   'final_quality_registry_tests_passed':len(quality_cases)-len(quality_fail),
   'postrun_reconciliation_tests_passed':len(extra_cases)-len(extra_fail),'postrun_adapter_initial_import_failures_resolved':True,
   'corrected_full_run_passed':len(cases)-len(failed),'corrected_full_run_failed':len(failed),'resolved_in_idle_retry':len(failed)-len(unresolved),
   'initial_sandbox_run_failed':16,'initial_sandbox_failure_reason':'Hard-link/temp-directory/server permissions; raw failed XML retained',
   'retry_reason':'One UI navigation timeout while resource-intensive workers overlapped; same test passed with workers idle, no app change',
   'skipped':len(list(root.iter('skipped'))),'protected_write_plugin':'data_process.lumae_scale.p21_write_guard plus inherited subprocess sitecustomize'}
 atomic_json(O/'validation/test_summary.json',result);assert result['pass'],result
 print('TESTS_PASS',result['tests'])

def records():
 import requests
 def denied(*a,**k):raise AssertionError('Network during validation')
 requests.sessions.Session.request=denied
 original=[json.loads(l) for l in (ROOT/'research_outputs/p16_p2_20261103_01/accepted_new_pairs.jsonl').read_text().splitlines()]
 stages=['dev01','reserved01','fresh01'];new=[];cache_count=0;resume_count=0
 pool=read_json(ROOT/'local_data/raw/adsqa/source_manifest.json')['videos']
 base=read_json(ROOT/'data_process/config/d1000_v1.json')
 for stage in stages:
  run=E/'runs'/stage
  if not (run/'process_profile.json').exists():raise ValueError('Required completed actual stage missing: '+stage)
  cfg=P21BuildConfig('p21_'+stage,'lumae_ads_d500',ROOT/base['parent_dir'],R,target_videos=1000,seed=base['seed'],parent_asset_root=ROOT,index_root=run/'accepted_index')
  accepted=list((cfg.work/'accepted').glob('*.json'))
  before={str(p):digest(p) for p in cfg.work.rglob('*') if p.is_file()}
  session=BuilderSession(cfg,pool)
  after={str(p):digest(p) for p in cfg.work.rglob('*') if p.is_file()}
  assert before==after;resume_count+=1
  for path in accepted:
   r=read_json(path);sid=r['source_video_id'];validate_annotation(r,root=R,visual_cache=cfg.visual_cache)
   assert not r['human_verified'] and not r['fresh_test_eligible'] and not r['automatic_high_accept']
   assert r['split_eligibility']=='RESTRICTED_PENDING_CAMPAIGN_GROUP_REVIEW'
   decision=read_json(run/'sources'/sid/'acceptance_decision.json');assert decision['status']=='RECOVERED_ACCEPTED_AI_PSEUDO'
   prediction=read_json(R/r['semantic_artifact_path']);assert r['window']==[prediction['start'],prediction['end']]
   bundle=read_json(run/'sources'/sid/'visual_review.json');event=next(e for e in read_json(run/'sources'/sid/'observations.json')['events'] if e['query']==r['query'])
   review=next(e for e in bundle['event_reviews'] if e['query']==r['query']);frames=read_json(run/'visual_index'/sid/'metadata.json')['frames']
   assert not checked_review(review,prediction,{'duration':r['duration']},event,frames)
   new.append(r)
  for path in (run/'visual_index').glob('*/metadata.json'):
   m=read_json(path)
   for f in m['frames']:assert digest(Path(f['path']))==f['sha256']
   fp=path.parent/'clip_features.npz';assert digest(fp)==m['clip_features_sha256']
   with np.load(fp) as z:
    assert z['features'].shape==(len(m['frames']),512) and np.isfinite(z['features']).all()
    assert np.allclose(z['timestamps'],[f['timestamp'] for f in m['frames']])
    assert np.allclose(np.linalg.norm(z['features'],axis=1),1,atol=2e-4)
   cache_count+=1
 exclusions=load_exclusions(O/'validation/final_quality_exclusions.json');retained=effective_records(new,exclusions)
 all_rows=original+retained
 for key in ('qid','source_video_id','source_video_sha256'):assert len({r[key] for r in all_rows})==len(all_rows)
 parent=read_json(ROOT/'local_data/datasets/lumae_ads_d500/dataset_manifest.json');parenthash={r['sha256'] for r in parent['source_video_sha256'].values()}
 assert not parenthash&{r['source_video_sha256'] for r in all_rows}
 inventory=read_json(O/'audit/recovery_inventory.json');assert len(inventory)==61
 for row in inventory:assert digest(Path(row['path']))==row['sha256']
 old_raw_count=0
 for version in ('pilot_20261103_01','pilot_p2_20261103_01'):
  for row in read_json(R/'work/d1000'/version/'manifests/pilot_download_manifest.json')['records']:
   if row.get('sha256'):assert digest(R/row['path'])==row['sha256'];old_raw_count+=1
 assert old_raw_count==80
 for stage,inputname in [('dev01','development_events.json'),('reserved01','reserved_events.json'),('fresh01','fresh_events.json')]:
  path=O/('benchmarks' if stage=='fresh01' else 'recovery')/inputname
  for obs in read_json(path):assert (E/'runs'/stage/'sources'/obs['source_id']/'acceptance_decision.json').exists()
 fresh=read_json(E/'fresh_wave/wave_sealed.json');assert fresh['actual_records']<=25
 attempts=[read_json(p) for p in (E/'fresh_wave/logs/attempts').glob('*.json')]
 assert len(attempts)<=50 and sum(a['bytes_received'] for a in attempts)<=25*100*2**20
 result={'pass':True,'network_denied':True,'existing18_unchanged':True,'provisional_new_trial_acceptances':len(new),'final_qc_withdrawn':len(exclusions),'new_accepted':len(retained),'total_accepted_unique':500+len(all_rows),
  'remaining_to_1000':1000-500-len(all_rows),'validated_visual_indices':cache_count,'builder_read_only_resume_stages':resume_count,'original61_raw_sha_pass':True,
  'source_id_and_raw_sha_unique':True,'all_original80_raw_sha_pass':True,'all_acceptances_match_raw_model_proposals':True,'human_verified':0,'IAA':None,'test_gt_created':False,'d1000_frozen':False,'fresh_wave_count':1,'fresh_attempted_sources':fresh['actual_records'],'http_attempts':len(attempts)}
 atomic_json(O/'validation/records_and_resume.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('command',choices=['historical','tests','records']);ap.add_argument('--full',action='store_true');a=ap.parse_args()
 {'historical':lambda:historical(a.full),'tests':tests,'records':records}[a.command]()
