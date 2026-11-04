"""One explicitly authorized P21 validation wave through the unchanged Pilot."""
import argparse,json,sys,time,shutil
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from p21_audit import O,E,R,ROOT
from data_process.lumae_scale.models import read_json,digest,atomic_json
from data_process.lumae_scale.incremental import immutable_write,json_bytes
from data_process.lumae_scale.acquisition import Pilot,validate_config,storage_check
from data_process.lumae_scale.builder import BuildConfig

class ValidationPilot(Pilot):
 def __init__(self,c,build,pool,**kwargs):
  super().__init__(c,build,pool,**kwargs)
  self.work=E/'fresh_wave'
  eligible={r['source_video_id']:r for r in self.plan['candidates']}
  selection=read_json(O/'candidate_ranking/fresh25_selection.json')
  chosen=[eligible[r['source_video_id']] for r in selection['selected']]
  assert len(chosen)==25 and len({r['source_video_id'] for r in chosen})==25
  # The target is scoped to this new single wave. Previous bytes/IDs still
  # participate in duplicate exclusion and pinned parent verification.
  self.plan.update(candidates=chosen,previous_accepted_sources=0,scope='P21_SINGLE_VALIDATION_WAVE',selection_sha256=digest(O/'candidate_ranking/fresh25_selection.json'))

def frozen_guard():
 f=read_json(O/'manifests/experimental_pipeline_freeze.json')
 for p,h in f['code_sha256'].items():
  if digest(ROOT/p)!=h:raise ValueError('Experimental pipeline changed after freeze: '+p)
 for p,key in [('annotation_v4/config.json','config_sha256'),('candidate_ranking/policy.md','candidate_policy_sha256'),('candidate_ranking/fresh25_selection.json','selection_sha256')]:
  if digest(O/p)!=f[key]:raise ValueError('Frozen experimental input changed')
 if not read_json(O/'validation/historical_preflight.json')['pass']:raise PermissionError('Historical gate')
 extra=O/'benchmarks/fresh_execution_code_freeze.json'
 if extra.exists():
  for p,h in read_json(extra)['code_sha256'].items():
   if digest(ROOT/p)!=h:raise ValueError('Fresh execution helper changed after freeze: '+p)

def configuration():
 frozen_guard();c=read_json(ROOT/'research_outputs/p16_p2_20261103_01/p2_config.json')
 prior=R/'work/d1000/pilot_p2_20261103_01/manifests/pilot_download_manifest.json'
 c.update(run_id='p21_fresh_20261103_01',mode='pilot',target_accepted=25,max_candidates=25,max_http_requests=50,
    max_download_bytes=25*100*2**20,max_staging_bytes=3*2**30,processing_reserve_bytes=2*2**30,
    max_runtime_seconds=8*3600,wave_size=25,network_execution_authorized=True,
    operational_gate=str(O/'benchmarks/fresh_operational_gate.json'),engineering_gate=None)
 c['previous_runs']=c['previous_runs']+[{'manifest':prior.relative_to(R).as_posix(),'sha256':digest(prior)}]
 c['source_permission']={**c['source_permission'],'scope':'P21 one validation wave, max25 previously unattempted allowlisted candidates',
    'authorization_attachment':'4719e9f8-aa3f-4902-acee-36eb6b0a48a1/Pasted text.txt sections10.2 and21; prior operator attestation retained'}
 return c

def pilot(live=False):
 c=read_json(O/'benchmarks/fresh_wave_config.json')
 b=BuildConfig.load(ROOT/c['builder_config'],ROOT)
 return ValidationPilot(c,b,read_json(ROOT/c['source_manifest'])['videos'],live=live)

def initialize():
 c=configuration();validate_config(c);free=storage_check(c,R)
 tests=read_json(O/'validation/prewave_test_summary.json')
 if not tests['pass']:raise PermissionError('P21 prewave regression tests failed')
 gate={'protected_assets_pass':True,'storage_pass':True,'source_permissions_attested':True,'crawler_tests_pass':True,
    'source_manifest_sha256':c['source_manifest_sha256'],'free_bytes':free,'test_evidence_sha256':digest(O/'validation/prewave_test_summary.json'),
    'annotation_status':'EXPERIMENTAL; no automatic acceptance or production promotion',
    'authorization':'Current task explicitly permits one <=25-source fresh wave; previous run remains sealed.'}
 immutable_write(O/'benchmarks/fresh_operational_gate.json',json_bytes(gate));immutable_write(O/'benchmarks/fresh_wave_config.json',json_bytes(c))
 p=pilot();assert len(p.plan['candidates'])==25
 immutable_write(O/'benchmarks/fresh_wave_plan.json',json_bytes(p.plan));print('P21_SINGLE_WAVE_READY',free)

def wave():
 frozen_guard();p=pilot(True)
 if (E/'fresh_wave/wave_sealed.json').exists():raise RuntimeError('One-wave budget exhausted; no additional run')
 helper_files=['scripts/p21_fresh_wave.py','scripts/p21_fresh_prepare.py','scripts/p21_fresh_notes.py']
 immutable_write(O/'benchmarks/fresh_execution_code_freeze.json',json_bytes({'code_sha256':{f:digest(ROOT/f) for f in helper_files},'scope':'Frozen before live timing and fresh image exposure'}))
 if not (O/'benchmarks/fresh_pipeline_clock.json').exists():
  immutable_write(O/'benchmarks/fresh_pipeline_clock.json',json_bytes({'started_epoch':time.time(),'start_stage':'frozen selection ready; no setup/coding included','selection_seconds':None,'selection_reason':'Selection precomputed during setup, unmeasured cost excluded; no ranking superiority claimed'}))
 started=time.perf_counter();metrics=p.run(max_wave=25)
 immutable_write(E/'fresh_wave/wave_sealed.json',json_bytes({'single_wave_complete':True,'actual_records':len(p.records()),'download_and_probe_wall_seconds':time.perf_counter()-started,'metrics':metrics,'no_second_wave_authorized':True}))
 print(json.dumps(metrics,indent=2))

def timer(phase,action):
 folder=O/'benchmarks/interactive_timing';folder.mkdir(exist_ok=True)
 path=folder/(phase+'_'+action+'.json')
 immutable_write(path,json_bytes({'phase':phase,'action':action,'epoch':time.time(),'clock':'actual wall clock; no reconstructed per-source allocation'}))
 print(path)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('command',choices=['init','wave','timer']);ap.add_argument('--phase');ap.add_argument('--action',choices=['start','end']);a=ap.parse_args()
 if a.command=='init':initialize()
 elif a.command=='wave':wave()
 else:timer(a.phase,a.action)
