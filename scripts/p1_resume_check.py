"""Resume completed real pilot with an HTTP trap; never initiates a request."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from data_process.lumae_scale.acquisition import Pilot,TERMINAL
from data_process.lumae_scale.builder import BuildConfig
from data_process.lumae_scale.models import ROOT,read_json,atomic_json,digest

out=ROOT/'research_outputs/p1_20261103_01';c=read_json(out/'pilot_config.json')
class DenyHTTP:
    def get(self,*args,**kwargs):raise AssertionError('Unexpected network during completed pilot resume')
pilot=Pilot(c,BuildConfig.load(ROOT/c['builder_config'],ROOT),read_json(ROOT/c['source_manifest'])['videos'],live=True,session=DenyHTTP())
records=pilot.records()
if len(records)!=c['max_candidates'] or any(r['status'] not in TERMINAL for r in records):raise ValueError('This check only resumes a fully processed bounded cohort')
before={p.name:digest(p) for p in (pilot.work/'records').glob('*.json')};attempts=pilot.attempts()
value=pilot.run(max_wave=1)
after={p.name:digest(p) for p in (pilot.work/'records').glob('*.json')}
result={'pass':before==after and len(attempts)==len(pilot.attempts()),'before_attempts':len(attempts),'after_attempts':len(pilot.attempts()),'record_bytes_unchanged':before==after,'new_http_requests':0,'http_trap':'Any request would raise AssertionError','real_files_rehashed':len([r for r in records if r.get('sha256')]),'interrupted_transfer_test':'Isolated fault injection only; no deliberately corrupted historical or live files','accepted_annotation_resume':'Synthetic P0 builder test; no actual accepted annotation exists','metrics':value}
atomic_json(out/'resume_validation.json',result);print({k:v for k,v in result.items() if k!='metrics'})
