"""Reopen real accepted ledger and repeat exports with HTTP explicitly trapped."""
import runpy,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import requests
from data_process.lumae_scale.models import ROOT,read_json,atomic_json,digest
round2='--round2' in sys.argv;version='p14_p15_20261103_02' if round2 else 'p14_p15_20261103_01'
O=ROOT/'research_outputs'/version;ledger=Path('E:/LUMAE/work')/version
before={str(p):digest(p) for p in ledger.rglob('*') if p.is_file()};exports={p.name:digest(p) for p in O.glob('*.jsonl')};calls=[]
def deny(*args,**kwargs):calls.append(1);raise AssertionError('HTTP denied during real accepted-ledger resume')
requests.sessions.Session.request=deny
sealed=(O/'pilot30_freeze_manifest.json').exists()
if sealed:
    from data_process.lumae_scale.pilot_completion import validate_completion
    manifest=read_json(O/'pilot30_freeze_manifest.json')
    validate_completion(manifest['sources'],manifest['accepted_annotations'],root=Path('E:/LUMAE'),visual_cache=Path('E:/LUMAE/work/d1000')/version/'visual_index')
    for path,h in manifest['evidence_sha256'].items():
        if digest(Path(path))!=h:raise ValueError('Frozen evidence changed: '+path)
else:
    sys.argv=['scripts/pilot30_finalize.py']+(['--round2'] if round2 else []);runpy.run_path(str(ROOT/'scripts/pilot30_finalize.py'),run_name='__main__')
after={str(p):digest(p) for p in ledger.rglob('*') if p.is_file()};exports_after={p.name:digest(p) for p in O.glob('*.jsonl')}
result={'pass':before==after and exports==exports_after and not calls,'actual_accepted_records':len(list((ledger/'accepted').glob('*.json'))),'accepted_ledger_bytes_unchanged':before==after,'repeated_exports_identical':exports==exports_after,'HTTP_requests':len(calls),'accepted_ids':[r['qid'] for r in read_json(O/'pilot30_accepted_annotation_manifest.json')['annotations']],'rejected_151_658_seconds_preserved':read_json(Path('E:/LUMAE/work/d1000/pilot_20261103_01/records/b5b285ef15c83b8477c638bd0d2e468a.json'))['status']=='REJECTED'}
atomic_json(O/('pilot30_resume_validation.after_freeze.json' if sealed else 'pilot30_resume_validation.json'),result);print(result)
