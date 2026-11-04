"""Read-only, network-denied verification of real accepted pairs and resume state."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from p2_controlled import pilot,accepted_count
from p16_recovery import O,OLD,R,B
from data_process.lumae_scale.builder import BuilderSession
from data_process.lumae_scale.freeze import validate_annotation
from data_process.lumae_scale.pilot_quality import validate_run
from data_process.lumae_scale.models import read_json,atomic_json,digest
from data_process.annotation.assisted_deployment import verify_frozen_ranker
from dataclasses import replace
import requests

def validate(output):
    def denied(*args,**kwargs):raise AssertionError('HTTP attempted during offline validation')
    requests.sessions.Session.request=denied
    p=pilot(False);verify_frozen_ranker()
    config=replace(p.build,version='p1_'+p.c['run_id'],data_root=R)
    accepted=[read_json(f) for f in (config.work/'accepted').glob('*.json')]
    before={str(f):digest(f) for folder in (config.work,p.work/'reviews',p.work/'annotation_decisions') for f in folder.rglob('*') if f.is_file()}
    session=BuilderSession(config,p.pool)
    assert len(session.accepted)==len(accepted)
    after={str(f):digest(f) for folder in (config.work,p.work/'reviews',p.work/'annotation_decisions') for f in folder.rglob('*') if f.is_file()}
    assert before==after
    previous=[json.loads(l) for l in (O/'pilot30_accepted_pairs.jsonl').read_text(encoding='utf-8').splitlines()]
    for r in previous:validate_annotation(r,root=R,visual_cache=B/'visual_index')
    for r in accepted:
        validate_annotation(r,root=R,visual_cache=config.visual_cache)
        assert r['human_verified'] is False and r['fresh_test_eligible'] is False
        packet=read_json(p.work/'reviews'/(r['source_video_id']+'_blind_packet.json'))
        assert packet['qid']==r['qid']
    all_rows=previous+accepted
    for key in ('qid','source_video_id','source_video_sha256'):
        assert len({r[key] for r in all_rows})==len(all_rows)
    actual=validate_run(p.work,R,p.parent_hashes)
    assert actual['integrity_pass'],actual
    oldfreeze=OLD/'pilot30_freeze_manifest.json';m=read_json(oldfreeze)
    assert digest(oldfreeze)==oldfreeze.with_suffix('.sha256').read_text().strip()
    for field in ('processing_code_sha256','evidence_sha256'):
        for name,h in m[field].items():assert digest(Path(name))==h,name
    for name,h in m['artifact_sha256'].items():assert digest(OLD/name)==h,name
    # Selection reconstruction is deterministic, and no completed IDs can be fetched anew.
    again=pilot(False);assert again.plan==p.plan
    original={r['source_video_id'] for r in p.previous};assert not original & {r['source_video_id'] for r in p.plan['candidates']}
    result={'pass':True,'network_denied':True,'resume_unchanged_sha256':before==after,'accepted_new_pairs':accepted_count(),
        'unique_source_query_hashes':True,'remaining_new_pairs':max(0,500-accepted_count()),'p14_frozen_integrity_pass':True,
        'semantic_v3_frozen_pass':True,'human_verified':0,'human_IAA':None,'source_validation':actual,
        'completed_source_ids_reused':False,'downloaded_original30_again':False,'final_d1000_freeze':False}
    atomic_json(output,result);print(json.dumps(result,indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);args=ap.parse_args();validate(args.output)
