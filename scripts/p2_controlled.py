"""Bounded waves through the existing Pilot; one persistent global budget clock."""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from p16_recovery import O,R,VERSION,extract,sheet
from data_process.lumae_scale.acquisition import Pilot,validate_scale_gate,storage_check
from data_process.lumae_scale.builder import BuildConfig
from data_process.lumae_scale.incremental import immutable_write,json_bytes
from data_process.lumae_scale.models import ROOT,atomic_json,read_json,digest

CONFIG=O/'p2_config.json'

def initialize():
    c=read_json(ROOT/'research_outputs/p1_20261103_01/scale_config.disabled.json')
    gate=O/'engineering_gate.json';validate_scale_gate({'engineering_gate':{'path':str(gate),'sha256':digest(gate)}})
    operational=O/'operational_gate.json'
    atomic_json(operational,{'protected_assets_pass':read_json(O/'historical_preflight_validation.json')['pass'],
        'storage_pass':shutil.disk_usage(R).free-c['max_staging_bytes']-c['processing_reserve_bytes']>=c['min_free_bytes'],
        'source_permissions_attested':True,'crawler_tests_pass':read_json(O/'test_summary.json')['pass'],
        'source_manifest_sha256':c['source_manifest_sha256'],
        'permission_evidence':'Operator previously confirmed rights; current P1.6/P2 request explicitly authorizes successive controlled waves on existing allowlist. No independent license document verified.'})
    c.update(run_id='p2_20261103_01',network_execution_authorized=True,
        engineering_gate={'path':str(gate),'sha256':digest(gate)},operational_gate=str(operational),
        pilot_pass_evidence=None,annotation_target=500,wave_size=25)
    c['source_permission']={**c['source_permission'],'scope':'P2 controlled acquisition on existing source pool, within persistent configured budgets',
        'authorization_attachment':'fef3cda7-76e3-4659-abb8-8d8500dd8ada/Pasted text.txt'}
    immutable_write(CONFIG,json_bytes(c));storage_check(c,R)
    print(json.dumps({'configuration':str(CONFIG),'initial_accepted_pairs':accepted_count(),'remaining':max(0,500-accepted_count()),'free_bytes':shutil.disk_usage(R).free}))

def pilot(live=False):
    c=read_json(CONFIG)
    assert digest(ROOT/c['source_manifest'])==c['source_manifest_sha256']
    build=BuildConfig.load(ROOT/c['builder_config'],ROOT)
    p=Pilot(c,build,read_json(ROOT/c['source_manifest'])['videos'],live=live)
    clock=p.work/'budget_clock.json'
    if clock.exists():p.budget_started=read_json(clock)['started_epoch']
    return p

def accepted_count():
    old=[json.loads(l) for l in (O/'pilot30_accepted_pairs.jsonl').read_text(encoding='utf-8').splitlines()]
    folder=R/'work/p1_p2_20261103_01/accepted'
    current=[read_json(p) for p in folder.glob('*.json')]
    rows=old+current
    assert len({r['source_video_id'] for r in rows})==len(rows)
    assert len({r['source_video_sha256'] for r in rows})==len(rows)
    return len(rows)

def wave():
    p=pilot(True);validate_scale_gate(p.c)
    capacity=O/'p2_capacity_assessment.json'
    if capacity.exists() and read_json(capacity).get('stop'):
        raise RuntimeError('Projected capacity blocker recorded; use a new validated capacity assessment before further acquisition')
    # Every acquired wave must be reviewed before selecting another one.
    pending=[r for r in p.records() if r.get('technical',{}).get('pass') and not (p.work/'annotation_decisions'/(r['source_video_id']+'.json')).exists()]
    if pending:raise RuntimeError('Complete actual annotation review of pending wave before further acquisition')
    before={r['source_video_id'] for r in p.records()};started=time.time()
    metrics=p.run(max_wave=min(25,p.c['wave_size']))
    new=[r for r in p.records() if r['source_video_id'] not in before]
    directory=p.work/'wave_checkpoints';directory.mkdir(exist_ok=True)
    number=len(list(directory.glob('*_download.json')))+1
    result={'wave':number,'created_epoch':time.time(),'elapsed_seconds':time.time()-started,'new_source_ids':[r['source_video_id'] for r in new],
        'global_crawler_metrics':metrics,'accepted_unique_new_pairs':accepted_count(),'remaining_new_pairs':max(0,500-accepted_count()),
        'candidate_pool_remaining':len(p.plan['candidates'])-len(p.records()),'annotation_status':'PENDING_ACTUAL_VISUAL_REVIEW' if any(r.get('technical',{}).get('pass') for r in new) else 'NO_TECHNICALLY_VALID_SOURCES',
        'free_bytes':shutil.disk_usage(R).free,'final_d1000_freeze':False}
    immutable_write(directory/f'{number:04d}_download.json',json_bytes(result))
    atomic_json(O/'p2_progress.json',result)
    print(json.dumps(result,indent=2))

def overviews():
    import numpy as np
    p=pilot()
    for r in p.records():
        if not r.get('technical',{}).get('pass'):continue
        sid=r['source_video_id'];folder=p.work/'evidence'/sid
        if (folder/'overview.json').exists():continue
        p.check_limits(64*2**20)
        if shutil.disk_usage(R).free-64*2**20<p.c['min_free_bytes']:raise RuntimeError('Processing free-space reserve')
        duration=r['technical']['probe']['duration'];frames=extract(R/r['path'],np.arange(0,duration-.05,2),folder/'frames')
        image=folder/'overview.jpg';sheet(frames,image)
        immutable_write(folder/'overview.json',json_bytes({'source_id':sid,'frames':frames,'contact_sheet':str(image),'sha256':digest(image),'sampling_seconds':2,'source_sha256':r['sha256']}))
        print(sid,str(image),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['init','wave','overviews']);a=ap.parse_args()
    {'init':initialize,'wave':wave,'overviews':overviews}[a.command]()
