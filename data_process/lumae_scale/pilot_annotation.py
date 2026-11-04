"""Offline P1 proposal using inspected events and the unchanged frozen ranker.

No unattended query generation and no automatic human verification. Proposals
are not accepted annotations; full visual verifier evidence is still required.
"""
from pathlib import Path
import argparse
import time
from .models import ROOT,read_json,atomic_json,digest
from .builder import safe_under
from .visual_index import extract_clip_features
from .query_generator import AgentEvidenceProvider,validate_events
from .temporal_labeler import localize,WRAPPER_VERSION
from .clip_runtime import ClipRuntime
from .quality_gate import query_errors,interval_errors
from ..annotation.assisted_deployment import ALGORITHM_SHA256

def propose(config,sid):
    c=read_json(config);root=Path(c['data_root']).resolve();work=root/'work/d1000'/('pilot_'+c['run_id']);cache=root/'cache'/('p1_'+c['run_id'])/'visual_index'
    record=read_json(work/'records'/(sid+'.json'))
    if not record.get('technical',{}).get('pass'):raise ValueError('Source is not technically valid')
    if digest(safe_under(root,record['path']))!=record['sha256']:raise ValueError('Source changed')
    index=read_json(cache/sid/'metadata.json')
    events=AgentEvidenceProvider(work=work,visual_cache=cache).events(sid)
    events=validate_events(sid,events,record['technical']['probe']['duration'],visual_cache=cache)
    if not events:raise ValueError('Actual inspected visual_events.json is required; do not invent model output')
    if any(query_errors(e['query']) for e in events):raise ValueError('Query sanitation failed')
    import torch
    torch.set_num_threads(4)
    weights=ROOT/'local_data/cache/clip_weights/ViT-B-32.pt'
    runtime=ClipRuntime(str(weights),device='cpu',download_root=root/'cache'/('p1_'+c['run_id'])/'models')
    start=time.monotonic()
    extract_clip_features(sid,clip_runtime=runtime,strict=True,visual_cache=cache)
    proposals=[]
    for i,e in enumerate(events):
        prediction=localize(sid,e['query'],events,record['technical']['probe']['duration'],i,clip_runtime=runtime,visual_cache=cache)
        if interval_errors(prediction['start'],prediction['end'],record['technical']['probe']['duration']):raise ValueError('Invalid proposed interval')
        proposals.append({'source_video_id':sid,'source_video_sha256':record['sha256'],'source_video_path':record['path'],'query':e['query'],'prediction':prediction,
                          'status':'AI_PSEUDO_PENDING_VISUAL_INTERVAL_REVIEW','human_verified':False,'query_provider':'AgentEvidenceProvider','events_sha256':digest(work/'visual_index'/sid/'visual_events.json'),
                          'visual_index_sha256':digest(cache/sid/'metadata.json'),'clip_model_weights_sha256':digest(weights),'semantic_v3_algorithm_sha256':ALGORITHM_SHA256,'boundary_wrapper_version':WRAPPER_VERSION})
    value={'proposals':proposals,'elapsed_seconds':time.monotonic()-start,'accepted_annotations':0,'human_verified':0,'required_next':'Actual before/inside/after verification, full freeze.validate_annotation contract, campaign acceptance, then P1 accept-annotation; independent human review remains separate'}
    atomic_json(work/'annotations'/(sid+'_proposals.json'),value)
    from ..annotation.independent_review import ReviewStore
    store=ReviewStore(work/'reviews','p1_'+c['run_id'])
    for i,p in enumerate(proposals):
        qid=f'p1_{sid}_{i}'
        store.register(qid,'lumae_ads_'+sid,p['query'],record['technical']['probe']['duration'],[p['prediction']['start'],p['prediction']['end']])
        atomic_json(work/'reviews'/(qid+'_blind_packet.json'),store.blind_packet(qid))
    return value

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--config',type=Path,required=True);ap.add_argument('--source-id',required=True);a=ap.parse_args()
    if not __import__('re').fullmatch(r'[a-f0-9]{32}',a.source_id):raise ValueError('Unsafe source ID')
    import json
    print(json.dumps(propose(a.config,a.source_id),ensure_ascii=True,indent=2))

if __name__=='__main__':main()
