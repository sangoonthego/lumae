"""Evidence imports require actual visual inspection; no automatic pass defaults."""
import argparse
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from p2_controlled import pilot,accepted_count
from p16_recovery import O,R,extract,sheet
from data_process.lumae_scale.models import ROOT,atomic_json,read_json,digest
from data_process.lumae_scale.incremental import immutable_write,json_bytes
from data_process.lumae_scale.quality_gate import query_errors
from data_process.lumae_scale.freeze import validate_annotation

def sync_rejected_status(p, source, review):
    if review['decision']=='ACCEPTED_AI_PSEUDO':return
    status='PENDING_CAMPAIGN_REVIEW' if review['decision']=='PENDING_REVIEW' else 'ANNOTATION_REJECTED'
    if source.get('annotation_status')==status:return
    source={**source,'annotation_status':status,'annotation_review_decision':review['decision'],
        'annotation_review_reason':review['reason'],'human_review_status':'NOT_STARTED'}
    p.checkpoint(source);p.event('ANNOTATION_REVIEW_COMPLETED',source_video_id=source['source_video_id'],decision=review['decision'],human_verified=False)

def generate(events_path):
    import numpy as np
    import torch
    from data_process.lumae_scale.clip_runtime import ClipRuntime
    from data_process.lumae_scale.visual_index import extract_clip_features
    from data_process.lumae_scale.temporal_labeler import localize
    p=pilot();torch.set_num_threads(4)
    weights=ROOT/'local_data/cache/clip_weights/ViT-B-32.pt'
    runtime=ClipRuntime(str(weights),device='cpu',download_root=p.work/'model_cache')
    cache=R/'cache'/('p1_'+p.c['run_id'])/'visual_index'
    sources={r['source_video_id']:r for r in p.records()}
    for event in read_json(events_path):
        started=time.time();sid=event['source_id'];r=sources[sid];base=p.work/'evidence'/sid
        folder=base if event.get('revision',1)==1 else base/'revisions'/str(event['revision'])
        folder.mkdir(parents=True,exist_ok=True)
        if (folder/'proposal.json').exists():continue
        if not r.get('technical',{}).get('pass'):raise ValueError('Technically invalid source')
        if not event.get('actual_overview_inspected') or event.get('human_verified') is not False:raise ValueError('Actual AI review attestation required')
        if query_errors(event['query']):raise ValueError(query_errors(event['query']))
        immutable_write(folder/'visual_event.json',json_bytes(event))
        duration=r['technical']['probe']['duration'];start,end=event['coarse_start'],event['coarse_end']
        if not 0<=start<end<=duration:raise ValueError('Invalid event phase')
        p.check_limits(64*2**20)
        if event.get('revision',1)>1:
            # Reuse immutable sampled frames/features, preserving all earlier model outputs.
            metadata=read_json(cache/sid/'metadata.json');times=[f['timestamp'] for f in metadata['frames']]
        else:
            times=sorted(set(np.arange(0,duration-.05,2).tolist()+np.arange(max(0,start-2),min(duration-.05,end+2),.5).tolist()))
            frames=extract(R/r['path'],times,cache/sid)
            metadata={'video_id':sid,'video_path':str(R/r['path']),'video_sha256':r['sha256'],'duration':duration,
                'frames':[{'timestamp':f['timestamp'],'path':Path(f['path']).name,'sha256':f['sha256']} for f in frames],
                'contact_sheets':[str(base/'overview.jpg')],'clip_features_status':'pending'}
            atomic_json(cache/sid/'metadata.json',metadata)
            extract_clip_features(sid,clip_runtime=runtime,visual_cache=cache)
        with np.load(cache/sid/'clip_features.npz') as z:
            t=float(z['timestamps'][int(np.argmax(z['features']@runtime.encode_text(event['query'])[0]))])
        immutable_write(folder/'blind_first.json',json_bytes({'query':event['query'],'interval':[max(0,t-2),min(duration,t+2)],'procedure':'Query-only full-source retrieval; no phase or generator bounds as numerical inputs','same_clip_weights':True,'human_verified':False}))
        event={**event,'evidence_timestamps':[t for t in times if start<=t<end]}
        pred=localize(sid,event['query'],[event],duration,0,clip_runtime=runtime,visual_cache=cache)
        immutable_write(folder/'proposal.json',json_bytes(pred))
        qc=extract(R/r['path'],np.arange(max(0,pred['start']-1),min(duration-.05,pred['end']+1),.5),folder/'qc')
        sheet(qc,folder/'qc.jpg');immutable_write(folder/'qc.json',json_bytes(qc))
        immutable_write(folder/'processing_time.json',json_bytes({'seconds':time.time()-started,'weights_sha256':digest(weights)}))
        print(sid,pred['start'],pred['end'],flush=True)

def decide(decisions_path):
    p=pilot();cache=R/'cache'/('p1_'+p.c['run_id'])/'visual_index'
    sources={r['source_video_id']:r for r in p.records()}
    for review in read_json(decisions_path):
        sid=review['source_id'];r=sources[sid];base=p.work/'evidence'/sid
        folder=base if review.get('revision',1)==1 else base/'revisions'/str(review['revision'])
        target=p.work/'annotation_decisions'/(sid+'.json')
        if target.exists():
            sync_rejected_status(p,r,read_json(target));continue
        if not review.get('reason') or review.get('human_verified') is not False:raise ValueError('Honest AI review provenance required')
        if review['decision']=='ACCEPTED_AI_PSEUDO':
            if not review.get('actual_qc_inspected') or not review.get('full_event_covered') or not review.get('no_excessive_unrelated_shots'):raise ValueError('Incomplete acceptance review')
            if r.get('duplicate_candidates'):raise ValueError('Resolve fingerprint warnings with separate comparison evidence before acceptance')
            pred=read_json(folder/'proposal.json');event=read_json(folder/'visual_event.json');idx=cache/sid/'metadata.json';raw=folder/'proposal.json';qc=folder/'qc.jpg'
            window=[pred['start'],pred['end']]
            record={'qid':'p2_'+sid+'_'+str(review.get('revision',1)),'source_video_id':sid,'source_video_path':r['path'],'source_video_sha256':r['sha256'],'source_url':r['source_url'],'duration':r['technical']['probe']['duration'],
                'query':event['query'],'window':window,'query_sanitation':{'passed':True,'errors':[]},'approximate_event_phase_seconds':[event['coarse_start'],event['coarse_end']],
                'visual_evidence':{'contact_sheets':read_json(idx)['contact_sheets'],'actual_overview_and_qc_inspected':True},'visual_index_sha256':digest(idx),
                'verifier':{'pass':True,'notes':review['reason'],'human_verified':False,'model_id':'Codex multimodal interface','model_revision':'NOT_EXPOSED'},
                'verification_image_path':qc.relative_to(R).as_posix(),'verification_image_sha256':digest(qc),'semantic_artifact_path':raw.relative_to(R).as_posix(),'semantic_artifact_sha256':digest(raw),'semantic_v3_prediction_sha256':digest(raw),'semantic_v3_algorithm_sha256':pred['semantic_v3_algorithm_sha256'],
                'query_source':'AI_GENERATED','query_review_status':'AI_AUTO_ACCEPTED_FOR_TRAINING','temporal_label_source':'SEMANTIC_V3_PSEUDO_LABEL','review_provenance':'AI_PSEUDO_LABELED','reviewer_id':'AI_PIPELINE',
                'query_provider_config':{'provider':'Actual Codex visual review','model_revision':'NOT_EXPOSED'},'semantic_confidence':pred['semantic_confidence'],'human_verified':False,
                'reviewed_interval':review['observed_event_interval'],'campaign_group_id':'campaign_provisional_'+sid,'split_eligibility':'RESTRICTED_PENDING_CAMPAIGN_GROUP_REVIEW','fresh_test_eligible':False,'automatic_high_accept':False,'manual_ai_adjudication':True}
            validate_annotation(record,root=R,visual_cache=cache)
            campaign={'reviewer_id':'AI_PIPELINE','reviewed_at':time.time(),'evidence_paths':[str((base/'overview.jpg').relative_to(p.work)),str((folder/'visual_event.json').relative_to(p.work))],
                'campaign_id':record['campaign_group_id'],'decision':'RETAIN_SPLIT_RESTRICTED','comparison_scope':'D500_AND_PILOT','relationship':'UNKNOWN','fresh_test_eligible':False}
            p.review_source(sid,campaign);p.accept_annotation(record)
        review={**review,'reviewer_interface':'Codex multimodal interface','model_revision':'NOT_EXPOSED','human_IAA':None,'saved_epoch':time.time()}
        immutable_write(target,json_bytes(review))
        sync_rejected_status(p,r,review)
    result=p.export('ANNOTATION_WAVE_CHECKPOINT')
    atomic_json(O/'p2_annotation_progress.json',{'crawler_metrics':result,'accepted_unique_new_pairs':accepted_count(),'remaining_new_pairs':max(0,500-accepted_count()),'human_verified':0})
    print(accepted_count())

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['generate','decide']);ap.add_argument('evidence',type=Path);args=ap.parse_args()
    {'generate':generate,'decide':decide}[args.command](args.evidence)
