"""Actual image review import and strict existing-contract acceptance for P21."""
import argparse
import json
import math
import os
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from p21_process import O,E,R,ROOT,P21BuildConfig,source_rows,make_sheet,config
from data_process.lumae_scale.models import read_json,digest
from data_process.lumae_scale.incremental import immutable_write,json_bytes
from data_process.lumae_scale.annotation_v4 import review_errors
from data_process.lumae_scale.freeze import validate_annotation
from data_process.lumae_scale.builder import BuilderSession
from data_process.annotation.independent_review import ReviewStore

def sheets(stage):
    run=E/'runs'/stage
    for folder in sorted((run/'sources').iterdir()):
        if not (folder/'proposals.json').exists():continue
        frames=read_json(run/'visual_index'/folder.name/'metadata.json')['frames']
        observed=read_json(folder/'observations.json')['events']
        regions=[(max(0,e['coarse_start']-2),e['coarse_end']+2) for e in observed]
        proposals=read_json(folder/'proposals.json')
        regions += [(p['prediction']['start']-.5,p['prediction']['end']+.5) for p in proposals if 'prediction' in p]
        visible=[f for f in frames if any(s<=f['timestamp']<=e for s,e in regions)]
        files=[]
        for page in range(math.ceil(len(visible)/60)):
            target=folder/f'review_grid_{page+1:02d}.jpg'
            if not target.exists():make_sheet(visible[page*60:(page+1)*60],target)
            files.append({'path':str(target),'sha256':digest(target),'timestamps':[f['timestamp'] for f in visible[page*60:(page+1)*60]]})
        immutable_write(folder/'review_grid_manifest.json',json_bytes(files))
        print(folder.name,files,flush=True)

def checked_review(review,prediction,row,event,frames):
    errors=review_errors(review,prediction,row['duration'])
    c=config();step=c['short_step'] if event['coarse_end']-event['coarse_start']<=c['short_phase_max_seconds'] else c['dense_step']
    known={round(f['timestamp'],6) for f in frames}|{0.0,float(row['duration'])}
    for field in ('onset_bracket_seconds','offset_bracket_seconds'):
        bracket=review.get(field,[])
        if len(bracket)==2:
            if any(round(t,6) not in known for t in bracket):errors.append('Bracket lacks actual sampled timestamp')
            if bracket[1]-bracket[0]>step+1e-5:errors.append('Boundary evidence too sparse for fixed protocol')
    return errors

def import_reviews(stage,path):
    run=E/'runs'/stage;rows={r['source_id']:r for r in source_rows(stage)}
    base=read_json(ROOT/'data_process/config/d1000_v1.json')
    cfg=P21BuildConfig('p21_'+stage,'lumae_ads_d500',ROOT/base['parent_dir'],R,
        target_videos=1000,seed=base['seed'],parent_asset_root=ROOT,index_root=run/'accepted_index')
    pool=read_json(ROOT/'local_data/raw/adsqa/source_manifest.json')['videos'];builder=BuilderSession(cfg,pool)
    old=Path(read_json(O/'audit/audit_before_change.json')['latest_p2_directory'])
    previous=[json.loads(l) for l in (old/'accepted_new_pairs.jsonl').read_text(encoding='utf-8').splitlines()]
    prior_ids={r['source_video_id'] for r in previous};prior_hashes={r['source_video_sha256'] for r in previous}
    decisions=[]
    for bundle in read_json(path):
        sid=bundle['source_id'];row=rows[sid];folder=run/'sources'/sid;destination=folder/'visual_review.json'
        immutable_write(destination,json_bytes(bundle))
        if not bundle.get('actual_complete_overview_inspected'):raise ValueError('Full-source visual evidence required')
        observations=read_json(folder/'observations.json');events={e['query']:e for e in observations.get('events',[])}
        if not events:
            decision={'source_id':sid,'status':'NO_DEFENSIBLE_EVENT','reason':bundle['reason'],'human_verified':False}
            immutable_write(folder/'acceptance_decision.json',json_bytes(decision));decisions.append(decision);continue
        grids=read_json(folder/'review_grid_manifest.json')
        if set(bundle.get('inspected_review_grids',[]))!={g['path'] for g in grids}:raise ValueError('Actual inspection of all review pages required')
        for g in grids:
            if digest(Path(g['path']))!=g['sha256']:raise ValueError('Changed reviewed image')
        supplied={r['query']:r for r in bundle['event_reviews']}
        all_proposals=read_json(folder/'proposals.json');variant_decisions=[];eligible=[]
        for proposal in all_proposals:
            if 'prediction' not in proposal:
                variant_decisions.append({'variant':proposal['variant'],'status':'ABSTAIN','reason':proposal.get('reason',proposal['status'])});continue
            q=proposal['query'];review=supplied.get(q)
            if review is None:raise ValueError('Missing actual review for distinct candidate query')
            errors=checked_review(review,proposal['prediction'],row,events[q],read_json(run/'visual_index'/sid/'metadata.json')['frames'])
            verdict={'variant':proposal['variant'],'query':q,'window':[proposal['prediction']['start'],proposal['prediction']['end']],
                'status':'AI_VISUAL_BRACKETS_PASS' if not errors else 'REJECTED','errors':errors,'reason':review['reason'],
                'onset_bracket_seconds':review.get('onset_bracket_seconds'),'offset_bracket_seconds':review.get('offset_bracket_seconds'),
                'ground_truth_status':'AI_EVIDENCE_AGREEMENT_NOT_HUMAN_GT','visual_review_blinding':'UNBLINDED','human_verified':False}
            variant_decisions.append(verdict)
            if not errors and proposal['variant'] in ('B4','B3','B2','B0'):eligible.append(proposal)
        status='STILL_REJECTED';selected=None
        if eligible:
            priority={'B4':0,'B3':1,'B2':2,'B0':3};selected=min(eligible,key=lambda p:priority[p['variant']])
            status='RECOVERED_ACCEPTED_AI_PSEUDO'
            if row.get('duplicate_candidates'):
                resolution=O/'recovery/campaign_resolutions'/f'{sid}.json'
                if not resolution.exists():status='PENDING_CAMPAIGN_RESOLUTION'
                else:
                    proof=read_json(resolution)
                    if proof.get('exact_or_near_duplicate') is not False or not proof.get('actual_comparison_inspected') or set(proof.get('compared_source_ids',[]))!={r['source_video_id'] for r in row['duplicate_candidates']}:status='PENDING_CAMPAIGN_RESOLUTION'
                    for name,h in proof.get('evidence_sha256',{}).items():
                        if digest(Path(name))!=h:raise ValueError('Changed campaign comparison evidence')
        if status=='RECOVERED_ACCEPTED_AI_PSEUDO':
            if sid in prior_ids or row['sha256'] in prior_hashes:raise ValueError('Never re-count an original18 accepted source')
            variant=selected['variant'];pred=selected['prediction'];event=events[selected['query']];review=supplied[selected['query']]
            index=(run/'visual_index' if variant in ('B3','B4') else run/'sparse_index')/sid
            target=cfg.visual_cache/sid;target.mkdir(parents=True,exist_ok=True)
            immutable_write(target/'metadata.json',(index/'metadata.json').read_bytes())
            if not (target/'clip_features.npz').exists():os.link(index/'clip_features.npz',target/'clip_features.npz')
            image=Path(grids[0]['path']);semantic=folder/(variant+'_prediction.json')
            record={'qid':'p21_'+sid+'_'+stage,'source_video_id':sid,'source_video_path':Path(row['path']).relative_to(R).as_posix(),
                'source_video_sha256':row['sha256'],'source_url':row['source_url'],'duration':row['duration'],'query':selected['query'],
                'window':[pred['start'],pred['end']],'query_sanitation':{'passed':True,'errors':[]},
                'approximate_event_phase_seconds':[event['coarse_start'],event['coarse_end']],
                'visual_evidence':{'contact_sheets':[row['overview_path']]+[g['path'] for g in grids],'actual_complete_source_and_boundary_pages_viewed':True},
                'visual_index_sha256':digest(target/'metadata.json'),
                'verifier':{'pass':True,'notes':review['reason'],'human_verified':False,'model_id':'Codex multimodal image inspection','model_revision':'NOT_EXPOSED',
                    'onset_bracket_seconds':review['onset_bracket_seconds'],'offset_bracket_seconds':review['offset_bracket_seconds'],
                    'review_artifact_sha256':digest(destination),'review_blinding':'UNBLINDED; separate uniform query-only procedure recorded before localization'},
                'verification_image_path':image.relative_to(R).as_posix(),'verification_image_sha256':digest(image),
                'semantic_artifact_path':semantic.relative_to(R).as_posix(),'semantic_artifact_sha256':digest(semantic),
                'semantic_v3_prediction_sha256':digest(semantic),'semantic_v3_algorithm_sha256':pred['semantic_v3_algorithm_sha256'],
                'boundary_wrapper_version':pred['boundary_wrapper_version'],'selected_ablation_variant':variant,
                'query_source':'AI_GENERATED','query_review_status':'AI_AUTO_ACCEPTED_FOR_TRAINING','temporal_label_source':'SEMANTIC_V3_PSEUDO_LABEL',
                'review_provenance':'AI_PSEUDO_LABELED','reviewer_id':'AI_PIPELINE','query_provider_config':{'provider':'Actual Codex visual event discovery','model_revision':'NOT_EXPOSED','pipeline':'annotation_v4_1_experimental'},
                'semantic_confidence':pred['semantic_confidence'],'human_verified':False,'reviewed_interval':[review['onset_bracket_seconds'][1],review['offset_bracket_seconds'][0]],
                'campaign_group_id':'campaign_provisional_'+sid,'split_eligibility':'RESTRICTED_PENDING_CAMPAIGN_GROUP_REVIEW','fresh_test_eligible':False,
                'manual_ai_adjudication':True,'automatic_high_accept':False,'p21_index_root':str(cfg.visual_cache),'p21_stage':stage}
            validate_annotation(record,root=R,visual_cache=cfg.visual_cache)
            if sid not in builder.accepted:builder.prepare_existing(sid,Path(row['path']));builder.accept(record)
            else:
                if builder.accepted[sid]!=record:raise ValueError('Accepted source revision requires a new version')
            store=ReviewStore(run/'human_review',stage);store.register(record['qid'],'lumae_ads_'+sid,record['query'],record['duration'],record['window'])
        decision={'source_id':sid,'status':status,'selected_variant':selected['variant'] if selected else None,
            'variants':variant_decisions,'human_verified':False,'new_dataset_frozen':False,'campaign_independence_proven':False,
            'review_path':str(destination),'review_sha256':digest(destination)}
        immutable_write(folder/'acceptance_decision.json',json_bytes(decision));decisions.append(decision)
        print(sid,status,flush=True)
    immutable_write(run/(path.stem+'_import.json'),json_bytes({'reviews_path':str(path),'reviews_sha256':digest(path),'decisions':decisions,'accepted_in_builder':len(builder.accepted),'human_verified':0}))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['sheets','import']);ap.add_argument('--stage',required=True);ap.add_argument('--reviews',type=Path);a=ap.parse_args()
    sheets(a.stage) if a.command=='sheets' else import_reviews(a.stage,a.reviews)
