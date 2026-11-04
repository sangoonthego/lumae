"""Prepare/revalidate a new pilot-only ledger; old P1 and D500 remain read-only."""
import argparse,csv,json,statistics,sys,shutil
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from data_process.lumae_scale.models import ROOT,read_json,atomic_json,digest
from data_process.lumae_scale.builder import BuildConfig,BuilderSession,normalized_url
from data_process.lumae_scale.freeze import validate_annotation
from data_process.lumae_scale.pilot_completion import validate_completion,freeze_completion,CompletionConfig
from data_process.lumae_scale.incremental import immutable_write,json_bytes
ROUND2='--round2' in sys.argv
VERSION='p14_p15_20261103_02' if ROUND2 else 'p14_p15_20261103_01'
O=ROOT/'research_outputs'/VERSION;E=Path('E:/LUMAE/work/d1000')/VERSION;C=E/'visual_index';R=Path('E:/LUMAE')
OLD_O=ROOT/'research_outputs/p14_p15_20261103_01';OLD_E=R/'work/d1000/p14_p15_20261103_01'
if ROUND2:
    for name in ('pilot30_inventory.csv','visual_qc_notes.json','visual_observations.json'):
        if not (O/name).exists():
            if (O/'pilot30_freeze_manifest.json').exists():raise FileNotFoundError('Frozen input missing; no automatic restoration')
            shutil.copyfile(OLD_O/name,O/name)
    for p in OLD_E.glob('*_parent_overview.jpg'):
        if not (E/p.name).exists():shutil.copyfile(p,E/p.name)
rows=list(csv.DictReader((O/'pilot30_inventory.csv').open(encoding='utf-8-sig')));notes=read_json(O/'visual_qc_notes.json');observations={x[0]:x for x in read_json(O/'visual_observations.json')}

def jsonl(name,items):
    data=b''.join((json.dumps(x,sort_keys=True,ensure_ascii=False,allow_nan=False)+'\n').encode() for x in items)
    path=O/name
    if '--repair-export-provenance' in sys.argv and path.exists() and path.read_bytes()!=data:
        if (O/'pilot30_freeze_manifest.json').exists():raise ValueError('Frozen export cannot be corrected in place')
        old=path.read_bytes();h=__import__('hashlib').sha256(old).hexdigest()
        immutable_write(O/'provenance_revisions'/(name+'.'+h+'.jsonl'),old)
        immutable_write(O/'provenance_revisions'/(name+'.'+h+'.correction.json'),json_bytes({'reason':'Align raw proposal/generation pointers with explicit query revision; labels and accepted record bytes unchanged','before_sha256':h,'after_sha256':__import__('hashlib').sha256(data).hexdigest()}))
        temporary=path.with_suffix('.repair.tmp');temporary.write_bytes(data);temporary.replace(path)
    else:immutable_write(path,data)

def csvout(name,items):
    with (O/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(items[0]));w.writeheader();w.writerows(items)

def prepare():
    if (O/'pilot30_freeze_manifest.json').exists():raise PermissionError('Frozen pilot: run pilot30_resume_validate.py for read-only verification')
    parent=read_json(ROOT/'local_data/datasets/lumae_ads_d500/dataset_manifest.json')
    parent_hashes={x['sha256'] for x in parent['source_video_sha256'].values()};base=[json.loads(l) for l in (ROOT/'local_data/datasets/lumae_ads_d500/lumae_ads_d500_all.jsonl').read_text().splitlines()]
    parent_ids={r['vid'].removeprefix('lumae_ads_') for r in base};urls={normalized_url(r['metadata']['source_url']) for r in base if r.get('metadata',{}).get('source_url')}
    assert len({r['sha256'] for r in rows})==30
    for r in rows:
        assert digest(Path(r['path']))==r['sha256'] and r['sha256'] not in parent_hashes and r['source_id'] not in parent_ids and normalized_url(r['source_url']) not in urls
    cfg=CompletionConfig(VERSION,'lumae_ads_d500',ROOT/'local_data/datasets/lumae_ads_d500',R)
    pool=read_json(ROOT/'local_data/raw/adsqa/source_manifest.json')['videos'];session=BuilderSession(cfg,pool)
    accepted=[];decisions=[];queries=[];proposals=[];reviews=[];reconciliation=[];groups=[]
    for n,r in enumerate(rows,1):
        sid=r['source_id'];valid=r['technical_pass']=='True';accepted_here=False
        if ROUND2 and n in (5,7,16,17):
            record=next(a for a in read_json(OLD_O/'pilot30_accepted_annotation_manifest.json')['annotations'] if a['source_video_id']==sid)
            session.prepare_existing(sid,Path(r['path']));session.accept(record);accepted.append(record)
            for name,target in [('pilot30_query_candidates.jsonl',queries),('pilot30_temporal_proposals.jsonl',proposals),('pilot30_ai_review_results.jsonl',reviews),('pilot30_annotation_decisions.jsonl',decisions)]:
                target.extend(v for v in [json.loads(l) for l in (OLD_O/name).read_text(encoding='utf-8').splitlines()] if v.get('source_id')==sid)
            groups.append(next(g for g in read_json(OLD_O/'pilot30_campaign_manifest.json') if g['source_id']==sid))
            reconciliation.append(next(v for v in csv.DictReader((OLD_O/'pilot30_final_reconciliation.csv').open(encoding='utf-8-sig')) if v['source_id']==sid))
            continue
        if not valid:
            reconciliation.append({'candidate_id':r['candidate_id'],'source_id':sid,'final_status':'REJECTED_TECHNICAL','source_accepted':False,'accepted_annotations':0,'reason':'DURATION_EXCEEDS_150_SECONDS','split_eligibility':'NONE','evidence':r['path'],'remaining_blocker':'Existing151.658s policy rejection retained'});continue
        out=E/'annotations'/sid;orig=read_json((OLD_E if ROUND2 else E)/'annotations'/sid/'proposal.json');proposal_path=out/'proposal.json';proposal=read_json(proposal_path) if ROUND2 else read_json(out/'proposal_revision2.json') if n==7 else orig
        if ROUND2 and n==30:
            failed=read_json(out/'proposal.json');failed_qid=f'p14_{sid}_2'
            queries.append({'qid':failed_qid,'source_id':sid,'query':failed['query'],'revision':2,'generation_evidence':str(out/'generation_input.json'),'generation_evidence_sha256':digest(out/'generation_input.json'),'human_verified':False})
            proposals.append({'qid':failed_qid,**failed,'raw_proposal_sha256':digest(out/'proposal.json')})
            decisions.append({'qid':failed_qid,'source_id':sid,'decision':'REJECTED_AMBIGUOUS_QUERY','reason':'Frozen sanitation blocked missing concrete subject phrase; grammar corrected in revision3, gate unchanged','human_verified':False})
        if ROUND2:
            for name in ('candidate_revision.json','query_revision.json','schema_revision.json'):
                if (out/name).exists():proposal_path=out/name;proposal=read_json(proposal_path)
        for revision,p in enumerate([orig,proposal] if ROUND2 or n==7 else [orig],1):
            if ROUND2 and n==30 and revision==2:revision=3
            genpath=(OLD_E/'annotations'/sid/'generation_input.json') if ROUND2 and revision==1 else proposal_path if ROUND2 and revision>=2 and 'event' in proposal else out/'generation_input.json'
            qid=f'p14_{sid}_{revision}';queries.append({'qid':qid,'source_id':sid,'query':p['query'],'generation_evidence':str(genpath),'generation_evidence_sha256':digest(genpath),'revision':revision,'human_verified':False})
            proposals.append({'qid':qid,**p,'raw_proposal_sha256':digest(proposal_path) if ROUND2 and revision>=2 else digest((OLD_E if ROUND2 else E)/'annotations'/sid/('proposal_revision2.json' if n==7 and revision==2 else 'proposal.json'))})
        qid=f'p14_{sid}_{3 if ROUND2 and n==30 else 2 if ROUND2 or n==7 else 1}';pred=proposal['prediction'];decision,reviewed,reason=notes[str(n)]
        if n in (5,7,16,17):decision='ACCEPTED_AI_PSEUDO'
        elif n==15:decision='REJECTED_BOUNDARY_UNRELIABLE';reason+=' Dense2FPS shows stage sequence from12.5 through17.5s, outside original13-17s.'
        elif n==29:decision='REJECTED_SEMANTIC_MISMATCH';reason='The visible device is a phone; query incorrectly asserts a laptop. Do not accept unsupported device identity.'
        if ROUND2:decision,reason=read_json(O/'round2_visual_decisions.json')[str(n)]
        if n==7:reviewed=[100.5,102.5];reason='Neutral revision2 removes unsupported pronoun; goggles are lifted100.5-102.5s, original99-103s covers full movement with surfacing context.'
        evidence=read_json(out/('qc.json' if ROUND2 else 'boundary_2fps.json'));window=[pred['start'],pred['end']]
        image=E/(f'pilot30_{n:02d}_qc.jpg' if ROUND2 else f'pilot30_{n:02d}_boundary2.jpg')
        blind=read_json(out/('blind_query_revision.json' if ROUND2 and (out/'query_revision.json').exists() else 'blind_first.json' if ROUND2 else 'blind_first_revision2.json' if n==7 else 'blind_first_estimate.json'))
        phase=observations[n][3:5];positive=[observations[n][5]]
        if ROUND2:
            inp=read_json(out/'generation_input.json');phase=inp['visible_phase'];positive=inp['positive_evidence_timestamps']
            if 'event' in proposal:phase=[proposal['event']['coarse_start'],proposal['event']['coarse_end']];positive=proposal['event']['evidence_timestamps']
        review={'qid':qid,'source_id':sid,'query':proposal['query'],'original_interval':window,'blind_first_estimate':blind,'visually_reviewed_interval':reviewed,'decision':decision,'reason':reason,'review_method':'Separate full-video CLIP query-only blind estimate followed by actual Codex visual adjudication on overview,1FPS and2FPS evidence','distinct_model':False,'independent_human':False,'human_verified':False,'human_IAA':None,'model_id':'Codex multimodal interface','model_revision':'NOT_EXPOSED','evidence_path':str(image),'evidence_sha256':digest(image),'sampling_fps':2,'limitation':'Same CLIP weights; procedural blind review, not an independently calibrated semantic model or human GT'}
        review['evidence_path']=str(image);review['evidence_sha256']=digest(image)
        review_path=out/('visual_adjudication_schema_revision.json' if ROUND2 and n==30 else 'visual_adjudication.json')
        immutable_write(review_path,json_bytes(review));reviews.append(review)
        if n==7:decisions.append({'qid':f'p14_{sid}_1','source_id':sid,'decision':'REJECTED_AMBIGUOUS_QUERY','reason':'Unsupported gendered pronoun; superseded by revision2, not silently overwritten'})
        if ROUND2:decisions.extend(v for v in [json.loads(l) for l in (OLD_O/'pilot30_annotation_decisions.jsonl').read_text(encoding='utf-8').splitlines()] if v['source_id']==sid)
        decisions.append({'qid':qid,'source_id':sid,'decision':decision,'raw_interval':window,'reviewed_interval':reviewed,'reason':reason,'human_verified':False})
        if decision=='ACCEPTED_AI_PSEUDO':
            semantic=out/('semantic_prediction_schema_revision.json' if ROUND2 and n==30 else 'semantic_prediction.json');immutable_write(semantic,json_bytes(pred))
            idx=read_json(C/sid/'metadata.json')
            times={'before':[x['timestamp'] for x in evidence if x['timestamp']<window[0]],'inside':[x['timestamp'] for x in evidence if window[0]<=x['timestamp']<window[1]],'after':[x['timestamp'] for x in evidence if x['timestamp']>=window[1]]}
            record={'qid':qid,'source_video_id':sid,'source_video_path':Path(r['path']).relative_to(R).as_posix(),'source_video_sha256':r['sha256'],'source_url':r['source_url'],'duration':float(r['duration']),'query':proposal['query'],'window':window,
                    'query_sanitation':{'passed':True,'errors':[]},'approximate_event_phase_seconds':phase,'visual_evidence':{'timestamps':positive,'contact_sheets':idx['contact_sheets'],'note':'Actual full overview and dense evidence inspected'},'visual_index_sha256':digest(C/sid/'metadata.json'),
                    'verifier':{'pass':True,'notes':reason,'evidence_timestamps':times,'model_id':'Codex multimodal visual adjudication','model_revision':'NOT_EXPOSED','human_verified':False,'review_artifact_sha256':digest(review_path),'video_start_no_before_context':window[0]==0},
                    'verification_image_path':image.relative_to(R).as_posix(),'verification_image_sha256':digest(image),'semantic_artifact_path':semantic.relative_to(R).as_posix(),'semantic_artifact_sha256':digest(semantic),'semantic_v3_prediction_sha256':digest(semantic),'semantic_v3_algorithm_sha256':pred['semantic_v3_algorithm_sha256'],
                    'query_source':'AI_GENERATED','query_review_status':'AI_AUTO_ACCEPTED_FOR_TRAINING','temporal_label_source':'SEMANTIC_V3_PSEUDO_LABEL','review_provenance':'AI_PSEUDO_LABELED','reviewer_id':'AI_PIPELINE','query_provider_config':{'provider':'actual Codex multimodal visual inspection','model_revision':'NOT_EXPOSED','CLIP':'ViT-B/32 CPU; weights hash in original inference evidence'},'semantic_confidence':pred['semantic_confidence'],'human_verified':False,'reviewed_interval':reviewed,'campaign_group_id':'campaign_provisional_'+sid,'split_eligibility':'RESTRICTED_PENDING_CAMPAIGN_GROUP_REVIEW','automatic_high_accept':False,'manual_ai_adjudication':True}
            session.prepare_existing(sid,Path(r['path']))
            session.accept(record);accepted.append(record);accepted_here=True
            from data_process.annotation.independent_review import ReviewStore
            store=ReviewStore(E/'human_review',VERSION)
            store.register(qid,'lumae_ads_'+sid,record['query'],record['duration'],record['window'])
            immutable_write(E/'human_review'/(qid+'_blind_packet.json'),json_bytes(store.blind_packet(qid)))
        groups.append({'source_id':sid,'campaign_group_id':'campaign_provisional_'+sid,'brand_observed':observations[n][1],'campaign_status':'PROVISIONAL_NOT_VERIFIED_INDEPENDENT','source_acceptance':'ACCEPTED_SPLIT_RESTRICTED','fresh_test_eligible':False,'restriction':'Do not freeze any partition before campaign registry resolution; distinct videos sharing a campaign must stay together'})
        reconciliation.append({'candidate_id':r['candidate_id'],'source_id':sid,'final_status':'ACCEPTED_SOURCE_AND_ANNOTATION' if accepted_here else 'REJECTED_ANNOTATION_QUALITY','source_accepted':True,'accepted_annotations':int(accepted_here),'reason':decision,'split_eligibility':'RESTRICTED_PENDING_CAMPAIGN_GROUP_REVIEW','evidence':str(out/'visual_adjudication.json'),'remaining_blocker':'Campaign registry unresolved; AI-only labels cannot become blind Test GT' if accepted_here else reason})
    jsonl('pilot30_query_candidates.jsonl',queries);jsonl('pilot30_temporal_proposals.jsonl',proposals);jsonl('pilot30_ai_review_results.jsonl',reviews);jsonl('pilot30_annotation_decisions.jsonl',decisions);jsonl('pilot30_accepted_annotations.jsonl',accepted);jsonl('pilot30_rejected_annotations.jsonl',[d for d in decisions if d['decision']!='ACCEPTED_AI_PSEUDO'])
    csvout('pilot30_final_reconciliation.csv',reconciliation)
    if not (O/'pilot30_inventory.at_start.csv').exists():shutil.copyfile(O/'pilot30_inventory.csv',O/'pilot30_inventory.at_start.csv')
    final_status={r['source_id']:r['final_status'] for r in reconciliation}
    for r in rows:r['status']=final_status[r['source_id']]
    csvout('pilot30_inventory.csv',rows)
    atomic_json(O/'campaign_group_manifest.json',groups);atomic_json(O/'pilot30_campaign_manifest.json',groups)
    atomic_json(O/'pilot30_accepted_source_manifest.json',{'source_acceptance_count':29,'source_and_annotation_pairs':len(accepted),'split_unrestricted_count':0,'sources':[{'source_id':r['source_id'],'sha256':r['sha256'],'path':r['path'],'split_restricted':True} for r in rows if r['technical_pass']=='True']})
    atomic_json(O/'pilot30_accepted_annotation_manifest.json',{'annotations':accepted,'human_verified':0,'purpose':'AI pseudo training pool only; not blind Test GT'})
    atomic_json(O/'storage_bindings.json',{'data_root':str(R),'visual_cache':str(cfg.visual_cache),'builder_work':str(cfg.work),'original_source_root':str(R/'work/d1000/pilot_20261103_01/downloads'),'read_only_parent':str(cfg.parent_dir),'original_p1_immutable':True})
    relationships=[]
    for a,b,reason in [('5e4e41d967149975220c2b6bab354c0c','47ad45ea18e1a859c8355efc27d44fc2','AmFam family/dance insurance ad versus Sharpie browser-recipe highlighter explainer'),('5e4e41d967149975220c2b6bab354c0c','aae60e7083aa64347cf079c69e98f1bc','AmFam family/dance insurance ad versus Jooble jobs montage with PPE workers'),('bf5accad69519c61b4ea27ee0d2a0136','dda6f2b2f59239c6c5a8fc6b9045c588','SPAR holiday/community montage versus Pepsi cola concert/dance ad')]:
        relationships.append({'a':a,'b':b,'relationship':'DISTINCT_CAMPAIGN','content_duplicate':False,'evidence':reason,'confidence':'HIGH_CONTENT_DISTINCT; CAMPAIGN_INFERRED','content_evidence_classification':'VERIFIED_VISUAL_DIFFERENCE','campaign_evidence_classification':'INFERRED_FROM_CONTENT_AND_BRANDING_NOT_PUBLISHER_REGISTRY','review_method':'Actual complete sampled visual sequences and branding inspected','parent_evidence':str(E/(b+'_parent_overview.jpg')),'parent_evidence_sha256':digest(E/(b+'_parent_overview.jpg'))})
    atomic_json(O/'campaign_relationships.json',relationships);csvout('duplicate_resolution.csv',[{'source_id':r['a'],'matched_id':r['b'],'relationship':r['relationship'],'duplicate':False,'evidence':r['evidence']} for r in relationships])
    result=validate_completion(rows,accepted,root=R,visual_cache=C);atomic_json(O/'pilot30_strict_validation.json',result)
    print(result)

ap=argparse.ArgumentParser();ap.add_argument('--freeze',action='store_true');ap.add_argument('--round2',action='store_true');ap.add_argument('--repair-export-provenance',action='store_true');args=ap.parse_args()
if not args.freeze:prepare()
else:
    import xml.etree.ElementTree as ET
    for name in ('regression.xml','recovery_validation.xml'):
        suites=list(ET.parse(O/name).getroot().iter('testsuite'))
        assert suites and all(int(s.get('failures',0))+int(s.get('errors',0))==0 for s in suites)
    accepted=[json.loads(l) for l in (O/'pilot30_accepted_annotations.jsonl').read_text(encoding='utf-8').splitlines()];result=validate_completion(rows,accepted,root=R,visual_cache=C)
    protected=read_json(O/'historical_final_validation.json');assert protected['pass'] and read_json(O/'pilot30_resume_validation.json')['pass']
    from data_process.annotation.assisted_deployment import ALGORITHM_SHA256,verify_frozen_ranker
    verify_frozen_ranker()
    pinned_json={'campaign_relationships.json','campaign_group_manifest.json','pilot30_campaign_manifest.json','pilot30_accepted_source_manifest.json','pilot30_accepted_annotation_manifest.json','pilot30_strict_validation.json','storage_bindings.json','round2_visual_decisions.json','visual_qc_notes.json','visual_observations.json','pilot30_resume_validation.json','historical_final_validation.json','provenance_alignment.json','metrics.json','regression_summary.json'}
    manifest={'version':VERSION,'original_p1_run_id':'20261103_01','original_p1_manifest_sha256':digest(R/'work/d1000/pilot_20261103_01/manifests/pilot_download_manifest.json'),'final_d1000_freeze':False,'original_candidates':30,'accepted_source_count':29,'source_annotation_pairs':result['accepted_annotations'],'accepted_annotations':accepted,'sources':rows,'reconciliation':list(csv.DictReader((O/'pilot30_final_reconciliation.csv').open(encoding='utf-8-sig'))),'annotation_decisions':[json.loads(l) for l in (O/'pilot30_annotation_decisions.jsonl').read_text(encoding='utf-8').splitlines()],'campaigns':read_json(O/'pilot30_campaign_manifest.json'),'human_verified':0,'human_IAA':None,'quality_pass':result['accepted_annotations']>=20,'decision':'NO-GO','model_identity':{'CLIP_weights_sha256':digest(ROOT/'local_data/cache/clip_weights/ViT-B-32.pt'),'semantic_v3_algorithm_sha256':ALGORITHM_SHA256,'visual_interface':'Codex multimodal interface','model_revision':'NOT_EXPOSED','distinct_independent_model_used':False},'processing_code_sha256':{str(p):digest(p) for p in [Path(__file__),ROOT/'scripts/pilot30_propose.py',ROOT/'scripts/pilot30_round2.py',ROOT/'scripts/pilot30_resume_validate.py',ROOT/'scripts/pilot30_review_setup.py',ROOT/'data_process/lumae_scale/pilot_completion.py']},'evidence_sha256':{str(p):digest(p) for folder in ([OLD_E,E,R/'work'/VERSION] if ROUND2 else [E,R/'work'/VERSION]) for p in sorted(folder.rglob('*')) if p.is_file()},'artifact_sha256':{p.name:digest(p) for p in sorted(O.iterdir()) if p.is_file() and (p.suffix in ('.csv','.jsonl') or p.name in pinned_json)},'historical_validation_sha256':digest(O/'historical_final_validation.json')}
    print(freeze_completion(O,manifest))
