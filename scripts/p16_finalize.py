"""Reconcile and seal new recovery decisions without altering any old record."""
import argparse
import csv
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from p16_recovery import O,E,R,OLD,A,B,VERSION,ALTERNATIVES,PRIOR_REVIEWS,inventory
from data_process.lumae_scale.models import atomic_json,read_json,digest
from data_process.lumae_scale.incremental import immutable_write,json_bytes
from data_process.lumae_scale.freeze import validate_annotation
from data_process.lumae_scale.pilot_quality import validate_video
from data_process.annotation.assisted_deployment import verify_frozen_ranker

ALT_REVIEWS={
1:([21,22],'REJECTED_BOUNDARY_UNRELIABLE','Duet only21-22; raw19.5-22.5 spans ensemble and two other performers.'),
2:([24.5,27.5],'REJECTED_AMBIGUOUS_QUERY','Walking montage changes child to adults; raw23-28 also includes outdoor group and logo.'),
3:([25,27.5],'REJECTED_INSUFFICIENT_EVIDENCE','Car is already on ramp and subsequently burns; evidence does not establish that it is being pulled.'),
4:([20,24.5],'REJECTED_AMBIGUOUS_QUERY','Several armored characters and repeated combat shots prevent a unique onset/offset for the generic swing query.'),
6:([13.5,16.5],'REJECTED_BOUNDARY_UNRELIABLE','Worker-panel transition begins13-13.5; raw12.5-16.5 includes preceding text animation.'),
10:([11,13.5],'REJECTED_BOUNDARY_UNRELIABLE','Raw9-13.5 begins with stadium running and exercise before newspaper shot11.'),
11:([31,33],'REJECTED_SEMANTIC_MISMATCH','Boy retrieves cookies from a dark cabinet; refrigerator identity unsupported and raw29-34 includes dog/photo/clock.'),
19:([14,20.5],'REJECTED_AMBIGUOUS_QUERY','Sustained couch/TV description lacks distinct action boundaries; raw starts after shot onset and ends in approach shot.'),
23:([34.5,35.5],'REJECTED_BOUNDARY_UNRELIABLE','Lipstick visible34.5-35.5; raw33-36 includes men facing one another and a separate couple.'),
24:([32.5,34],'REJECTED_AMBIGUOUS_QUERY','Sleeping is a sustained scene description; raw31-34 includes unrelated illuminated road structure.'),
25:([25.5,29.9],'REJECTED_BOUNDARY_UNRELIABLE','Two pub groups raise drinks, separated by cut; raw24-30 begins with product display and entrance shot.'),
27:([5,6.5],'REJECTED_BOUNDARY_UNRELIABLE','Paramedic walking is visible5-6.5; raw3.5-6 starts with a different group and truncates action.'),
28:([23,27],'REJECTED_BOUNDARY_UNRELIABLE','Ambulance recedes23-27; raw22-28 includes doctor close-up and foundation end card.'),
}

def jsonl(name,rows):
    immutable_write(O/name, ''.join(json.dumps(v,ensure_ascii=False,sort_keys=True)+'\n' for v in rows).encode('utf-8'))

def prepare():
    if (O/'pilot30_recovery_freeze_manifest.json').exists():raise PermissionError('Already sealed')
    verify_frozen_ranker()
    previous=read_json(OLD/'pilot30_freeze_manifest.json')
    old={v['source_video_id']:v for v in previous['accepted_annotations']}
    rows=inventory();assert len(rows)==len({v['source_id'] for v in rows})==30
    accepted=[];rejected=[];reviews=[];reconciliation=[];queries=[];probes=[]
    for n,r in enumerate(rows,1):
        sid=r['source_id'];assert digest(Path(r['path']))==r['sha256']
        technical=validate_video(Path(r['path']));probes.append({'source_id':sid,**technical})
        assert technical['pass']==(n!=22)
        if n==22:
            decision='REJECTED_DURATION';reason='151.658s exceeds frozen150s limit; no authorized extraction policy; no trimming.'
            rejected.append({'source_id':sid,'decision':decision,'reason':reason})
        else:
            if n in PRIOR_REVIEWS:
                passed,observed,reason=PRIOR_REVIEWS[n];record=old[sid]
                validate_annotation(record,root=R,visual_cache=B/'visual_index')
                decision='ACCEPTED_AI_PSEUDO' if passed else 'SUPERSEDED_QUALITY_EXCLUDED'
                raw=record['window'];query=record['query'];qid=record['qid']
                image=(A/f'pilot30_{n:02d}_boundary2.jpg') if n in (5,7,16,17) else (B/f'pilot30_{n:02d}_qc.jpg')
                semantic=R/record['semantic_artifact_path']
                blind={'procedure':'Previously frozen query-only CLIP estimate retained in P14 temporal review; this audit is unblinded visual reassessment','distinct_model':False}
                if passed:accepted.append({**record,'recovery_version':VERSION,'recovery_observed_event_interval':observed,'recovery_review_reason':reason})
            else:
                out=E/'annotations'/sid;p=read_json(out/'proposal.json');raw=[p['prediction']['start'],p['prediction']['end']]
                observed,decision,reason=ALT_REVIEWS[n];query=p['query'];qid='p16_'+sid+'_1';image=E/f'pilot30_{n:02d}_alternative_qc.jpg';semantic=out/'proposal.json';blind=read_json(out/'blind_first.json')
                queries.append({'qid':qid,'source_id':sid,'query':query,'alternative_to_prior_failed_query':True,'prior_decision':'REJECTED_BOUNDARY_UNRELIABLE','generation_input_path':str(out/'generation_input.json'),'generation_input_sha256':digest(out/'generation_input.json'),'raw_prediction_path':str(semantic),'raw_prediction_sha256':digest(semantic),'human_verified':False})
            review={'qid':qid,'source_id':sid,'query':query,'decision':decision,'raw_semantic_proposal':raw,'prior_accepted_contextual_envelope':old[sid]['window'] if sid in old else None,'observed_event_interval':observed,'intervals_are_separate':True,'reason':reason,'full_overview_path':str(A/f'pilot30_{n:02d}_overview.jpg'),'full_overview_sha256':digest(A/f'pilot30_{n:02d}_overview.jpg'),'qc_path':str(image),'qc_sha256':digest(image),'raw_artifact_path':str(semantic),'raw_artifact_sha256':digest(semantic),'blind_first':blind,'reviewer_interface':'Codex multimodal image inspection','model_revision':'NOT_EXPOSED','independent_model':False,'human_verified':False,'human_IAA':None,'sampling_fps':2}
            reviews.append(review)
            if decision!='ACCEPTED_AI_PSEUDO':rejected.append(review)
        reconciliation.append({'candidate_id':r['candidate_id'],'source_id':sid,'path':r['path'],'sha256':r['sha256'],'technical_pass':technical['pass'],'prior_accepted':sid in old,'final_status':decision,'accepted_pairs':int(decision=='ACCEPTED_AI_PSEUDO'),'reason':reason,'campaign_status':'UNKNOWN_SPLIT_RESTRICTED' if n!=22 else 'NOT_ELIGIBLE','fresh_test_eligible':False})
        print(n,decision,flush=True)
    assert len(reviews)==29 and len(queries)==13 and len(accepted)==7
    for record in accepted:validate_annotation(record,root=R,visual_cache=B/'visual_index')
    jsonl('pilot30_accepted_pairs.jsonl',accepted);jsonl('pilot30_rejected_pairs.jsonl',rejected);jsonl('pilot30_query_revisions.jsonl',queries);jsonl('pilot30_temporal_reviews.jsonl',reviews)
    with (O/'pilot30_final_reconciliation.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(reconciliation[0]));w.writeheader();w.writerows(reconciliation)
    atomic_json(O/'pilot30_technical_revalidation.json',probes)
    campaigns={'groups':read_json(OLD/'pilot30_campaign_manifest.json'),'triaged_flagged_relations':read_json(OLD/'campaign_relationships.json'),'exact_content_checks':'RawSHA reverified against original30 and frozen completeD500 baseline; no exact duplicates','unknown_policy':'UNKNOWN retained only as candidate training pool with group restriction; no claim of partition independence','fresh_test_eligible':False,'lineage_sha256':digest(OLD/'pilot30_freeze_manifest.json')}
    atomic_json(O/'pilot30_campaign_resolution.json',campaigns)
    metrics={'original_candidates':30,'downloaded_historical':30,'technically_valid':29,'technical_rejected':1,'previous_accepted':16,'reaudited':16,'retained_previous_pairs':7,'superseded_previous_pairs':9,'failed_sources_recovery_attempted':13,'recovered_pairs':0,'accepted_unique_new_pairs':len(accepted),'remaining_new_pairs':500-len(accepted),'source_retained_split_restricted':29,'human_verified':0,'human_IAA':None,'target20_quality_signal':False,'target20_is_hard_gate':False,'visual_cache':str(B/'visual_index'),'final_d1000_freeze':False}
    atomic_json(O/'pilot30_recovery_metrics.json',metrics)
    report='# P1.6 — Final Pilot30 Recovery\n\nĐã đối soát30/30 rawSHA và probe; rà soát lại16/16 cặp cũ bằng overview toàn video và ảnh2FPS. Giữ7, loại9 trong phiên bản mới; bản cũ giữ nguyên. Đã thử13 sự kiện/query thay thế qua CLIP/SemanticV3 nguyên trạng; chưa có cặp phục hồi đạt kiểm tra chất lượng. Giữ rejection151.658s.\n\nCác khoảng raw model, envelope cũ và quan sátAI được lưu riêng trong temporal_reviews. Cảnh tĩnh, ranh giới kéo sang shot khác và hành động chưa đủ bằng chứng đều được loại. Không sửa output model để hợp thức hóa interval.\n\nBlind-first là truy hồi query-only trước khi gọi localize, dùng cùng CLIP; đây là độc lập thủ tục, không phải mô hình độc lập hay humanGT/IAA.16 cặp cũ được re-audit có biết bounds, không được mô tả là blind. Model revision không được interface cung cấp.\n\nCampaign:29 nguồn giữ hạn chế split; quan hệ chưa biết không đổi thành DISTINCT.3 cặp fingerprint đã được xem và phân biệt nội dung trong P14, kết luận campaign vẫn là suy luận từ hình ảnh/nhãn hiệu, chưa có publisher registry. Không freeze split hoặc Test mới.\n\n7 cặp AI-pseudo đủ contract trong pool nghiên cứu; số20 chỉ là tín hiệu chất lượng. Engineering gate còn phụ thuộc regression, historical protection và freeze checksum. Chi tiết từng nguồn: CSV và temporal_reviews.\n'
    (O/'pilot30_recovery_audit.md').write_text(report,encoding='utf-8')

def seal():
    verify_frozen_ranker()
    assert read_json(O/'historical_preflight_validation.json')['pass']
    assert read_json(O/'test_summary.json')['pass']
    freeze={'version':VERSION,'final_d1000_freeze':False,'engineering_pass':True,'original_candidates':30,'reconciled_candidates':30,'prior_annotations_reaudited':16,'failed_sources_recovery_attempted':13,'original_p1_manifest_sha256':digest(R/'work/d1000/pilot_20261103_01/manifests/pilot_download_manifest.json'),'p14_freeze_sha256':digest(OLD/'pilot30_freeze_manifest.json'),'accepted_pairs':read_json(O/'pilot30_recovery_metrics.json')['accepted_unique_new_pairs'],'artifact_sha256':{p.name:digest(p) for p in O.iterdir() if p.is_file() and p.name.startswith('pilot30_')},'evidence_sha256':{str(p):digest(p) for p in E.rglob('*') if p.is_file()},'processing_code_sha256':{str(p):digest(p) for p in (Path(__file__),Path(__file__).with_name('p16_recovery.py'))},'human_verified':0,'human_IAA':None}
    path=O/'pilot30_recovery_freeze_manifest.json';immutable_write(path,json_bytes(freeze));immutable_write(path.with_suffix('.sha256'),(digest(path)+'\n').encode())
    atomic_json(O/'engineering_gate.json',{'schema':'LUMAE_P16_ENGINEERING_V1','engineering_pass':True,'critical_failure':False,'reconciled_candidates':30,'prior_annotations_reaudited':16,'failed_sources_recovery_attempted':13,'freeze_path':str(path),'freeze_sha256':digest(path),'quality_signal_20_pairs':False})

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--freeze',action='store_true');a=ap.parse_args()
    seal() if a.freeze else prepare()
