"""Seal the actual partial acquisition result, never a padded D1000 dataset."""
import csv
import json
import shutil
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from p2_controlled import pilot,accepted_count
from p16_recovery import O,R,OLD,VERSION
from data_process.lumae_scale.models import ROOT,read_json,atomic_json,digest
from data_process.lumae_scale.incremental import immutable_write,json_bytes

def csvout(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def jsonl(path,rows):
    immutable_write(path,''.join(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n' for r in rows).encode('utf-8'))

def deliver():
    if (O/'p2_execution_manifest.json').exists():raise PermissionError('Execution checkpoint already sealed; validate read-only or create a new version')
    assert read_json(O/'p2_offline_validation_final.json')['pass']
    assert read_json(O/'historical_final_validation.json')['pass']
    assert read_json(O/'p2_storage_feature_validation.json')['pass']
    p=pilot();records=p.records();reviews={r['source_id']:r for r in [read_json(f) for f in (p.work/'annotation_decisions').glob('*.json')]}
    previous=[json.loads(l) for l in (O/'pilot30_accepted_pairs.jsonl').read_text(encoding='utf-8').splitlines()]
    current=[read_json(f) for f in (R/'work/p1_p2_20261103_01/accepted').glob('*.json')]
    accepted=previous+current;assert len(accepted)==accepted_count()==18
    parent_ids={r['source_video_id'] for r in p.previous} # Original30, all states excluded.
    assert not parent_ids & {r['source_video_id'] for r in records}
    assert not p.parent_hashes & {r['source_video_sha256'] for r in current}
    for field in ('qid','source_video_id','source_video_sha256'):assert len({r[field] for r in accepted})==len(accepted)
    count=Counter(r['decision'] for r in reviews.values());assert len(reviews)==len(records)==50
    jsonl(O/'accepted_new_pairs.jsonl',accepted)
    jsonl(O/'p2_accepted_pairs.jsonl',current)
    pair_index=[]
    for r in accepted:
        observed=r.get('recovery_observed_event_interval',r.get('reviewed_interval'))
        pair_index.append({'source_id':r['source_video_id'],'query_id':r['qid'],'query':r['query'],'duration':r['duration'],
            'raw_model_start':r['window'][0],'raw_model_end':r['window'][1],'ai_observed_start':observed[0],'ai_observed_end':observed[1],
            'source_sha256':r['source_video_sha256'],'source_url':r['source_url'],'source_path':str(R/r['source_video_path']),
            'review_provenance':r['review_provenance'],'campaign_group':r['campaign_group_id'],'human_verified':False,
            'verification_image':str(R/r['verification_image_path'])})
    csvout(O/'accepted_pair_index.csv',pair_index)
    jsonl(O/'p2_rejected_and_pending_pairs.jsonl',[r for r in reviews.values() if r['decision']!='ACCEPTED_AI_PSEUDO'])
    source_rows=[]
    for r in records:
        review=reviews[r['source_video_id']]
        source_rows.append({'source_id':r['source_video_id'],'source_url':r['source_url'],'path':str(R/r['path']),'bytes':r['bytes'],'sha256':r['sha256'],
            'technical_pass':r['technical']['pass'],'source_state':r['status'],'annotation_state':r['annotation_status'],'final_decision':review['decision'],
            'annotation_accepted':int(review['decision']=='ACCEPTED_AI_PSEUDO'),'campaign_status':r.get('campaign_status'),
            'fingerprint_warnings':len(r.get('duplicate_candidates',[])),'fresh_test_eligible':False,'human_verified':False})
    csvout(O/'p2_source_annotation_inventory.csv',source_rows)
    # Every candidate proposal remains separately identifiable; superseded != accepted.
    proposals=[]
    for path in sorted((p.work/'evidence').rglob('proposal.json')):
        sid=path.relative_to(p.work/'evidence').parts[0];v=read_json(path);revision=int(path.parent.name) if path.parent.parent.name=='revisions' else 1
        selected=reviews[sid].get('revision',1)==revision
        proposals.append({'source_id':sid,'revision':revision,'query':v['query'],'start':v['start'],'end':v['end'],
            'path':str(path),'sha256':digest(path),'selection_status':reviews[sid]['decision'] if selected else 'SUPERSEDED_NOT_ACCEPTED',
            'final_review_reason':reviews[sid]['reason'],'human_verified':False})
    jsonl(O/'p2_query_revision_inventory.jsonl',proposals)
    waves=[read_json(f) for f in sorted((p.work/'wave_checkpoints').glob('*_review.json'))]
    summary=[];prior_bytes=0
    for w in waves:
        cumulative=w['global_crawler_metrics']['total_bytes_downloaded']
        summary.append({'wave':w['wave'],'attempted':len(w['new_source_ids']),'downloaded':len(w['new_source_ids']),'technical_valid':len(w['new_source_ids']),
            'accepted_pairs':w['accepted_in_wave'],'rejected_annotations':w['rejected_annotations'],'pending_campaign':w['pending_campaign_comparison'],
            'download_bytes':cumulative-prior_bytes,'cumulative_download_bytes':cumulative,'annotation_inference_seconds':w['annotation_inference_seconds'],
            'annotation_wall_seconds':w['annotation_wall_seconds'],'remaining_new_pairs':w['remaining_new_pairs'],'candidate_pool_remaining':w['candidate_pool_remaining']})
        prior_bytes=cumulative
    csvout(O/'p2_wave_progress.csv',summary)
    capacity=read_json(O/'p2_capacity_assessment.json');assert capacity['stop'] and not capacity['budget_actually_exhausted']
    storage=read_json(O/'p2_storage_feature_validation.json')
    confidence=Counter(r['semantic_confidence'] for r in accepted)
    inventory={'version':VERSION,'historical_parent_version':'lumae_ads_d500','historical_parent_pairs':500,'historical_splits':{'train':440,'val':30,'test':30},
        'historical_parent_dataset_sha256':p.parent['dataset_sha256'],'historical_parent_manifest_sha256':digest(p.build.parent_dir/'dataset_manifest.json'),'accepted_unique_new_pairs':18,'accepted_unique_new_sources':18,
        'accepted_pilot_pairs':7,'accepted_p2_pairs':11,'accepted_total_pool':518,'remaining_new_pairs':482,
        'pilot_raw_sources':30,'p2_raw_sources':50,'additional_raw_sources_total':80,'technically_valid_additional_sources':79,
        'technically_valid_retained_without_accepted_pair':61,'p2_annotation_rejected':33,'p2_campaign_comparison_pending':6,
        'p2_current_source_quarantine':39,'all_new_accepted_pairs_split_restricted':True,'human_verified_new_pairs':0,'human_IAA':None,
        'acquisition_complete':False,'final_d1000_freeze':False,'fresh_test_human_query_moment_labels':0,'required_fresh_test_labels':300,
        'required_test_double_annotation_fraction':.10,'evaluation_ready':False,'canonical_1000_dataset_created':False,
        'legacy_d500_feature_cache_changed':False,'new_training_features_generated':False,'annotation_only_clip_index_count':storage['clip_indexes_checked'],
        'semantic_confidence_distribution':dict(confidence),'stop_reason':'PROJECTED_ANNOTATION_THROUGHPUT_EXCEEDS_REMAINING_RUNTIME'}
    atomic_json(O/'d1000_acquisition_inventory.json',inventory)
    testcases={};test_runs={}
    for name in ('regression_final.xml','p2_capacity_tests.xml'):
        cases=list(ET.parse(O/name).getroot().iter('testcase'));assert all(c.find('failure') is None and c.find('error') is None for c in cases)
        for c in cases:testcases[(c.get('classname'),c.get('name'))]=True
        test_runs[name]=len(cases)
    assert len(testcases)==318
    atomic_json(O/'final_validation_summary.json',{'pass':True,'test_runs':test_runs,'distinct_tests_passed':len(testcases),
        'historical_readable_files_checked':44387,'historical_sha_preflight_pass':True,'historical_final_metadata_inventory_pass':True,
        'frozen_p14_and_semantic_v3_pass':True,'accepted18_strict_contract_pass':True,'source50_reprobe_pass':True,
        'network_denied_resume_pass':True,'resume_state_sha_unchanged':True,'candidate_selection_deterministic':True,
        'compileall_pass':True,'git_diff_check_pass':True,'capacity_stop_enforcement':read_json(O/'p2_stop_gate_validation.json')})
    wave_table='\n'.join(f"| {w['wave']} | {w['downloaded']} | {w['accepted_pairs']} | {w['rejected_annotations']} | {w['pending_campaign']} | {w['download_bytes']:,} | {w['remaining_new_pairs']} |" for w in summary)
    report=f'''# P1.6 / P2 — Kết quả thực thi

## A. Executive summary

**D1000 PARTIALLY COMPLETE — BLOCKED.** Pool hiện có **500 historical +18 new =518 cặp**, chưa tạo dataset1000 hoặc freeze split mới. Còn thiếu **482 cặp**. P1.6 engineering gate PASS; mốc20 không được dùng làm hard gate. Đã thực thi hai wave live liên tiếp, không dừng ở kế hoạch/pilot:50/50 video mới tải thành công và qua kỹ thuật,11 cặp P2 đạt contract;7 cặp pilot được giữ sau audit. Không có critical integrity/permission failure.

## B. Pilot30 final recovery

30/30 source được đối soát SHA và probe,29 hợp lệ.16/16 annotation cũ được xem lại overview toàn video và ảnh ranh giới2FPS:giữ7, loại9 trong phiên bản mới.13/13 nguồn thất bại được thử sự kiện/query khác qua pipeline nguyên trạng;0 phương án đạt kiểm tra cuối. Rejection151.658s vẫn giữ, không trim. Raw proposal, envelope cũ, observed AI interval và lý do từng source nằm trong `pilot30_temporal_reviews.jsonl` / CSV. Bản16 accepted cũ không bị sửa.

9 artifact P1.6 bắt buộc đã được tạo, đóng SHA trong `pilot30_recovery_freeze_manifest.json`; đây là freeze recovery, không phải D1000 freeze. Lineage gắn P1 và freeze P1.4/P1.5 `{digest(OLD/'pilot30_freeze_manifest.json')}`.

## C. Annotation quality / provenance

P2 có{len(proposals)} proposal/query revision trên34 source;16 source được loại sớm hoặc giữ pending sau actual overview.11 cặp cuối được kiểm tra actual dense QC rồi qua `validate_annotation` và BuilderSession. Không chép interval AI sửa tay vào output V3. Semantic confidence ghi nguyên trạng:{dict(confidence)}; acceptance có manual AI adjudication, không giả thành automatic HIGH confidence.

Query được sửa khi hình ảnh chứng minh sai:gel thay cream; một người đàn ông ngồi dậy thay cả hai người; bỏ mục đích suy diễn start work. Cảnh chắp tay không được gọi là hôn; dắt xe không được gọi là đạp xe. Cặp nhận vẫn là AI-pseudo training pool, không phải publication-grade GT.

**Hạn chế blind review:**numerical retrieval ban đầu không nhận scalar bounds nhưng dùng sampling dày theo phase của generator, vì vậy chỉ partially blinded. Đã bổ sung uniform0,2,4,… full-source query-only crosscheck cho47 proposal; đây là **post hoc**, không đổi tên thành blind-first. Cùng CLIP weights không phải độc lập mô hình. Re-audit16 cặp cũ là unblinded. Actual Codex multimodal interface không cung cấp model revision; không bịa tên version. Human verified0; human IAA=null. Xem `review_procedure_limitations.json`.

## D. Duplicate / campaign

Exact source/hash/URL duplicate được chặn bởi crawler và builder;18 accepted source/hash/qid duy nhất, không trùng D500. Đã bổ sung fingerprint triage với original pilot trong reference set.6 nguồn P2 có59 candidate fingerprint relations vẫn UNKNOWN và không được tính accepted. Các cảnh báo chỉ là triage, không phải bằng chứng duplicate hoặc độc lập campaign.

Quyền nguồn được ghi OPERATOR_ATTESTED theo xác nhận trước của user và scope P2 hiện tại; không tuyên bố đã kiểm chứng giấy phép độc lập hoặc quyền redistribution. Chỉ dùng HTTPS allowlist hiện có.

11 accepted P2 dùng RETAINED_SPLIT_RESTRICTED, relationship UNKNOWN;7 pilot cũng giữ hạn chế. Không tự đánh dấu DISTINCT, không one-video-per-brand, không freeze partition.3 flagged parent/pilot relationships của P14 được giữ với bằng chứng nội dung đã audit và SHA nguyên trạng; suy luận campaign từ branding không được nâng thành publisher confirmation. Fresh Test vẫn cần300 genuine human labels và10% double annotation;0 nhãn mới đạt điều kiện đó.

## E. Per-wave progress

| Wave | Tải/tech valid | Accepted pairs | Annotation rejected | Campaign pending | Byte tải riêng wave | Còn thiếu |
|---|---:|---:|---:|---:|---:|---:|
{wave_table}

Concurrency1,spacing≥2s,attempts≤2/source,maxfile100MiB,duration≤150s.50 HTTP requests,0 failed requests;951,401,435 byte tải. Không tải lại original30. State/attempt counters và budget clock dùng chung một run qua cả hai wave; không reset ngân sách theo batch. Wave checkpoints immutable đặt tại `{p.work/'wave_checkpoints'}`.6 pending không được cộng vào accepted.

## F. D1000 inventory

Historical500 giữ440/30/30 và manifest SHA cũ.18 cặp mới gồm7 pilot,5 ở wave1 và6 ở wave2; total inventory518,remaining482. Ngoài historical có80 raw source,79 technical-valid,1 duration rejection;61 technical-valid source chưa có accepted pair trong phiên bản hiện hành. Riêng P2:11 accepted,33 annotation rejected,6 campaign pending. Raw/source retention khác annotation acceptance;39 record P2 còn source-quarantined.

`accepted_new_pairs.jsonl` chứa18 record đầy đủ provenance;`accepted_pair_index.csv` liệt kê đủ source ID,query ID,query text,raw/reviewed interval,SHA,URL và evidence path. `p2_source_annotation_inventory.csv`, `p2_query_revision_inventory.jsonl`, `d1000_acquisition_inventory.json` cho số liệu chi tiết. Không tạo/pad canonical1000, không freeze700/150/150, không đánh giá Test mới.

## G. Storage / resource capacity

Ổ E còn{storage['disks']['E']['free']/2**30:.2f}GiB lúc kiểm tra; reserve15GiB vẫn đạt. P2 retained work+annotation-cache:{storage['p2_retained_bytes']:,}B. Logical checkpoint high-water:{storage['crawler_high_water_bytes']:,}B; số này bảo thủ vì check annotation có tính64MiB dự trù, không phải đo physical peak liên tục. Ngân sách giữ nguyên20GiB network/24GiB staging/+2GiB processing/8h/1500 requests. Không ZIP/migrate, không GPU training; E là working storage trên cùng physical disk theo audit trước, không phải backup độc lập. CloudZIP content vẫn UNVERIFIED theo hạn chế provider trước đó.

## H. Validation

Full isolated regression314/314 PASS; targeted capacity/gate/crawler31/31 PASS; hợp318 distinct tests. Compileall isolated bytecode và git diff --check PASS. Lượt đầu có3 lỗi do environment path trỏ assets thay P0 root; corrected runner4/4 PASS; lỗi fixture trùng gate filename đã sửa và8/8 PASS, không sửa historical data để làm test pass.

44,387 readable protected files đã qua full SHA preflight; final metadata/file inventory không changed/added. D48 vẫn AccessDenied như baseline, nên không tuyên bố kiểm tra được phần bị từ chối đọc. P14 frozen evidence/code/artifacts và frozen SemanticV3 kiểm tra lại PASS.50 P2 video reprobe/hash,18 accepted strict contract/unique IDs,actual cache shapes/finite vectors và network-denied resume PASS. Resume không đổi accepted ledger/review SHA, không HTTP và không tải lại original30. Legacy `pilot_pass` vẫn false do old accepted_target20; P2 dùng engineering gate P1.6 đã pin SHA, không giả true cho field cũ.

## I. Changes delivered

Targeted extensions ở acquisition:versioned engineering gate; prior-pilot fingerprint references; restricted UNKNOWN campaign retention; review qid giữ đúng accepted qid. Technical validator hiểu split-restricted state; test write guard bảo vệ thêm hai P14 versions. New modules/scripts:capacity forecast, P1.6 recovery/finalize, P2 bounded controller, cached annotation/revisions, uniform post-hoc review, offline validation và delivery. Không sửa frozen P14 producer scripts, V3 ranker, D500 feature cache/checkpoint hoặc thresholds để tăng yield. `changed_files.csv` liệt kê từng source/config/research artifact và SHA; scratch test/bytecode không được coi là dataset artifact. CLI và code SHA được đóng trong execution manifest.

## J. Blockers / limits

**Projected annotation throughput trong ngân sách8h không đủ**, theo điều kiện dừng tại mục7.4 của yêu cầu. Tại checkpoint dừng đã dùng{capacity['elapsed_seconds']/60:.1f} phút wall clock,còn{capacity['remaining_runtime_seconds']/3600:.2f} giờ. Dự báo lạc quan với Wilson95 upper yield{capacity['optimistic_yield_wilson_95_upper_including_pending']:.3f}, thậm chí coi cả6 pending là thành công, cần{capacity['optimistic_projected_seconds_to_target']/3600:.2f} giờ thêm để đạt mục tiêu. Nếu loại50% observed overhead,dự báo vẫn khoảng{capacity['optimistic_projected_seconds_to_target']/7200:.2f} giờ, vượt phần ngân sách còn lại.

Đây là **INFERRED capacity forecast**, không phải chứng minh mọi URL còn lại thất bại. Wall time gồm implementation/review overhead; các wave sau có thể nhanh hơn và thứ tự seed được giả định đại diện.1,037 URL chưa thử vẫn còn; chưa hết disk,byte budget hoặc8h. Lý do dừng không phải dưới20 hay thiếu human Test. Chưa có quy trình mới đã kiểm chứng đủ throughput trong scope, nên không mở wave3 hoặc tự tăng budget. Cần một assessment phiên bản mới dựa trên throughput cải thiện đã đo, hoặc explicit budget change trước live continuation; offline review có thể dùng lại toàn bộ raw/cache hiện có. Unknown campaign và genuine-human-Test cũng là blockers của evaluation freeze, không phải lý do chặn toàn bộ collection trước đó.

## K. Final status

**D1000 PARTIALLY COMPLETE — BLOCKED**

Acquisition inventory518/1000;18/500 new accepted;remaining482. Recovery và P2 partial checkpoint được niêm phong; không có claim D1000 acquisition complete hoặc research-grade evaluation ready.
'''
    (O/'p16_p2_final_report.md').write_text(report,encoding='utf-8')
    (O/'RESUME.md').write_text('''# Resume / validation

Set PYTHONDONTWRITEBYTECODE=1. All commands run from repository root.

Read-only real-data validation:
`python scripts/p2_validate.py --output research_outputs/NEW_VERSION/offline_validation.json`

The output parent must exist. No network is required; HTTP is explicitly denied.
The accepted builder, query revisions, cached features, source records, immutable
wave checkpoints and global budget_clock remain on E:. Original30 are bound by
previous_runs manifest SHA and never selected again.

The current run is stopped by the saved projected-capacity assessment. Do not
delete/reset its gate, counters or clock to bypass the authorized budget. A new
validated throughput assessment/version is required for live continuation; any
budget/domain expansion needs explicit authorization. Dataset1000 freeze is not
available with only18 new pairs, and AI labels cannot satisfy fresh human Test.
''',encoding='utf-8')
    code=[ROOT/'data_process/lumae_scale'/name for name in ('acquisition.py','pilot_quality.py','wave_capacity.py')]
    code += [ROOT/'scripts'/name for name in ('p16_recovery.py','p16_finalize.py','p2_controlled.py','p2_annotate.py','p2_uniform_review.py','p2_validate.py','p2_delivery.py')]
    code += [ROOT/'data_process/tests'/name for name in ('conftest.py','test_p16_gate.py','test_wave_capacity.py')]
    existing_modified={'acquisition.py','pilot_quality.py','conftest.py'}
    changes=[{'change':'MODIFIED_EXISTING' if f.name in existing_modified else 'NEW','path':str(f),'sha256':digest(f),'bytes':f.stat().st_size,'kind':'SOURCE_OR_TEST'} for f in code]
    changes.extend({'change':'NEW','path':str(f),'sha256':digest(f),'bytes':f.stat().st_size,'kind':'RESEARCH_ARTIFACT'} for f in O.iterdir() if f.is_file())
    for folder in (p.work,R/'work/p1_p2_20261103_01',R/'cache/p1_p2_20261103_01',R/'work/d1000/p16_p2_20261103_01'):
        changes.extend({'change':'NEW','path':str(f),'sha256':digest(f),'bytes':f.stat().st_size,'kind':'ACQUISITION_OR_ANNOTATION_EVIDENCE'} for f in folder.rglob('*') if f.is_file())
    for name in ('changed_files.csv','p2_execution_manifest.json','p2_execution_manifest.sha256'):
        changes.append({'change':'NEW','path':str(O/name),'sha256':'SEE_EXECUTION_MANIFEST_OR_ITS_SIDECAR','bytes':None,'kind':'SEAL_METADATA_SELF_REFERENCE_EXCLUDED'})
    csvout(O/'changed_files.csv',changes)
    manifest={'version':VERSION,'status':'D1000 PARTIALLY COMPLETE — BLOCKED','final_d1000_freeze':False,
        'accepted_unique_new_pairs':18,'accepted_total_pool':518,'remaining_new_pairs':482,
        'pilot_recovery_freeze_sha256':digest(O/'pilot30_recovery_freeze_manifest.json'),
        'artifact_sha256':{f.name:digest(f) for f in O.iterdir() if f.is_file() and f.name not in ('p2_execution_manifest.json','p2_execution_manifest.sha256')},
        'processing_code_sha256':{str(f):digest(f) for f in code},
        'p2_work_and_cache_sha256':{str(f):digest(f) for folder in (p.work,R/'work/p1_p2_20261103_01',R/'cache/p1_p2_20261103_01') for f in folder.rglob('*') if f.is_file()},
        'human_verified':0,'human_IAA':None,'capacity_blocker_sha256':digest(O/'p2_capacity_assessment.json')}
    path=O/'p2_execution_manifest.json';immutable_write(path,json_bytes(manifest));immutable_write(path.with_suffix('.sha256'),(digest(path)+'\n').encode())
    print(json.dumps({'manifest':str(path),'sha256':digest(path),'accepted_new':18,'remaining':482,'distinct_tests':len(testcases)},ensure_ascii=True))

if __name__=='__main__':deliver()
