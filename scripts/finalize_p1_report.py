"""Consolidate observed P1 evidence; no HTTP and no historical/E: mutations."""
from __future__ import annotations
import csv
from datetime import datetime,timezone
import json
import math
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import xml.etree.ElementTree as ET
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from data_process.lumae_scale.models import ROOT,read_json,atomic_json,digest
from data_process.lumae_scale.incremental import immutable_write,json_bytes

O=ROOT/'research_outputs/p1_20261103_01';W=Path('E:/LUMAE/work/d1000/pilot_20261103_01');C=Path('E:/LUMAE/cache/p1_20261103_01')
protected=read_json(O/'protected_asset_final_validation.json');validation=read_json(O/'pilot_integrity_validation_final.json');resume=read_json(O/'resume_validation.json')
if not protected['pass'] or not validation['integrity_pass'] or not resume['pass']:raise RuntimeError('Final integrity evidence failed')
manifest=read_json(W/'manifests/pilot_download_manifest.json');metrics=manifest['metrics'];rows=manifest['records'];valid=[r for r in rows if r.get('technical',{}).get('pass')]
for directory in ('manifests','annotations','reviews','reports'):
    for p in (W/directory).glob('*'):
        if p.is_file() and p.suffix in ('.json','.jsonl','.csv'):shutil.copyfile(p,O/p.name)
tests={}
for name in ('full_validation_fixed.xml','final_targeted_validation.xml','final_export_validation.xml'):
    suites=list(ET.parse(O/name).getroot().iter('testsuite'))
    tests[name]={k:sum(int(s.attrib.get(k,0)) for s in suites) for k in ('tests','failures','errors','skipped')}
    if tests[name]['failures'] or tests[name]['errors']:raise RuntimeError('Final test failure')
atomic_json(O/'validation_summary.json',tests)

inventory=[]
for root in (W,C):
    for p in sorted(root.rglob('*')):
        if p.is_file():inventory.append({'path':str(p),'bytes':p.stat().st_size,'sha256':digest(p)})
with (O/'pilot_asset_inventory.csv').open('w',encoding='utf-8-sig',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=['path','bytes','sha256']);writer.writeheader();writer.writerows(inventory)
sizes=[r['bytes'] for r in valid];mean=statistics.mean(sizes);ordered=sorted(sizes);p90=ordered[math.ceil(.9*len(ordered))-1]
cache_bytes=sum(p.stat().st_size for p in C.rglob('*') if p.is_file());free=shutil.disk_usage('E:/').free
proposals=[read_json(p) for p in (W/'annotations').glob('*_proposals.json')];index=read_json(W/'reports/visual_index_results.json')
flagged=[r for r in valid if r.get('duplicate_candidates')];edges=sum(len(r.get('duplicate_candidates',[])) for r in valid)
scenarios=[{'assumed_final_acceptance_probability':p,'expected_attempts':math.ceil(500/p),'expected_network_bytes_using_observed_valid_mean':math.ceil(500/p)*mean,'classification':'ASSUMED sensitivity; final acceptance rate not observed'} for p in (.25,.5,.75,.9)]
forecast={'decision':'NO-GO','measured':{'accepted_sources':0,'remaining_accepted_required':500,'valid_video_mean_bytes':mean,'valid_video_median_bytes':statistics.median(sizes),'valid_video_p90_bytes':p90,'technical_success_rate':len(valid)/len(rows),'network_success_rate':metrics['successful_downloads']/metrics['download_attempts'],'final_acceptance_rate':'UNOBSERVABLE_WHILE_REVIEW_PENDING; progress=0/30','e_free_bytes':free,'pilot_cache_bytes':cache_bytes,'pilot_cache_bytes_per_valid_video':cache_bytes/len(valid),'source_pool_metadata_eligible':1117,'already_attempted':30,'remaining_candidates_without_repeating_pilot':1087,'all_proposals':sum(len(p['proposals']) for p in proposals),'proposal_generation_seconds_one_real_video':proposals[0]['elapsed_seconds'],'mean_seconds_per_accepted_video':None},
          'conditional_storage_estimates':{'new_500_raw_mean_bytes':500*mean,'new_500_raw_p90_bytes':500*p90,'new_500_raw_100MiB_each_worst_case_bytes':500*100*2**20,'new_500_preprocessing_bytes':500*cache_bytes/len(valid),'annotation_allowance_bytes':500*128*2**10,'training_features_reference_estimate_bytes':25902016,'feature_estimate_basis':'ASSUMED same per-video rate as frozen D500 cache; new pilot training features not extracted','mean_scenario_final_e_free_bytes':free-(500*mean+500*cache_bytes/len(valid)+500*128*2**10+25902016),'minimum_e_reserve_bytes':15*2**30,'credit_for_quarantined_video_bytes':0},'attempt_scenarios':scenarios,'expected_attempts_from_empirical_acceptance':None,'warning':'Technical download yield does not establish campaign independence or temporal-label acceptance. Annotation estimates use one pending proposal, not a validated throughput distribution.'}
atomic_json(O/'d1000_capacity_forecast.json',forecast)
scale=read_json(O/'pilot_config.json');scale.update(run_id='d1000_scale_review_required_01',mode='scale',target_accepted=500,max_candidates=1087,max_http_requests=1500,max_download_bytes=20*2**30,max_staging_bytes=24*2**30,processing_reserve_bytes=2*2**30,max_runtime_seconds=8*3600,network_execution_authorized=False,pilot_pass_evidence=str(O/'pilot_integrity_validation_final.json'),previous_runs=[{'manifest':(W/'manifests/pilot_download_manifest.json').relative_to(Path('E:/LUMAE')).as_posix(),'sha256':digest(W/'manifests/pilot_download_manifest.json')}])
immutable_write(O/'scale_config.disabled.json',json_bytes(scale))
commands={'dry_run':'python -m data_process.lumae_scale.acquisition plan --config research_outputs/p1_20261103_01/scale_config.disabled.json --output <NEW_PLAN_PATH>',
          'gated_scale_command':'python -m data_process.lumae_scale.acquisition run --config research_outputs/p1_20261103_01/scale_config.disabled.json --enable-live-network --enable-scale-budget --max-wave 25',
          'current_result':'REFUSED: real pilot_pass=false and network_execution_authorized=false','required_before_execution':['Resolve campaign reviews for 29 quarantined sources; exclude confirmed duplicates','Acquire >=20 accepted source/annotation pairs under the unchanged acceptance rules','Complete actual temporal visual evidence; keep independent human states truthful','Passing real pilot validation, independent review backlog/annotation throughput and reserve verified','Operator explicitly approves larger budget via NEW config/run; rebind current prior-manifest SHA','Use waves of at most25; checkpoint manifests after every wave; no automatic limit increases'],'full_scale_started':False}
atomic_json(O/'scale_execution_plan.json',commands)
changed={
 'data_process/lumae_scale/acquisition.py':'NEW: bounded P1 CLI/collector, durable ledger, retry/resume/budgets, triage/review/import and prior-run bindings',
 'data_process/lumae_scale/pilot_quality.py':'NEW: real frame/container/timestamp checks, dHash triage and independent run validation',
 'data_process/lumae_scale/pilot_annotation.py':'NEW: offline actual-evidence CLIP/Semantic V3 proposal and independent blind review packet',
 'data_process/lumae_scale/README_P1.md':'NEW: executable commands, state/permission/resume contracts and scale gates',
 'data_process/adsqa/video_validation.py':'MODIFIED: explicit UTF-8 replacement decoding for Windows ffprobe metadata; old data unchanged',
 'data_process/lumae_scale/visual_index.py':'MODIFIED: optional explicit root/cache bindings; legacy defaults retained',
 'data_process/lumae_scale/query_generator.py':'MODIFIED: optional cache binding for existing evidence validator',
 'data_process/lumae_scale/temporal_labeler.py':'MODIFIED: optional cache binding; frozen ranker and boundary logic unchanged',
 'data_process/tests/conftest.py':'MODIFIED P0 file: additionally protect all audit_outputs and original P0 output directory',
 'data_process/tests/test_cached_runner.py':'MODIFIED P0 file: parity report goes to tmp_path instead of protected P0 assets',
 'data_process/tests/test_p1_acquisition.py':'NEW: isolated interruption, Range, budget, access, ledger, prior-binding and idempotency tests',
 'data_process/tests/test_p1_video_quality.py':'NEW: actual ffmpeg fixture decode/cache isolation and HTML rejection',
 'scripts/p1_preflight.py':'NEW: protected snapshots, D500/hash/storage/source/model audit and final full-SHA verification',
 'scripts/p1_network_preflight.py':'NEW: explicit small metadata network probe, no video body',
 'scripts/p1_model_smoke.py':'NEW: CPU CLIP and frozen Semantic V3 dependency smoke',
 'scripts/p1_prepare_run.py':'NEW: bind successful preflight/test evidence to reviewed pilot config',
 'scripts/p1_processing.py':'NEW: isolated source index and 500-video read-only fingerprint baseline',
 'scripts/p1_resume_check.py':'NEW: real-source completed resume with an HTTP-deny trap',
 'scripts/finalize_p1_report.py':'NEW: consolidate measured evidence, forecast, disabled scale plan and Vietnamese report'}
atomic_json(O/'p1_code_changes.json',[{'path':str(ROOT/k),'purpose':v,'sha256':digest(ROOT/k)} for k,v in changed.items()])
atomic_json(O/'git_after.json',{'revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'status':subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True),'commit_created':False})

report=f'''# LUMAE — P1 Controlled D1000 Data Acquisition

Ngày 03/11/2026, Asia/Saigon. Evidence: `D:\\DUT\\Se07\\CV\\lumae\\research_outputs\\p1_20261103_01`.
VERIFIED = đọc/đo/chạy thực tế; INFERRED = suy luận có nguồn; ASSUMED = kịch bản dự toán; TODO = chưa đạt điều kiện.

## A. Executive Summary

**P1 đã chạy pilot thật nhưng chưa PASS chất lượng; D1000 scale-up: NO-GO.**
30 candidate, 30 HTTP download thành công, 658.331.255 byte; 29 video qua kỹ thuật, 1 bị loại vì dài 151,658s >150s. **Accepted unique source videos = 0; accepted temporal annotations = 0.** 29 video quarantine chờ xác nhận campaign; không cộng thành D529/D1000. Có 1 proposal AI thật và blind packet chờ review, không có human verification/IAA.

P1.0 local preflight PASS theo xác nhận quyền nguồn của operator; P1.1 thực thi có hạn mức, target25 accepted chưa đạt; P1.2 integrity/resume PASS nhưng acceptance gate FAIL; P1.3 đã chuẩn bị forecast/config bị khóa. Không hạ tiêu chí để đạt số lượng.

## B. Operational Preflight

Git base `9a78badc0f695d78245e15ab30dfe6f1907c8cbf`. Đã đọc audit/P0, giữ các thay đổi chưa commit; `git_before.json` lưu diff/status. Không commit, crawl qua host khác, xóa/migrate lịch sử hoặc thay kết quả nghiên cứu.

VERIFIED: D500 checksum canonical/splits 440/30/30; đủ500 source SHA khác nhau, raw8.836.814.052B. ZIP SHA mới kiểm tra `132621746f9d2538d199e8c5cba1ddec0f0586854ae26c3628478de83ef19345`. M0 SHA `ad5facec03008800536466c7d33f9232c3db2b71a1bd445073957bd6f4d0edbf`; freeze P0/base hash hợp lệ. D200/E0 checkpoint và D500 ZIP trên Drive chỉ có metadata hiện tại: `cloud_checkpoint_metadata.json`; không gọi đó là content verification. Cloud ZIP SHA vẫn UNVERIFIED do giới hạn provider đã ghi ở P0.

Full SHA-256 trước/sau của **{protected['checked']:,} tệp đọc được: PASS; 0 thay đổi byte/mtime, 0 tệp mới trong vùng bảo vệ**. Bao gồm raw/datasets/canonical/annotations/intermediate/manifests/exports/reports/cache, toàn audit và P0. Live D48 directory vẫn Access Denied, được ghi rõ; không suy luận là rỗng hoặc tuyên bố parity cho phần không đọc được. Caveat sự cố P0 cũ được giữ nguyên, P1 không khôi phục/ghi lại lịch sử.

E: đầu pilot54.578.716.672B free; reserve15GiB, projected peak2,75GiB PASS. C:/D:/E: dung lượng hiện tại trong `storage_preflight.json`; cùng physical disk là INFERRED từ IOCTL audit trước, E: không là backup độc lập. Parent500 ở D:, không copy sang E:.

1.117 candidate metadata eligible sau loại parent/source URL/ID và terminal rejection. CSV audit có raw_exists, historical_state, rejection reason; 116 raw ngoài D500 không tự được tái dùng. HTTP host chỉ HTTPS `video.adsoftheworld.com`, không credential/port lạ, mọi redirect kiểm allowlist. User xác nhận đã có quyền qua chat, lưu `permission_attestation.json` = OPERATOR_ATTESTED, không giả là giấy phép đã kiểm chứng. [AdsQA hướng dẫn acquisition](https://github.com/TsinghuaC3I/AdsQA#data-acquisition); [AOTW Legal](https://www.adsoftheworld.com/legal) liên kết [Clio terms](https://clios.com/clio-awards-llc-terms-of-service/). robots403 được ghi nhận; URL video HEAD200; các video GET thực tế đều200. Không bypass access denial.

## C. Pilot Acquisition Results

| Chỉ số VERIFIED | Kết quả |
| --- | ---: |
| Wave thực tế | 3 → 7 → 20 candidate |
| Candidate/HTTP request/download thành công | 30 / 30 / 30 |
| Network retry/failure trên video thực tế | 0 / 0 |
| Video kỹ thuật hợp lệ | 29 |
| Rejected | 1 (duration151,658s) |
| Quarantine / source accepted / annotation accepted | 29 / 0 / 0 |
| HTTP bytes nhận | 658.331.255 |
| Raw video retained (gồm rejected, không xóa) | 658.331.255B |
| Tổng logical work/cache tại lúc export | {metrics['total_bytes_retained']:,}B |
| Observed storage high water | {metrics.get('observed_storage_high_water_bytes',metrics['total_bytes_retained']):,}B |
| Trung bình / median valid file | {mean:,.2f} / {statistics.median(sizes):,}B |
| Tổng thời gian các download attempt | {metrics['download_elapsed_seconds']:.3f}s |
| Trung bình thời gian / accepted video | null — chưa có accepted |

Binary mới chỉ ở `E:\\LUMAE\\work\\d1000\\pilot_20261103_01` và cache `E:\\LUMAE\\cache\\p1_20261103_01`. Budget30candidate,45requests,2GiB received,100MiB/file,1h wall clock,2,5GiB work/cache,256MiB processing reserve,15GiB free,concurrency1,2s request interval,2attempt/source,3consecutive errors,domain error rate60% sau5requests. Không tự tăng budget. Dừng vì CANDIDATE_BUDGET; export cuối có stop_reason OFFLINE_REBUILD, không phải một đợt tải mới.

Các deliverable download manifest, source ledger (attempt/status/bytes/retry/path/hash/probe), integrity CSV, annotation ledger, review queue, duplicate candidates, storage metrics và JSONL log đều được copy metadata vào thư mục report; nguồn authoritative ở E:. `pilot_asset_inventory.csv` hash mọi file P1 E:, gồm30 MP4. Peak là logical bytes quan sát ở checkpoint và cuối run, không là NTFS allocated bytes hay peak memory; atomic-write transient chưa được đo liên tục.

## D. Video and Annotation Quality

VERIFIED: 30 SHA khác nhau, không exact-content conflict với D500; probe codec/dimensions/duration, sample decoding 3frame/video, kiểm container MP4/MOV và timestamps đoạn đầu bằng lần validation độc lập. 29 PASS, rejection151,658s được giữ ledger/file. Không tuyên bố giải mã toàn bộ từng frame.

Đã tạo frame/contact sheet cho29video: **{sum(x['frames'] for x in index)} frames, {sum(x['contact_sheets'] for x in index)} contact sheets**. Baseline500D500 fingerprint được tính bằng đọc source video gốc và ràng buộc hash. Sau triage, **{len(flagged)} nguồn có {edges} match nghi vấn** (min dHash distance<=4), chưa xác nhận duplicate. File CSV ghi ID đối chiếu. Ba frame có thể bỏ sót edit/campaign; cảnh tĩnh/endcard có thể false positive. Không có match không chứng minh độc lập campaign. Vì chưa có registry/review campaign đủ căn cứ, cả29 giữ quarantine.

CLIP local CPU image/text smoke (1,512), finite; torch2.8.0+cpu, CUDA false; Semantic V3 frozen hash `cc7097db06a1376819a193b09054c135f8e5fadcef67ba58a82c34b24ba5d5fd` giữ nguyên. Full annotation không unattended: compact VLM đã thất bại negative control và verifier abstains. P1 bổ sung cache/root binding cho producer hiện hữu, không đổi thuật toán ranker/refinement.

Tôi đã xem trực tiếp đủ2contact sheet của `145840e86ebc6bc5fb86d315f8db8eee`, ghi query AI “A man seated at a desk spreads both arms outward.”, evidence frame12s, coarse8–14s. CLIP+Semantic V3 thật trả11–14s, confidenceLOW, inside-minus-outside≈−0,009985; **giữ AI_PSEUDO_PENDING_VISUAL_INTERVAL_REVIEW**, không nhận làm GT. Raw prediction/model SHA/evidence SHA lưu ở E: annotations và bản copy report. Runtime proposal {proposals[0]['elapsed_seconds']:.3f}s trên1video không đại diện throughput toàn bộ annotation. Một tham chiếu sheet ban đầu thiếu prefix đã bị validator chặn; input lỗi giữ riêng, sau đó sửa đúng đường dẫn đã xem.

Blind packet qua ReviewStore chỉ có qid/vid/query/duration; AI bounds bảo toàn trong store. Human labels0, human-verified0, IAA=null. 29 nguồn chờcampaign/query review, 1query có proposal chờinterval verification. Đường `accept-annotation` dùng đầy đủ P0 frozen validator và không tự chuyển AI thành human. Không freeze Test mới.

## E. Resume and Recovery Validation

VERIFIED actual: các wave sau dùng cùng runID, bỏ qua completed/rejected source. Lần resume cuối đặt HTTP trap, rehash30file: attempts30→30, newHTTP0, source record bytes không đổi (`resume_validation.json`).

VERIFIED isolated fixtures: đứt stream rồi Range+If-Range, server bỏ qua Range, sai Content-Range/validator, thiếu byte, HTML/403/redirect lạ, hết byte/disk, immutable config/parent, corrupt/missing export rebuild, previous-run tamper/exclusion, rejected source không thử lại. P0 builder tests kiểm resume sau accepted/rejected annotation và acceptance idempotency. Không tạo record giả trong dataset thật. Chưa có accepted annotation thật nên phần resume đó chỉ được xác minh bằng fixture.

Lock OS được giải phóng sau crash; ledger authoritative là atomic per-attempt/per-source JSON và immutable event segments. JSONL/CSV/manifest là derived exports có thể rebuild. Partial/temp không thành accepted; malformed authoritative input fail-closed. Không giả định file .mp4 không rỗng là download hợp lệ.

## F. Storage Forecast

`d1000_capacity_forecast.json` tách observed và ASSUMED. Còn cần500accepted;29pending chưa được tính. E: free lúc forecast **{free:,}B**; raw mean valid22,229MB, median14,474MB, p90={p90:,}B. Dự toán raw500 theo mean≈{500*mean/1e9:.3f}GB; theo p90≈{500*p90/1e9:.3f}GB; worst100MiB/file≈52,429GB **không đủ nếu giữ reserve15GiB**. Cache/frame dự toán theo29video≈{500*cache_bytes/len(valid)/1e6:.2f}MB/500; annotation allowance64? xem exact JSON (500×128KiB), training feature reference25.902.016B theo tỷ lệ D500, **chưa trích training features mới**. Không copy parent500 hoặc ZIP sangE:.

Observed HTTP success100%, technical29/30=96,67%; tỷ lệ final acceptance còn UNOBSERVABLE vì review pending, không xem pending là100%rejection. Vì vậy empirical attempts-to-500=null. Các kịch bản final acceptance25/50/75/90% là ASSUMED: cần2.000/1.000/667/556attempt. Pool còn1.087 candidate không lặp pilot; trường hợp25% không đủpool. Không nhân500HTTP thành500accepted. Meanprocessing/accepted và empiricalIAA vẫnnull. Full preprocessing/annotation throughput chưa có phân phối đủ mẫu.

## G. Code Changes

Chỉ các file P1 sau được thêm/sửa; thay đổi P0 khác giữ nguyên. Không sửa bytes frozen algorithm, canonical, checkpoints, predictions/metrics lịch sử hoặc reports P0.

| File | Mục đích |
| --- | --- |
'''
report+=''.join(f'| `{path}` | {purpose} |\n' for path,purpose in changed.items())
report+='''
Danh sách absolute path + SHA trong `p1_code_changes.json`. Metadata/versioned evidence mới nằm trong P1 output; video binary không được add/commit vào Git.

## H. Test Results

| Evidence | PASS | FAIL/ERROR/SKIP |
| --- | ---: | --- |
| `full_validation_fixed.xml` | 299 | 0/0/0 |
| `final_targeted_validation.xml` | 41 | 0/0/0 |
| `final_export_validation.xml` | 19 | 0/0/0 |

Các tập có overlap, không cộng thành số unique. Full suite gồm compatibility D48/D200/D500, cached runner parity/RNG, campaign/human provenance và legacy frozen semantic tests. Final targeted bao gồm2case mới về prior-run binding/baseline đầy đủ. Synthetic tests không được dùng thay bằng chứng30video tải thật.

Lệnh đã chạy (PYTHONDONTWRITEBYTECODE=1, LUMAE_P0_ASSETS trỏ P0 read-only):

```powershell
python -m pytest data_process/tests -q -x -p no:cacheprovider --basetemp research_outputs/p1_full_tests_02 --junitxml=research_outputs/p1_20261103_01/full_validation_fixed.xml
python -m pytest data_process/tests/test_p1_acquisition.py data_process/tests/test_p1_video_quality.py data_process/tests/test_general_builder.py data_process/tests/test_campaign_review.py -q -x -p no:cacheprovider --basetemp research_outputs/p1_final_tests_01 --junitxml=research_outputs/p1_20261103_01/final_targeted_validation.xml
python -m pytest data_process/tests/test_p1_acquisition.py -q -x -p no:cacheprovider --basetemp research_outputs/p1_export_tests_01 --junitxml=research_outputs/p1_20261103_01/final_export_validation.xml
python -m compileall -q data_process experiments scripts
git diff --check
```

Compile uses PYTHONPYCACHEPREFIX under new P1 evidence, không ghi pycache lịch sử; compile/diff PASS. Test guard mở rộng đã chặn test cached runner định ghi P0 trước khi mở file; đã đổi sang tmp_path. Trước sửa có full run75PASS/1FAIL do guard, không có P0 mutation. Fixture staging budget quá nhỏ cũng được phát hiện/sửa trong quá trình phát triển; giữ XML lần đầu. Windows ffprobe Unicode decode lỗi được phát hiện trên raw parent thật, sửa explicitUTF-8 và hoàn thành đủ500fingerprint. Một số pytest subprocess có pyreadline console destructor warning khi thoát, exitcode0/JUnitPASS, không phải assertion failure. Final protected fullSHA chứng minh không thay đổi historical output.

## I. Remaining Risks

- Campaign review:29quarantine,2source/3potentialedges; chưa có bằng chứng independence đủ để accepted. Có cả rủi ro missed edit và false-positive dHash.
- Annotation: chỉ1proposalLOW;28video chưa cóquery, mọi source còn chờcampaign; actual full temporal acceptance0. Không có approved unattended local VLM/verifier. Cần kiểm before/inside/after và frozen acceptance contract, không suy correctness từ timestamps.
- Human: không có reviewer/IAA mới; freshTest>=300independent verifiedqueries,>=10%doubles, campaign-disjoint700/150/150 đều TODO.
- Source permission là operator attestation, chưa có independent license-document verification. Không upload/redistribute raw. Availability/rate-limit tương lai chưa được đảm bảo bởi30success.
- Backup: cloudZIPcontentUNVERIFIED; E cùngdisk; không cleanup/migrate. D48 live access chưa giải quyết.
- Storage/throughput: reserve pass cho pilot; worst500×100MiB khôngfit; acceptance/annotation rate chưa đo được. Highwater là observedlogicalbytes, không continuous allocation telemetry.
- Resume sau accepted annotation thật chưa đo vì chưa cóaccepted; fault-injection PASS không thay điều kiện quality/human. Không có critical defect còn biết trong phạm vi đãtest, chưa claim production approval cho fullscale.

## J. D1000 Scale-Up Decision

**NO-GO cho full acquisition.** Historical integrity, bounded real download, technical validation và resume đạt; accepted target/annotation coverage chưa đạt. Không launch500-video campaign.

Reviewed planning config: `research_outputs/p1_20261103_01/scale_config.disabled.json`. Target500additionalaccepted,1087remainingcandidates,1500HTTP,20GiBdownload,24GiBstaging+2GiBprocessingreserve,15GiBfree,8h,concurrency1. Prior-pilot manifestSHA được bind;30ID đã thử bị loại để không tải lại, sourceaccepted trước đó nếu có được tính và reference asset thay vìcopy. Quarantine không được tính. Thay prior-manifest phải tạo reviewed binding mới.

Lệnh dry-run thực thi được, khôngnetwork:

```powershell
python -m data_process.lumae_scale.acquisition plan --config research_outputs/p1_20261103_01/scale_config.disabled.json --output <NEW_PLAN_PATH>
```

Lệnh fullscale cógate đã được chuẩn bị, **hiện phải từ chối**:

```powershell
python -m data_process.lumae_scale.acquisition run --config research_outputs/p1_20261103_01/scale_config.disabled.json --enable-live-network --enable-scale-budget --max-wave 25
```

Trước khi dùng: giải quyết29campaignreviews/near matches; đủ>=20source+annotationaccepted; realpilotvalidationPASS; nguồn/reserve/backlog/throughput được kiểm; operatorexplicitlyapproveslargerbudget bằng config/runversionmới với network_execution_authorized=true và current prior binding. Sau đó chạywave<=25 vàcheckpoint; không tự tăng giới hạn. Configđínhkèmfalse vàpilot_passfalse nêncommandhiệnkhôngkhởichạyfullscale. D1000finalfreeze/Test/training vẫnkhóa saucácđiềukiệnnghiêncứu riêng.
'''
report=report.replace('annotation allowance64? xem exact JSON (500×128KiB)','annotation allowance65.536.000B (500×128KiB)')
report=report.replace('Live D48 directory vẫn Access Denied', 'Đối chiếu 39.878 hash vùng lịch sử với base P0 cũng PASS; bản khôi phục provenance đã được user duyệt ở P0 được tách riêng trong `p0_to_p1_baseline_comparison.json`. Live D48 directory vẫn Access Denied')
report+='\nVERIFIED command gate: `scale_gate_validation.json` xác nhận plan có 1.087 candidate, không network; lệnh scale bị từ chối, không tạo work directory scale và download attempts vẫn30. Query do AI agent xem ảnh trực tiếp; exact revision của model thị giác không được runtime công bố, không giả định có model revision pin cho phần này. CLIP weights và Semantic V3 source được pin bằng SHA.\n'
for before,after in {'29video':'29 video','30video':'30 video','500video':'500 video','500source':'500 source','1query':'1 query','28video':'28 video','30file':'30 file','500fingerprint':'500 fingerprint','2contact sheet':'2 contact sheet','3frame/video':'3 frame/video','sourceaccepted':'source accepted','annotationaccepted':'annotation accepted','videoaccepted':'video accepted','candidateHTTP':'candidate HTTP','concurrency1':'concurrency 1','raw8.836':'raw 8.836','target25':'target 25','30candidate':'30 candidate','45requests':'45 requests','2attempt/source':'2 attempt/source','3consecutive':'3 consecutive','5requests':'5 requests','500additionalaccepted':'500 additional accepted','1087remainingcandidates':'1087 remaining candidates','1500HTTP':'1500 HTTP','15GiBfree':'15 GiB free','8h,concurrency':'8h, concurrency','29campaignreviews':'29 campaign reviews','20source+annotationaccepted':'20 source + annotation accepted','realpilotvalidationPASS':'real pilot validation PASS','operatorexplicitlyapproveslargerbudget':'operator explicitly approves larger budget','config/runversionmới':'config/run version mới','current prior binding':'current prior binding','chạywave':'chạy wave','checkpoint;không':'checkpoint; không','Configđínhkèmfalse':'Config đính kèm false','pilot_passfalse':'pilot_pass false','commandhiệnkhôngkhởichạyfullscale':'command hiện không khởi chạy full scale','D1000finalfreeze/Test/training vẫnkhóa saucácđiềukiệnnghiêncứu riêng':'D1000 final freeze/Test/training vẫn khóa sau các điều kiện nghiên cứu riêng','attempts vẫn30':'attempts vẫn 30'}.items():report=report.replace(before,after)
(O/'p1_engineering_report.md').write_text(report,encoding='utf-8')
checks={p.name:digest(p) for p in O.iterdir() if p.is_file() and p.name!='evidence_checksums.json'}
atomic_json(O/'evidence_checksums.json',checks)
print(json.dumps({'report':str(O/'p1_engineering_report.md'),'decision':'NO-GO','downloaded':30,'technical_valid':len(valid),'accepted':0,'new_root_bytes_observed':sum(r['bytes'] for r in inventory),'flagged_sources':len(flagged),'tests':tests},ensure_ascii=True))
