"""Observed pilot30 report; reads scientific evidence and writes metadata only."""
import csv,json,statistics,sys
from collections import Counter
from pathlib import Path
import xml.etree.ElementTree as ET
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from data_process.lumae_scale.models import ROOT,read_json,atomic_json,digest
O=ROOT/'research_outputs/p14_p15_20261103_02';B=ROOT/'research_outputs/p14_p15_20261103_01'
A=Path('E:/LUMAE/work/d1000/p14_p15_20261103_01');E=A.with_name('p14_p15_20261103_02')
if (O/'pilot30_freeze_manifest.json').exists():raise PermissionError('Frozen report/metrics: use a new version for regeneration')
for name in ('regression.xml','recovery_validation.xml'):
    if not (O/name).exists():__import__('shutil').copyfile(B/name,O/name)
tests={}
for name in ('regression.xml','recovery_validation.xml'):
    suites=list(ET.parse(O/name).getroot().iter('testsuite'));tests[name]={k:sum(int(s.get(k,0)) for s in suites) for k in ('tests','failures','errors','skipped')}
atomic_json(O/'regression_summary.json',tests)
rows=list(csv.DictReader((O/'pilot30_inventory.csv').open(encoding='utf-8-sig')));recon=list(csv.DictReader((O/'pilot30_final_reconciliation.csv').open(encoding='utf-8-sig')))
accepted=read_json(O/'pilot30_accepted_annotation_manifest.json')['annotations'];decisions=[json.loads(l) for l in (O/'pilot30_annotation_decisions.jsonl').read_text(encoding='utf-8').splitlines()]
timers=[]
for folder in (A,E):
    for p in (folder/'annotations').glob('*/proposal.json'):
        if read_json(p).get('generation_seconds'):timers.append(read_json(p)['generation_seconds'])
new_folders=[A,E,Path('E:/LUMAE/work/p14_p15_20261103_01'),Path('E:/LUMAE/work/p14_p15_20261103_02')]
additional=sum(p.stat().st_size for f in new_folders for p in f.rglob('*') if p.is_file())
old=sum(p.stat().st_size for f in [Path('E:/LUMAE/work/d1000/pilot_20261103_01'),Path('E:/LUMAE/cache/p1_20261103_01')] for p in f.rglob('*') if p.is_file())
metrics={'original_candidates':30,'technical_valid':29,'technical_validity_rate':29/30,'exact_duplicates':0,'exact_duplicate_rate':0,'suspected_near_duplicate_edges_reviewed':3,'confirmed_near_duplicates':0,'confirmed_duplicate_fraction_of_flagged_edges':0,'provisional_campaign_groups':29,'verified_independent_campaign_group_count':None,'confirmed_distinct_campaign_relationships':3,'source_retained_accepted_split_restricted':29,'split_unrestricted_sources':0,'accepted_source_annotation_pairs':len(accepted),'accepted_temporal_annotations':len(accepted),'source_pair_acceptance_rate_of_all30':len(accepted)/30,'source_pair_acceptance_rate_of_valid29':len(accepted)/29,'rejected_source_count':1,'annotation_failed_source_count':29-len(accepted),'source_quarantine_due_to_known_duplicate':0,'campaign_split_restricted_source_count':29,'AI_pseudo_accepted':len(accepted),'human_verified':0,'pending_human_review_accepted_queries':len(accepted),'human_IAA':None,'mean_accepted_source_duration_seconds':statistics.mean(r['duration'] for r in accepted),'mean_accepted_interval_seconds':statistics.mean(r['window'][1]-r['window'][0] for r in accepted),'accepted_intervals_at_most2seconds':sum(r['window'][1]-r['window'][0]<=2 for r in accepted),'accepted_intervals_at_most2seconds_fraction':sum(r['window'][1]-r['window'][0]<=2 for r in accepted)/len(accepted),'annotation_decision_counts_all_explicit_attempts':dict(Counter(r['decision'] for r in decisions)),'original_raw_bytes':658331255,'additional_P14_P15_E_bytes_observed':additional,'all_pilot_E_bytes_observed':old+additional,'timed_feature_inference_attempts':len(timers),'timed_feature_inference_seconds':sum(timers),'feature_inference_attempts_per_second':len(timers)/sum(timers),'accepted_pairs_per_end_to_end_wallclock_second':None,'timing_scope':'Component timer excludes dense frame extraction, later query revisions and interactive visual adjudication; no full pipeline/human throughput claimed','new_video_downloads':0,'decision':'NO-GO'}
atomic_json(O/'metrics.json',metrics)
changed={
 'data_process/lumae_scale/pilot_completion.py':'New: strict pilot subset validation, immutable deterministic freeze, CompletionConfig cache binding for existing builder recovery.',
 'data_process/tests/test_pilot_completion.py':'New: immutable freeze, false-human gate, original30 identity and atomic-accept recovery fault injection.',
 'data_process/tests/conftest.py':'Modified: extend protected write guard to original P1 evidence and E: source/cache paths.',
 'scripts/pilot30_review_setup.py':'New: real30 inventory/hash/probe and complete visual overview; no downloader import.',
 'scripts/pilot30_propose.py':'New: first actual visual-query inference, blind-first numerical estimate and local dense QC.',
 'scripts/pilot30_round2.py':'New: bounded dense exact-seek recovery and explicit query/candidate revisions, same frozen algorithm.',
 'scripts/pilot30_finalize.py':'New: versioned BuilderSession acceptance, strict validation, all30 reconciliation and freeze CLI.',
 'scripts/pilot30_resume_validate.py':'New: real accepted-ledger HTTP-denied idempotency and sealed read-only verification.',
 'scripts/pilot30_report.py':'New: observed metrics and Vietnamese A–K report.'}
atomic_json(O/'changed_files.json',[{'path':str(ROOT/p),'purpose':reason,'sha256':digest(ROOT/p)} for p,reason in changed.items()])
table='| Source ID | Query ID | Raw interval (s) | Query |\n| --- | --- | --- | --- |\n'+''.join(f"| `{r['source_video_id']}` | `{r['qid']}` | {r['window']} | {r['query']} |\n" for r in accepted)
reject_table='| Candidate | Source ID | Disposition |\n| --- | --- | --- |\n'+''.join(f"| {r['candidate_id']} | `{r['source_id']}` | {r['reason']} |\n" for r in recon if r['accepted_annotations']=='0')
report=f'''# LUMAE — P1.4 / P1.5: audit và hoàn thiện pilot30

Ngày03/11/2026, Asia/Saigon. Final version `p14_p15_20261103_02`; version01 là evidence/ledger preliminary được bảo toàn. VERIFIED = đọc/đo/chạy/xem ảnh trực tiếp; INFERRED = suy luận; UNKNOWN = không có bằng chứng đủ. Không tải hoặc re-fetch video, không GPU training, không scale-up.

## A. Executive Summary

Đã audit đủ **30/30 candidate**, xem toàn bộ overview của29 video hợp lệ và QC từng proposal ở1–2FPS. Kết quả cuối: **16 cặp source–annotation ACCEPTED_AI_PSEUDO**,13 nguồn hợp lệ chưa có nhãn đạt gate,1 nguồn bị loại vì151,658s vượt150s. Không đổi giới hạn duration.

Giữ29 source khác SHA/ID/URL ở trạng thái **accepted with split restriction**; điều này không đồng nghĩa29 cặp accepted hay29 campaign độc lập. Cả29 chưa đủ điều kiện split mới/fresh Test. Quality target20 chưa đạt: **NO-GO cho acquisition tiếp theo**. Engineering validation đạt trong phạm vi procedure đã chạy; hạn chế model/review và short-event được giữ rõ.

## B. Original 30-Video Reconciliation

`pilot30_inventory.csv` chứa ID, original URL/path, size/SHA, duration/resolution/FPS/codec, original P1 status và status hiện tại. `pilot30_inventory.at_start.csv` giữ bản đầu. `pilot30_final_reconciliation.csv` đối chiếu đủ30 hàng:16 ACCEPTED_SOURCE_AND_ANNOTATION,13 REJECTED_ANNOTATION_QUALITY,1 REJECTED_TECHNICAL. Với13 nhãn thất bại, source bytes vẫn giữ và source acceptance riêng vẫn là split-restricted.

Rehash toàn bộ30 MP4:658.331.255B,30 SHA khác nhau, không exact duplicate với500 D500/ID/URL. Video rejected `b5b285ef15c83b8477c638bd0d2e468a` vẫn151,658s; readable video không đồng nghĩa policy-valid hoặc accepted annotation. Không xóa/move/copy lại raw video.

Các candidate chưa thành cặp accepted:

{reject_table}

## C. Campaign and Duplicate Audit

Đã xem actual sampled sequences của29 pilot và ba parent match, không suy nội dung từ tên file. Ba dHash match được giải quyết về nội dung. DISTINCT_CAMPAIGN là phân loại AI suy từ chuỗi hình và branding khác nhau, không phải publisher registry đã được xác minh:

| Pilot | Parent | Content difference VERIFIED; campaign classification INFERRED |
| --- | --- | --- |
| `5e4e41d967149975220c2b6bab354c0c` AmFam | `47ad45ea18e1a859c8355efc27d44fc2` Sharpie | DISTINCT_CAMPAIGN; family/dance insurance khác recipe/browser highlighter |
| `5e4e41d967149975220c2b6bab354c0c` AmFam | `aae60e7083aa64347cf079c69e98f1bc` Jooble | DISTINCT_CAMPAIGN; khác montage workers/PPE/jobs |
| `bf5accad69519c61b4ea27ee0d2a0136` SPAR | `dda6f2b2f59239c6c5a8fc6b9045c588` Pepsi | DISTINCT_CAMPAIGN; holiday/community khác cola/music performance |

Exact duplicates0; confirmed near-duplicate edits0/3 flagged edges. Đây là kết quả trong3 cảnh báo, không phải FPR đã hiệu chuẩn cho toàn bộ corpus. Có29 provisional campaign IDs, không có bằng chứng xác nhận29 campaign độc lập. Các quan hệ khác vẫn UNKNOWN; không gán DISTINCT chỉ vì SHA khác. Cùng campaign nhưng khác video không tự bị reject; mọi confirmed group tương lai phải nằm trong một split. `campaign_relationships.json`, `duplicate_resolution.csv`, `campaign_group_manifest.json` chứa evidence paths/SHA và restrictions. Không freeze Train/Val/Test mới.

## D. Temporal Annotation Recovery

Vòng01: genuine Codex multimodal inspection trên overview đầy đủ669 frame/80 contact sheet gốc, tạo query cho mọi29 source; CLIP/Semantic V3 chạy thật, thêm1FPS QC và2FPS quanh mọi boundary. Có4 nhãn được nhận qua frozen contract sau review, các interval lỗi/correction giữ riêng.

Vòng02: chỉ xử lý25 mẫu chưa đạt. Dùng exact-seek frames đã xem và positive timestamps của toàn đoạn sự kiện thay cho một anchor đơn lẻ; sampling1–2FPS được liệt kê tường minh, tạo cache annotation mới. **Không sửa thuật toán Semantic V3/ranker/boundary wrapper và không sửa historical training features.** Các query sửa được lưu revision riêng: outfit thay jacket không chắc chắn; six-copy grid; wide4DX shot; brown-cover→BW pages; posing thay dancing; phone/background transition thay laptop; shield strap attachment; bags có concrete subject đúng sanitation.

Vòng recovery thêm12 cặp, giữ nguyên bytes/IDs của4 record vòng01. Raw model outputs, query/candidate revisions và input evidence được giữ trong E: version01/02, có `raw_inference_artifact_index.csv`. Các CSV/JSONL là initial/latest revision index; mọi intermediate raw file vẫn nằm trong evidence manifest. Câu bags đầu tiên bị validator frozen chặn vì thiếu concrete subject phrase; sửa grammar bằng revision3, không sửa gate. Pointer export trước correction được giữ trong `provenance_revisions`; `provenance_alignment.json` đối chiếu exact final query/raw inference/blind estimate cho đủ16 mẫu.

Independent procedure: tính query-only CLIP estimate **trước** region-conditioned proposal, không đưa generator interval/coarse phase vào numerical retrieval. Sau đó một lượt actual visual QC/adjudication xem before/inside/after ở2FPS và xử lý bất đồng. Cùng CLIP weights và cùng Codex interface, không có distinct independently calibrated semantic model. Không gọi đây là hai human annotator, human IAA hoặc independent-model benchmark. Model revision của interface không được runtime cung cấp; lưu NOT_EXPOSED. CLIP weight SHA và frozen algorithm SHA được pin.

Proposal LOW cũ của `145840e86ebc6bc5fb86d315f8db8eee` đã được xem lại ở2FPS: gesture hai tay nhìn thấy11,5–13s;11–14s giữ như **AI contextual envelope**, có actual visual adjudication. ConfidenceLOW và negative CLIP margin giữ nguyên; không chuyển thành AUTO-HIGH hay Test GT. Existing `pipeline.py` cho phép MEDIUM đi qua actual verifier notes; chúng tôi dùng chính frozen validator/BuilderSession, không nới confidence threshold hoặc giả calibration.

Các interval accepted là pseudo envelopes theo wrapper có context lân cận khoảng1s, không phải frame-exact human boundaries. `reviewed_interval`/observed phase được giữ riêng; không thay raw V3 window để làm validator PASS. Trường hợp short event bị lẫn identity/shot hoặc coverage không đáng tin vẫn reject. Video-start/end không có outside context được ghi rõ, không tạo timestamp âm hoặc frame giả.

## E. Accepted Dataset Inventory

Tất cả16: AI_PSEUDO_LABELED, HUMAN_VERIFIED=false, reviewer_id=AI_PIPELINE; pending genuine human review. Mỗi record đã qua `freeze.validate_annotation` và default `BuilderSession.accept` với cache binding phù hợp. Mỗi source chỉ có một accepted query ID; revisions không được tính thành nhiều source.

{table}

## F. Pilot Acceptance Metrics

| Chỉ số observed | Giá trị |
| --- | ---: |
| Original / policy-valid |30 /29 (96,67%) |
| Exact duplicates / confirmed flagged near duplicates |0 /0 |
| Source retained with split restriction |29 |
| Cặp accepted / tất cả30 |16 /30 =53,33% |
| Cặp accepted / valid29 |16 /29 =55,17% |
| Rejected source / source có annotation chưa đạt |1 /13 |
| Campaign/split restricted / split unrestricted |29 /0 |
| Accepted AI-pseudo / human-verified |16 /0 |
| Pending human review accepted queries |16 |
| Mean accepted-video duration |{metrics['mean_accepted_source_duration_seconds']:.3f}s |
| Mean accepted raw interval length |{metrics['mean_accepted_interval_seconds']:.3f}s |
| Raw intervals<=2s |{metrics['accepted_intervals_at_most2seconds']}/16 = {100*metrics['accepted_intervals_at_most2seconds_fraction']:.2f}% |
| Additional P1.4/P1.5 E: logical storage |{additional:,}B |
| Old+new pilot E: logical storage |{old+additional:,}B |
| New video downloads |0 |

`metrics.json` chứa decision counts trên toàn bộ attempt revisions. Tỷ lệ short raw intervals không chứng minh wrapper đã giải quyết short events: context có thể làm envelope dài hơn sự kiện thực, và nhiều short cases vẫn reject. Có{len(timers)} feature/inference component timers, tổng{sum(timers):.3f}s, khoảng{len(timers)/sum(timers):.3f}attempt/s; không gồm extraction, revision bổ sung và thời gian agent xem ảnh. End-to-end accepted throughput=null vì không có timer thống nhất cho toàn phiên. Không báo human IAA. Batch30 nhỏ và purposive; không suy yield chính xác cho500 source tiếp theo.

## G. Pilot Freeze and Historical Integrity

Final artifacts ở `D:\\DUT\\Se07\\CV\\lumae\\research_outputs\\p14_p15_20261103_02`.
`pilot30_freeze_manifest.json` và `.sha256` niêm phong pilot subset, **final_d1000_freeze=false**. Manifest chứa original30/source hashes, reconciliation, accepted qids/windows, campaign restrictions, rejected reasons, originalP1ID, processing-code/model hashes và toàn bộ referenced evidence. E: evidence ở `work/d1000/p14_p15_20261103_01` và `...02`; accepted ledger mới ở `E:\\LUMAE\\work\\p14_p15_20261103_02`. `storage_bindings.json` làm rõ root/cache/parent, không sửa canonical paths cũ.

Frozen producers bị khóa; không regenerate vào sealed version. `freeze_completion` dùng deterministic sorted JSON, atomic immutable write, SHA và từ chối overwrite khác bytes. Before seal, corrected derived provenance exports giữ bản cũ/hash journal; không thay record đã accepted. `pilot30_resume_validation.json` kiểm16 actual records: IDs ổn định, accepted/event bytes không đổi, repeated JSONL export byte-identical, HTTP requests0. After seal validation chỉ đọc authoritative manifest/evidence, output diagnostic mới.

Full protected-state comparison và riêng1.253 tài sản P1/source/cache cũ: xem `historical_final_validation.json`; known D48 live directory Access Denied giữ nguyên, không gọi là rỗng hoặc tự restore. Không đổi D500, Val/Test, predictions/config/checkpoints lịch sử, E0/E1/E2 hoặc P0/P1 reports. Cloud backup SHA còn UNVERIFIED như trước, E cùng physical disk không là backup độc lập; không cleanup.

## H. Validation

Full suite **305 PASS,0 fail/error/skip**, và targeted recovery/freeze **5 PASS** có overlap; không cộng thành310 unique. Test mới gồm immutable freeze/tamper, false-human promotion, original30 identity và fault injection giữa accepted atomic file/ledger event với cache binding. Actual16 strict validation và no-network resume PASS. Syntax/diff check sử dụng bytecode prefix mới, không ghi pycache lịch sử.

Lệnh đã chạy, PYTHONDONTWRITEBYTECODE=1 và LUMAE_P0_ASSETS trỏ P0 read-only:

```powershell
python -m pytest data_process/tests -q -x -p no:cacheprovider --basetemp research_outputs/p14_p15_test_tmp --junitxml=research_outputs/p14_p15_20261103_01/regression.xml
python -m pytest data_process/tests/test_pilot_completion.py -q -x -p no:cacheprovider --basetemp research_outputs/p14_p15_recovery_tmp --junitxml=research_outputs/p14_p15_20261103_01/recovery_validation.xml
python scripts/pilot30_resume_validate.py --round2
python scripts/pilot30_finalize.py --round2 --freeze
python -m compileall -q data_process experiments scripts
git diff --check
```

Producer commands `pilot30_review_setup.py`, `pilot30_propose.py`, `pilot30_round2.py` đã chạy trước seal và giờ từ chối ghi vào frozen version. `pilot30_finalize.py --round2` tái lập acceptance trước seal; sau seal dùng read-only resume command. Không có download CLI nào được dùng. Gate sanitation từng bắt lỗi bags và versioned pointer correction được bảo toàn; không bỏ qua validation failure. Existing P0 cached runner/legacy D48/D200/D500, campaign/review regression đều trong suite.

## I. Changed Files

Chỉ thêm source/workflow P1.4 và mở rộng test guard; pre-existing uncommitted changes giữ nguyên, không commit.

'''
report+='\n'.join(f'- `{p}` — {v}' for p,v in changed.items())
report+='''

Absolute paths/SHA trong `changed_files.json`. New frame/proposal artifacts không được add binary vào Git. Không thay frozen ranker hoặc existing acceptance semantics.

## J. Remaining Blockers

- **Temporal/model quality:**13 valid sources vẫn không có nhãn đạt gate sau2 vòng. Short events còn lẫn nhiều shot/identity; một số envelopes quá rộng. Corrections được giữ riêng, không lén thay raw window để pass.
- **Campaign/split:**29 provisional groups, chỉ3 quan hệ match đã được review trực tiếp. UNKNOWN không đồng nghĩa DISTINCT; cần registry resolution trước partition/fresh Test. Shared campaign-distinct video có thể giữ khi group cùng split.
- **Independent verification limitation:** có genuinely separate query-only numerical procedure và actual AI visual adjudication, nhưng cùng CLIP weights/interface, không có distinct calibrated semantic reviewer. Không quảng bá16 nhãn như independently human-verified GT.
- **Human:**0 human labels/IAA; training AI-pseudo không đòi human identity giả. Muốn new blind Test phải genuine independent human labels/adjudication và các gate300queries/10%doubles, chưa thực hiện.
- **Access/backup:**D48 inaccessible phần đã ghi nhận; cloud ZIP content unverified; E không independent backup. Không liên quan quyền tải mới vì task này không download.
- **Implementation:**không còn known critical integrity defect trong scope đã kiểm. Chưa claim GPU/training/full D1000 readiness. Timing chỉ là component observations, không full throughput calibration.

## K. Final Decision

**NO-GO cho acquisition D1000 tiếp theo.** Pilot audit/reconciliation và engineering validation hoàn tất;16<20 cặp đáp ứng gate, còn13 temporal failures và campaign restrictions. Có thể dùng evidence để sửa workflow/chuẩn bị review tiếp, chưa đủ quality evidence để khởi động một chiến dịch500 accepted videos. Không tự launch, không re-fetch30 video hiện có, không freeze final D1000/Test hoặc training.
'''
(O/'pilot30_acceptance_report.md').write_text(report,encoding='utf-8')
for name,text in [('campaign_triage_report.md','Xem sectionC của pilot30_acceptance_report.md; three reviewed DISTINCT pairs,29 provisional split-restricted groups, no global independence claim.'),('pilot30_annotation_quality_report.md','Xem sectionsD–F và từng visual_adjudication/raw revision;16 accepted AI-pseudo,13 valid sources without accepted labels,0 human-verified.'),('pilot30_integrity_report.md','Xem sectionsG–H, strict_validation, historical_final_validation và resume_validation; final pilot-only immutable manifest, not D1000 freeze.'),('execution_plan.md','Đã thực hiện: inventory/hash/probe → full overview/duplicate-parent review → all29 query/V3 proposals →1–2FPS QC → bounded dense recovery → strict BuilderSession acceptance → actual ledger resume/HTTP trap → regression/protected comparison → immutable pilot subset freeze. Code/paths/commands được ghi trong report A–K.')]:
    (O/name).write_text(text+'\n',encoding='utf-8')
print({'accepted':len(accepted),'additional_E_bytes':additional,'tests':tests,'report':str(O/'pilot30_acceptance_report.md')})
