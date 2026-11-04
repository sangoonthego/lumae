"""Consolidate P0 evidence; never rewrites an already sealed freeze manifest."""
from __future__ import annotations
import csv
from collections import Counter
from datetime import datetime,timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import xml.etree.ElementTree as ET

R=Path(__file__).resolve().parents[1];O=R/'research_outputs/p0_20261103_01';A=R/'audit_outputs/2026-11-02'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(name,value):
    (O/name).write_text(value if isinstance(value,str) else json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
def csvwrite(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

base=read(O/'research_freeze_manifest.json');assets={x['path']:x for x in base['assets']}
protected_prefixes=[R/'local_data'/p for p in ['raw','datasets','canonical','annotations','intermediate','manifests','exports','reports','cache']]+[A]
changed=[];errors=[];verified=0;approved_restorations=[]
recovery=read(O/'manifest_recovery_executed.json') if (O/'manifest_recovery_executed.json').is_file() else None
for item in base['assets']:
    p=Path(item['path'])
    if not any(p.is_relative_to(d) for d in protected_prefixes):continue
    try:
        s=p.stat();verified+=1
        if s.st_size!=item['bytes'] or s.st_mtime_ns!=item['mtime_ns']:
            current_sha=sha(p)
            if current_sha!=item['sha256']:
                if recovery and str(p)==recovery['target'] and current_sha==recovery['after_sha256'] and recovery['status']=='RESTORED_VERIFIED_SNAPSHOT_WITH_EXPLICIT_USER_APPROVAL':approved_restorations.append(recovery)
                else:changed.append(str(p))
    except OSError as e:errors.append({'path':str(p),'error':str(e)})
if changed or errors:
    write('protected_state_anomalies.json',{'changed':changed,'errors':errors,'compared':verified})
    raise RuntimeError('Protected historical inputs changed or became unreadable; see protected_state_anomalies.json')
readonly={'pass':True,'readable_protected_files_compared':verified,'unapproved_remaining_changes':changed,'approved_snapshot_restorations':approved_restorations,'errors':errors,'mode':'current stat vs initial SHA snapshot; SHA recheck on metadata changes; explicit approved restore recorded separately','initial_read_errors':base['read_errors'],'baseline_caveat':'Initial freeze overlapped an early baseline run; exact pre-task bytes of the single rewritten provenance timestamp were not captured'}
write('protected_state_validation.json',readonly)
dataset=R/'local_data/datasets/lumae_ads_d500';manifest=read(dataset/'dataset_manifest.json')
raw=R/'local_data/raw/adsqa/videos';mismatches=[]
for item in manifest['source_video_sha256'].values():
    p=raw/item['video_filename'];actual=assets[str(p)]['sha256']
    if actual!=item['sha256']:mismatches.append(str(p))
if mismatches:raise RuntimeError('Raw D500 source SHA mismatch')
write('d500_raw_checksum_verification.json',{'pass':True,'videos':len(manifest['source_video_sha256']),'bytes':base['dataset']['raw_bytes'],'mismatches':mismatches,'scope':'fresh full SHA during P0.0, unchanged protected metadata at final validation'})
selection=read(dataset/'selection_manifest.json');states={x['source_video_id']:x for x in selection['candidates']};inherited=selection.get('inherited_terminal_rejections',{})
known={x['video_filename'] for x in manifest['source_video_sha256'].values()};extras=[]
for p in sorted(raw.glob('*.mp4')):
    if p.name in known:continue
    state=states.get(p.stem,{});status=state.get('status','INHERITED_REJECTED' if p.stem in inherited else 'NOT_IN_CURRENT_SELECTION')
    extras.append({'path':str(p),'bytes':p.stat().st_size,'sha256':assets[str(p)]['sha256'],'historical_state':status,'reason':state.get('rejection_reason',inherited.get(p.stem)),
        'd1000_accepted':False,'recommendation':'REVIEW_SOURCE_AND_ANNOTATION_REQUIRED; never auto-count raw files'})
csvwrite(O/'outside_d500_raw_assessment.csv',extras)
plan=read(O/'d1000_current_pool_dry_run_v2.json');parity=read(O/'cached_runner_parity_results.json');cloud=read(O/'current_cloud_research_payload.json')
checks=read(O/'asset_access.json');backup=read(O/'cloud_backup_status.json');nb=read(O/'current_notebook_payload.json')
if checks['status']!='VERIFIED' or parity['status']!='PASS' or not nb['prior_source_equal']:raise RuntimeError('P0.1 real-input gate failed')
current_features=read(O/'feature_materialization_payload.json')
if not all(f['prior_sha256_match'] for f in current_features):raise RuntimeError('Cloud feature identity changed')
tests={}
for p in sorted(O.glob('*.xml')):
    root=ET.parse(p).getroot();suites=list(root.iter('testsuite'));tests[p.name]={'tests':sum(int(s.get('tests',0)) for s in suites),'failures':sum(int(s.get('failures',0)) for s in suites),'errors':sum(int(s.get('errors',0)) for s in suites),'skipped':sum(int(s.get('skipped',0)) for s in suites)}
for name in ('full_validation_guarded_final.xml','final_rng_annotation_tests.xml','builder_storage_tests.xml','isolated_query_freeze_test.xml'):
    if name not in tests or tests[name]['failures'] or tests[name]['errors']:raise RuntimeError('Final test gate missing/failed: '+name)
write('validation_summary.json',tests)
environment={'python':platform.python_version(),'platform':platform.platform()}
for name in ['torch','numpy','scipy','pandas','scikit-learn','pytest','fastapi','streamlit']:
    try:environment[name]=importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:environment[name]=None
write('execution_environment.json',environment)
changed_files=['data_process/lumae_scale/freeze.py','data_process/lumae_scale/builder.py','data_process/config/d1000_v1.json',
    'data_process/splitting/campaign.py','data_process/annotation/independent_review.py','data_process/tests/test_cached_runner.py',
    'data_process/tests/test_general_builder.py','data_process/tests/test_campaign_review.py','data_process/tests/test_query_sanitation.py','experiments/M1_lumae_ads/cached_runner.py',
    'experiments/M1_lumae_ads/config_d500_cached.json','experiments/M1_lumae_ads/README_CACHED_D500.md','data_process/lumae_scale/README_D1000.md',
    'docs/reproducibility/d1000_split_policy.md','docs/reproducibility/human_annotation_protocol.md','scripts/research_freeze.py',
    'scripts/download_connected_asset.py','scripts/finalize_p0_report.py']
changed_files += ['data_process/tests/conftest.py','data_process/tests/test_research_write_guard.py','data_process/tests/test_clean_blind_benchmark.py',
    'data_process/tests/test_clean_blind_v3_benchmark.py','data_process/tests/test_fresh5_benchmark.py','data_process/tests/test_semantic_preannotator_v2.py','data_process/tests/test_semantic_v3.py']
source={str(R/p):{'bytes':(R/p).stat().st_size,'sha256':sha(R/p)} for p in changed_files}
write('changed_source_checksums.json',source)
patch=subprocess.check_output(['git','diff','--','data_process/lumae_scale/freeze.py','data_process/tests'],cwd=R).decode('utf-8')
write('legacy_compatibility.patch',patch)
dependencies={'dataset':'lumae_ads_d500','runner':str(R/'experiments/M1_lumae_ads/cached_runner.py'),'recipe':str(R/'experiments/M1_lumae_ads/config_d500_cached.json'),
    'upstream_commit':'b7e553ac3b0c898ee6b85e03ee507c064eab89ca','code_requirements':['LUMAE experiments/M1_lumae_ads/cached_runner.py + config','data_process/__init__.py','data_process/validation/*','data_process/schemas/*','pinned Moment-DETR model/dataset/matcher/transformer/span/position/misc, utils and standalone_eval'],
    'data_requirements':{'train_queries':440,'val_queries':30,'feature_files':940,'feature_bytes':24427377,'train_val_jsonl_bytes':2216546,'M0_bytes':16952721,'feature_manifest_bytes':138978,'frozen_recipe_bytes':538,'reference_minimum_payload_bytes':43736160},
    'raw_video_required':False,'zip_extraction_in_entrypoint':False,'CLIP_weights_required':False,'test_annotation_argument':False,
    'feature_root':'/content/drive/MyDrive/CV_Lumae','feature_manifest':'artifacts/m1_d500_features/feature_manifest.json',
    'feature_cache_directories':['artifacts/m1_d200_features','artifacts/m1_d500_features'],
    'local_fresh_content_validation':{'feature_files':66,'feature_pairs':33,'includes_all_val30':True,'includes_float16_train':True,'full_local_940_preflight':'TODO execute against full existing cache in Colab'},
    'dtype_policy':'Preserve original NPZ bytes; notebook float32 cast at read; historical float16 exception retained',
    'environment':environment,'original_gpu_environment':'TODO','full_gpu_retraining_executed':False,'resume':'new runner-owned checkpoint format only; config/cache/code/runtime fingerprint + Python/NumPy/Torch/CUDA/generator RNG state'}
write('cached_runner_dependencies.json',dependencies)
final=O/'freeze_final';final.mkdir(exist_ok=True)
frozen={'schema_version':2,'version':'p0_20261103_01_final','created_utc':datetime.now(timezone.utc).isoformat(),
    'base_freeze':{'path':str(O/'research_freeze_manifest.json'),'sha256':sha(O/'research_freeze_manifest.json'),'asset_count':len(base['assets']),'caveat':readonly['baseline_caveat']},
    'dataset':base['dataset'],'local_zip':base['local_zip'],'all_500_raw_sha_verified':True,'protected_state_validation':readonly,
    'cloud_backup':backup,'live_D48_access':read(O/'d48_access_recheck.json'),'approved_incident_recovery':recovery,
    'prior_audit_metadata_deltas':read(O/'prior_audit_metadata_deltas.json'),
    'feature_cache_prior_full_snapshot':base['feature_cache'],'current_feature_observations':current_features,
    'M0_current_content':checks,'checkpoint_metadata':{k:read(O/f'current_{k}_metadata.json') for k in ('M0','d200_best','d500_e0_best')},
    'checkpoint_status':'M0 content verified; M1-D200/E0 checkpoint current metadata + historical frozen expected hashes, not freshly downloaded/hashed',
    'current_notebook_identity':nb,'historical_experiments':cloud,'git_base_commit':base['git_commit'],'current_source_snapshot':source,
    'evaluator':{'commit':dependencies['upstream_commit'],'source_sha256':sha(O/'upstream/standalone_eval/eval.py')},
    'execution_environment':environment,'historical_gpu_environment':'TODO','human_reviews_created':0,
    'cleanup_or_destructive_migration_allowed':False,'test_validation':tests,'no_GPU_training_or_live_crawling':True}
with (final/'research_freeze_manifest.json').open('x',encoding='utf-8') as f:json.dump(frozen,f,ensure_ascii=False,indent=2)
(final/'research_freeze_manifest.sha256').write_text(sha(final/'research_freeze_manifest.json')+'\n')
(final/'historical_experiments_manifest.json').write_text(json.dumps({'prior_inventory_manifest':str(O/'historical_experiments_manifest.json'),'current_content_observations':cloud,'notebook_identity':nb,'execution_environment':environment},ensure_ascii=False,indent=2),encoding='utf-8')
checksum_rows=[{'path':a['path'],'bytes':a['bytes'],'sha256':a['sha256'],'evidence':a['evidence']} for a in base['assets']]
for row in checksum_rows:
    if recovery and row['path']==recovery['target']:row['sha256']=recovery['after_sha256'];row['evidence']='USER_APPROVED_VERIFIED_HISTORICAL_SNAPSHOT_RESTORE; exact pre-task bytes unproved'
checksum_rows += [{'path':checks['path'],'bytes':checks['bytes'],'sha256':checks['sha256'],'evidence':'VERIFIED_CURRENT_CONTENT'}]
checksum_rows += [{'path':'https://drive.google.com/file/d/'+x['id']+'/view','bytes':x['observed_content_bytes'],'sha256':x['sha256'],'evidence':x['evidence']} for x in cloud]
csvwrite(final/'research_asset_checksums.csv',checksum_rows)
backup_report=f'''# P0.0 — Backup integrity / research freeze

VERIFIED local ZIP8,844,848,885B SHA256 `{base['local_zip']['sha256']}`; all500raw MP4 SHA match frozen source manifest,8,836,814,052B. {len(base['assets'])} readable historical files hashed during initial freeze. {verified} readable protected files checked finally; the explicitly approved restoration is recorded separately, not called unchanged.

Incident: an existing test invoked the real clean-blind query freezer and rewrote its timestamp. Initial baseline overlapped freeze completion, so exact pre-task bytes were not captured for that one manifest. User explicitly approved restoration from independent781B provenance snapshot SHA2be651ec3dd33fb4b287bb1a5a7dba765bc00c7cff9abf6ddfbdf20277d80e7f; restore hash verified. Only freeze_timestamp_utc differs from incident version; query/GT/hash fields match. Earlier tests also re-emitted worklist/frozen-query CSV and benchmark reports, changing18file mtimes versus the prior audit. Their bytes match the initial P0 captured state except the explicitly restored manifest, but that capture overlapped baseline execution, so byte-exact pre-task parity for the re-emitted outputs is not proven. The frozen query CSV matches its independent protected SHA. Tests now use tmp metadata fixtures and an in-process historical-write guard; frozen benchmark source files are unchanged. Incident/recovery evidence retained; pre-task rollback is not claimed.

Cloud archive identity/current size was rechecked. Actual authenticated content-stream request failed HTTP413 because connector caps downloads at268,435,456B. **Cloud archive SHA UNVERIFIED**; matching metadata is not identity proof. No archive downloaded to C:/D:/E:; no overwrite. Cleanup/destructive migration remains blocked.

Live D48 folder still PermissionError even on explicit read outside sandbox; its existing ZIP/source references remain protected. Do not silently report empty directory as verified. The metadata freeze records this access limitation.

M0 historical checkpoint16,952,721B was retrieved into the new version and its SHA matched `ad5facec03008800536466c7d33f9232c3db2b71a1bd445073957bd6f4d0edbf`. Current66NPZ (3Train+all30Val pairs) match prior full-cache SHA snapshot, including original float16 exception. No NPZ was re-encoded. Nine small E0/E1/E2 config/metric/freeze reports were reread with fresh raw-content SHA. D200/E0 large checkpoints were not downloaded; their expected frozen hashes remain distinct from fresh content verification.

The final manifest links immutable base freeze, current observations, historical metrics/configs, real notebook bytes/source, environment, evaluator pin and new source hashes. Historical runtime CUDA lock remains TODO. Actual D500 notebook raw SHA `{nb['raw_sha256']}`,1904363B,23cells; all cell sources match the original audited source used by the parity oracle.
'''
(final/'backup_integrity_report.md').write_text(backup_report,encoding='utf-8')
write('cached_runner_parity_report.md',f'''# P0.1 — Cached E0 parity

VERIFIED source: fresh notebook05 source matches protected audit; M0 strict SHA/load PASS,4,228,873 parameters. Input order440Train/30Val; exact tensor/padding/targets on3Train(includingfloat16)+all30Val. Independent models/original-notebook AST loader/evaluator compare outputs/losses; forward/loss atol1e-6,rtol1e-5; inputs exact; metrics atol1e-8. Result `{parity['status']}`. Includes {', '.join(parity['loss_components'])}. M0 Val parity metrics are implementation checks, **not new M1 benchmark results**.

Training source preserves AdamW/AMP/StepLR400(per epoch)/clipping/patience/R1→mIoU selection. Seed is reset after model construction to match original Cell11 RNG, independently tested. Resume captures full RNGs including NumPy and fingerprints source/cache/runtime; it intentionally supports only its own versioned checkpoints. GPU optimizer/resume trajectory parity and full historical checkpoint reproduction were not run or claimed.

No raw videos/ZIP/extractor import. New runner cannot accidentally receive Test JSONL as a matching Train/Val checksum. Full470pair preflight must run on Colab's full existing cache; local integration intentionally downloads only33pairs. Dependencies/usage: cached_runner_dependencies.json and experiments/M1_lumae_ads/README_CACHED_D500.md.
''')
write('d1000_builder_readiness_report.md',f'''# P0.2 — Builder readiness

VERIFIED configurable coordinator supports target1000, parent500,new accepted required500. Metadata pool after identity/terminal-rejection exclusions: {len(plan['candidates'])} candidates; {len(plan['excluded'])} excluded. These are not approved videos. No downloads started. Inspected {len(extras)} raw files outsideD500: {dict(Counter(x['historical_state'] for x in extras))}; none auto-accepted. Exact files/hashes: outside_d500_raw_assessment.csv.

Synthetic tests verify deterministic1000 freeze, immutable parent, configured paths, inherited rejection, source-ID/URL/SHA dedup, HTTP Range interruption/resume, record acceptance/rejection/timestamps, default evidence validator at an external root, campaign conflicts and Train/Val-only feature references. New artifacts include explicit storage_binding, campaign evidence and cached-parent reuse policy. Legacy count-locked modules remain operational through their existing API,44legacy scaling tests and full suite PASS.

Actual live acquisition remains NOT_STARTED. Existing D500 server is retained; future operator uses generic coordinator and verified annotation producer contract, with explicit per-phase collection authorization. E:\\LUMAE is a configured future root, not populated/recreated/migrated. Default CLI is dry-run only; library fetch denies network by default. Review/campaign decisions, sufficient accepted sources and real frame/semantic evidence are required before actual D1000 freeze.
''')
write('d1000_evaluation_readiness_report.md','''# P0.3 — Evaluation readiness

VERIFIED implementation/tests: confirmed campaign connected components, identity/hash/URL collision checks, dhash near-match candidates, brand/product uncertainty quarantine, deterministic whole-group splits, exposure restrictions, nested140/440/700Train samples, human independent review/adjudication/revision chain and temporal IoU/boundary disagreement calculator. Synthetic1000video fixture reaches700/150/150videos,300Test queries,115independent doubles across1150query samples. No synthetic fixtures qualify as research data.

Actual readiness: NOT_EVALUATION_READY. New real Test registry/300distinct queries/human review records are absent; actual new human reviews=0; no empirical IAA is reported. All new Test rows must be human verified; >=10% final queries require independently submitted double annotations. Old Test30/other historically exposed video groups are not a fresh blind test. Alpha/checkpoints/model architecture must be frozen on Validation before Test. Historical model transfer evaluation is exploratory, not controlled scaling.

Operator CLI/workflow and external identity/permission responsibilities: docs/reproducibility/d1000_split_policy.md and human_annotation_protocol.md. This offline library validates attestations/evidence integrity, not the identity/truthfulness of a human without an external review authority.
''')
modified_tests=set(subprocess.check_output(['git','diff','--name-only','--','data_process/tests'],cwd=R,text=True).splitlines())
file_table='\n'.join('| `'+p+'` | '+('Modified: optional annotation storage root; legacy default retained' if p.endswith('/freeze.py') else 'Modified test: isolate writes into temporary metadata/output fixtures' if p in modified_tests else 'New source/config/test/documentation for corresponding P0 workstream')+' |' for p in changed_files)
file_table+='\n| `local_data/manifests/semantic_v3_clean_query_blind5_query_freeze.json` | Restored approved independent snapshot; recovery evidence and pre-task parity limitation recorded |'
test_table='\n'.join(f'| `{name}` | {tests[name]["tests"]} | {tests[name]["failures"]} | {tests[name]["skipped"]} | PASS |' for name in ('full_validation_guarded_final.xml','final_rng_annotation_tests.xml','builder_storage_tests.xml','isolated_query_freeze_test.xml'))
report=f'''# LUMAE — P0.0–P0.3 engineering report

Ngày03/11/2026,Asia/Saigon. VERIFIED=đo/đọc/test trực tiếp; INFERRED=suy luận từ evidence; ASSUMED=planning; TODO=validation/dữ liệu ngoài môi trường hiện tại. Pre-existing untracked `audit_outputs/` được bảo vệ; Git base `{base['git_commit']}`, không commit/rewrite history. New outputs: `{O}`. Không crawling, migration, xóa lịch sử, re-encode cache hoặc GPU retraining.

## A. Executive Summary

| Workstream | Trạng thái | Evidence |
| --- | --- | --- |
| P0.0 | VERIFIED local ZIP/raw; approved provenance recovery; cloud UNVERIFIED; live D48 access unresolved | final freeze + incident/recovery + HTTP413 evidence |
| P0.1 | IMPLEMENTED; CPU input/forward/loss/evaluator parity PASS | cached_runner_parity_results.json; fresh M0+33pairs; final seed test |
| P0.2 | IMPLEMENTED; synthetic1000 + real-source metadata dry-run PASS | builder_storage_tests.xml; d1000_current_pool_dry_run_v2.json |
| P0.3 | IMPLEMENTED and tested; actual evaluation NOT_READY | campaign/review tests, policy/workflow; real human reviews0 |

Full guarded final suite {tests['full_validation_guarded_final.xml']['tests']}PASS outside sandbox after RNG/storage and fixture-isolation fixes. Targeted5cached/RNG/annotation and7builder tests also PASS (overlap, not additional unique counts). Compile/diff checks PASS. **A legacy test accidentally rewrote one provenance timestamp; it was restored from a verified snapshot with explicit user approval.** Prior legacy tests also re-emitted small reports/worklist/frozen-query CSV. Exact pre-task byte parity for these re-emitted outputs is unproved; source video/checkpoints/canonical split hashes are verified. Remaining content differences versus initial P0 snapshot, other than approved restore:0. See incident evidence; do not call this an entirely untouched historical tree.

## B. Research Freeze and Backup Integrity

New immutable reference: [final manifest](freeze_final/research_freeze_manifest.json),[checksums](freeze_final/research_asset_checksums.csv),[historical experiments](freeze_final/historical_experiments_manifest.json),[backup report](freeze_final/backup_integrity_report.md). Links to base freeze with {len(base['assets'])} content-hashed files rather than copying23GB of raw/ZIP data. Dataset contains500uniquevideos,440/30/30rows,8,836,814,052raw bytes. All500raw hashes match frozen D500 source hashes. Local ZIP size8,844,848,885B and SHA132621746f9d2538d199e8c5cba1ddec0f0586854ae26c3628478de83ef19345 verified anew.

Cloud corresponding archive was located and fresh size/identity read. Authenticated stream attempt was rejected by provider's256MiB cap; cloud SHA remains UNVERIFIED. This blocks cleanup/destructive migration only. M0 content is fresh-verified; D200/E0 checkpoints retain current metadata + historical expected hashes, not freshly claimed content parity. Nine current E0/E1/E2 config/metric/freeze observations preserve historical metrics and selection settings, not new experiment claims. Historical GPU environment remains TODO.

Real notebook05 raw SHA `{nb['raw_sha256']}`,23cells; current cell sources match original parity oracle. Historical D48 live directory still raises PermissionError even on explicit read outside sandbox; existing D48 ZIP remains protected and inventoried. Metadata observations/expected hashes are never mislabeled fresh content verification. Float16 Train feature exception retains original bytes; current33pairs match prior SHA. Original annotation provenance/human states and frozen Val/Test remain unchanged.

## C. Cached-Feature Runner

Implemented `cached_runner.py` and frozen D500 config; usage in README_CACHED_D500.md. Resolves combined feature manifest across D200/newD500 caches, checks qid/vid coverage/splits/hash/schema, preserves float32 cast/L2/truncation, generates two TEF endpoints at runtime, uses official span targets/collate/matcher/loss and evaluator. No MP4/ZIP/CLIP extraction dependency. M0 strict load verified,4,228,873parameters. Existing M1-D200 is not initialization.

Effective E0 recipe verified from actual notebook: AdamW1e-5,wd1e-4,batch4,AMP,50epochs,patience10,clip0.1,StepLR400/gamma0.1 **per epoch**,seed20260930; select Val R1@0.5 then rounded mIoU. Reset RNG after module construction matches Cell11's already-built-model reseeding. Fresh output directories and strict resume fingerprints bind recipe/cache/runner/evaluator/runtime. Resume restores optimizer/scheduler/scaler/Python/NumPy/Torch/CUDA/DataLoader RNG. No Test argument/GT enters checkpoint selection.

Independent oracle extracts original notebook AST without executing notebook. All440Train/30Validentifier order checked; exact tensors/TEF/padding/spans on3realTrain+all30Val, includingfloat16. Independent strict M0 loads, forward/loss components and fullVal30officialmetrics match: inputs exact, outputs/loss atol1e-6 rtol1e-5, metrics atol1e-8. See cached_runner_parity_report.md and JSON. CPU M0 comparison is not a new fine-tuned E0 benchmark. Full940NPZ local preflight is pending full-cache Colab access; no full GPU training reproduction is claimed.

## D. D1000 Builder

Configurable `BuildConfig`/`BuilderSession` alongside legacy count-locked modules; optional root argument extends established annotation validator. Default config proposes E:\\LUMAE, keeps parent D500 in repo, no E: hardcoding in model logic. `storage_binding.json` distinguishes original parent assets and new assets without modifying historical path strings. Existing D200/D500 server behavior is retained rather than making its hardcoded limits silently generic.

Seeded candidate selection excludes parentIDs/URLs/hashes and inherited terminal rejections; append-only states retain accepted/rejected/retry records. Downloads are denied by default; future explicit collector reuses verified HTTPRange/.part handling. Probe/hash/evidence/timestamp gates precede acceptance; accepted JSON/parent are immutable. Freeze revalidates source,accepted hashes and full annotation evidence and writes deterministic version manifests/checksums/COMPLETE. New Train/Val asset manifests exclude Test qids/features/GT; parent feature action is BIND_EXISTING_CACHE_NO_REENCODE, new features EXTRACT_NEW_ONCE, readiness false until verified.

Real metadata-only plan: {len(plan['candidates'])}source candidates after exclusions, parent500,new accepted required500,download startedFalse. Inspected116raw outsideD500 with current hashes/statuses: {dict(Counter(x['historical_state'] for x in extras))}. None counted accepted. Synthetic1000construction, resume/Range/idempotency/dedup/immutability/path/evidence/campaign/Test-export tests PASS. Actual collection not started; sufficient approved records and operator visual review are still prerequisites.

## E. Dataset Split and Human Verification

Targets700/150/150uniquevideos and>=300freshTestquerymoments,>=2distinctqueries perTestvideo. Whole confirmed campaign/source/duplicate components stay in one split; brand/product/dhash candidates create quarantine, never independent-campaign proof. Uncertain matches require reviewed SAME/DISTINCT evidence. Default unknown exposure is true; fresh Test only uses reviewed unseen groups. Exact-count deviations are reported; leakage prevention wins. Source/URL/hash aliases must be deduplicated before final freeze.

Blind review packet omits AI GT/other boundaries. Actual independent human intervals, canonical identities,timestamps,comments and separate adjudicator are required; every revision hash-chained; later annotation invalidates prior verification. At least10%finalqueryset receives independent doubles. Temporal IoU and start/end disagreement are computable only from actual supplied reviews; missing pairs yield null, not fabricated agreement. CLI annotate response hides other annotations. External operator identity/account permissions are still operational responsibilities.

Nested140/440/700Train query subsets share new frozen Validation/Test, feature/init/evaluator/budget. Historical checkpoints on new Test are exploratory transfer, not controlled scaling. No Test tuning ofalpha/checkpoints/architecture. Synthetic1150query fixture validates300Testqueries+115doubles; **actual human reviews0**, no real agreement scores or real D1000 split created. See two policy documents and d1000_evaluation_readiness_report.md.

## F. Validation

| Evidence | Tests | Failures | Skipped | Status |
| --- | ---: | ---: | ---: | --- |
{test_table}

Final command: `python -m pytest data_process/tests backend/colab/test_lumae_api.py -q -x -p no:cacheprovider --basetemp research_outputs/p0_t16 --junitxml=research_outputs/p0_20261103_01/full_validation_guarded_final.xml` with LUMAE_P0_ASSETS pointing to verified new assets. Guarded full{tests['full_validation_guarded_final.xml']['tests']}PASS. Last targeted tests cover RNG/default annotation validation and storage bindings. `python -m compileall -q data_process experiments scripts backend/colab` with redirected pycache, `git diff --check`, frozenD500checksums/sourcehashes and protected-state comparison (approved restoration explicitly separated) PASS. Existing44legacy scaling tests independently PASS.

Sandbox baseline failures were environment-specific: NTFS os.link PermissionError and Streamlit local listener bind failure; backendTestClient also stalled in sandbox. Runs were stopped/bounded and retained as earlier evidence. A single full rerun outside sandbox passed282, including UI/backend. Initial builder str/Path guard defect was found by tests, fixed and revalidated; no unresolved final test failure. Earlier attempt XMLs are not substituted for final success evidence.

No GPUtrain, fullhistoricalcheckpointreproduction, actualcrawling, rawmigration, realhumanreviews or cloudZIPcontenthash was executed/claimed. No unnecessary dependencies installed or historical environments modified.

## G. Changed Files

| File | Purpose |
| --- | --- |
{file_table}

All new research metadata, test evidence, fresh M0/33pair parity fixtures and pinned source snapshot live in research_outputs versioned paths. Pre-existing audit_outputs/2026-11-02 remains untouched. Production change to the legacy builder is freeze.py's optional root resolution; legacy tests are additionally isolated/hardened. Frozen semantic benchmark source files were retained unchanged. The approved provenance restoration is separate from source changes. Complete source SHA/patch are included; no new commit was made.

## H. Remaining Blockers

- **Code defects:** none known in validated scope; GPU optimizer/AMP/resume trajectory is unexecuted and must be tested before production training.
- **Backup/access:** cloud archivecontent hash blocked by connector256MiB limit; liveD48read PermissionError remains. No credentials requested; existing connected Drive works for metadata/small files. Use an authorized large-file streaming client and verify cloud SHA before cleanup; do not grant destructive migration from this report.
- **Research integrity incident:** earlier baseline tests re-emitted18small outputs (see prior_audit_metadata_deltas.json). One timestamp differs in content and was restored with approval; other final bytes match the initial P0 capture, but that capture overlapped tests, so exact pre-task parity is not proved for those outputs. The new read-only guard and isolated metadata fixtures prevent repeat writes. Do not describe the entire historical tree as untouched.
- **Missing data:** no500newacceptedvideos; raw116notaccepted; approved campaign registry/source aliases/supplemental distinctTestqueries and full local470paircache are absent. Colab fullcache preflight is still required.
- **Human reviews:** realnewreviews0; no empiricalIAA;300Testqueryhuman verification and>=10%doubleannotation pending.
- **Future GPU experiment:** authorize a separate run with historical/package/CUDA lock and documented budget if full checkpoint reproduction or controlled D1000 scaling is desired. CPUforwardparity does not establish finalcheckpoint parity.

## I. Next Execution Stage

Ready now: offline metadata planning, explicit root bindings, inherited-rejection/source dedup, resumable collector API with network deny-default, full annotation evidence acceptance, deterministic dataset freeze, group-aware quarantine/split policy and blind review CLI; cached E0 entrypoint and CPU parity evidence. Next **authorized collection** phase must pin pool/roots, verify backup, inspect candidate provenance/near-duplicates, explicitly authorize video acquisition, run operator annotation review, and accumulate500validuniqueaccepted sources. No URL/file count substitutes acceptance. E:\\LUMAE has not been populated or recreated; capacity/independent backup must be checked before live collection. Build new unseen Test registry/300genuinehumanqueries; only then freeze evaluation and authorize GPUexperiments. Do not start crawling or cleanup automatically.
'''
write('p0_engineering_report.md',report)
write('final_status.json',{'P0.0':'LOCAL_DATA_VERIFIED_APPROVED_PROVENANCE_RECOVERY_CLOUD_UNVERIFIED_D48_ACCESS_BLOCKED','P0.1':'IMPLEMENTED_CPU_PARITY_PASS','P0.2':'IMPLEMENTED_SYNTHETIC_D1000_PASS_NO_LIVE_COLLECTION','P0.3':'IMPLEMENTED_TESTED_REAL_EVALUATION_PENDING','tests':tests,'protected_state':readonly,'historical_modifications':{'approved_provenance_restore':recovery,'re_emitted_output_metadata':str(O/'prior_audit_metadata_deltas.json'),'exact_pre_task_byte_parity_for_reemitted_outputs':'UNPROVED'},'live_downloads':0,'GPU_training_runs':0,'new_human_reviews':0})
print(json.dumps({'protected_files':verified,'cloud_backup':backup['status'],'raw500_pass':True,'extra_raw':len(extras),'metadata_candidates':len(plan['candidates']),'report':str(O/'p0_engineering_report.md')},ensure_ascii=True))
