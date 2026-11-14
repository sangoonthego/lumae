# D1500 Test-450 implementation report

## A. Executive summary

The versioned workflow stages timestamped Test evidence, records separate AI generation and blind review decisions, extracts CLIP text features, gates publication and prepares the three-model, three-seed experiment matrix. All 150 Test videos have staged evidence. One video completed the two-pass pilot, producing two accepted AI pseudo-labels and two text caches. The remaining 149 videos require generation and blind review. The release gate correctly blocks Test-450 publication.

Moment-DETR passed strict M0 loading and a CPU forward/loss check at all three registered training seeds. QD-DETR and UniVTG are blocked at checkpoint and feature/supervision integration. All nine training executions remain pending.

## B. Repository audit

The [initial audit](d1500_test450_initial_audit.md) records Git base `f2c532decbe47448ad28ee3c82881f412893c469`, source paths, frozen split SHA-256, the canonical dataset pin, upstream source snapshot and historical feature bindings. The [final checksum audit](../../research_outputs/d1500_test450_ai_v1/final_audit.json) re-read all 3,000 original NPZs, all eight sealed split files, the canonical dataset, the M0 checkpoint and the hashes and video streams of all 150 Test media. The original split still has 1,500 distinct VIDs and 1,500 distinct QIDs and one valid interval per query. The frozen original Test remains available independently.

The historical Drive manifest identifies [D500 E0](https://drive.google.com/file/d/1wGsJ0GxQb7iK6kBS4fqpYSjHhA-QpPmM/view) and [D200 fine-tuned](https://drive.google.com/file/d/1YAtAvqF1IVET2sTaBTTLwBqPDxLpO9O2/view) artifacts. Their local bytes were unavailable and their weights were excluded from initialization.

## C. Dataset expansion

The final column counts available originals plus accepted pilot rows. The latter remain in the versioned review ledger.

| Split | Videos | Original queries | New accepted | Available queries |
|---|---:|---:|---:|---:|
| Train | 1,200 | 1,200 | 0 | 1,200 |
| Validation | 150 | 150 | 0 | 150 |
| Test | 150 | 150 | 2 | 152 |
| Total | 1,500 | 1,500 | 2 | 1,502 |

## D. Annotation quality

Two queries passed on the pilot video. The first blind review rejected the camera pullback wording. A fresh independent reviewer accepted the revised island-cut wording and the vent event. The ledger contains one rejected candidate, one revised attempt and 149 unresolved videos. Mean agreement for the two accepted queries is 0.9812 tIoU. Mean self-reported generation confidence is 0.94 and mean review confidence is 0.99. The pilot returned zero exact duplicates or similarity flags using the 0.94 cosine and 0.75 word Jaccard screens. These two examples cannot estimate corpus-wide quality or agreement. See [progress and review states](../../research_outputs/d1500_test450_ai_v1/annotation_progress.json).

| Cohort | ≤2 s | (2,4] s | (4,6] s | >6 s |
|---|---:|---:|---:|---:|
| Train-1200 | 463 | 388 | 244 | 105 |
| Val-150 | 58 | 42 | 21 | 29 |
| Original Test-150 | 60 | 41 | 31 | 18 |
| New accepted pilot | 2 | 0 | 0 | 0 |
| Original Test plus pilot | 62 | 41 | 31 | 18 |

## E. Feature cache

The audit verified 1,500 reusable video NPZs and 1,500 reusable text NPZs, with zero missing or invalid files. The single historical float16 video cache remains readable. The two new pilot text NPZs use 68,938 bytes. The real Moment-DETR loader produced the same `[9,514]` video tensor for the three query rows of the pilot video and distinct `[L,512]` text tensors. No video extraction ran. The Train/Val index contains exactly 1,350 VIDs and 1,350 QIDs, all drawn from the sealed Train and Val manifests.

## F. Model compatibility

| Model | Checkpoint | Loader and features | Preflight and remaining work |
|---|---|---|---|
| Moment-DETR | M0 verified by SHA and strict 92-tensor load | Cached CLIP video 512 plus TEF 2, text 512, L1 temporal spans and zero saliency loss | CPU forward and loss pass. CUDA training needs a GPU-enabled PyTorch runtime. |
| QD-DETR | QR-DETR `videoonly.ckpt` candidate identified | Standard recipe expects aligned SlowFast and CLIP plus TEF and saliency supervision. Candidate layout and D1500 adapter remain unverified. | BLOCKED. Stage bytes, verify checkpoint identity and projections, then bind aligned features and implement Train-only supervision. [Drive candidate](https://drive.google.com/file/d/1fDFuFXCww_8FeHip-PMNgKj3i1zMHtxL/view) · [Upstream](https://github.com/wjun0830/QD-DETR) |
| UniVTG | `model_best.ckpt` and `opt.json` candidates identified | Upstream has CLIP-only and SlowFast+CLIP variants. Candidate layout and D1500 adapter remain unverified. | BLOCKED. Stage both files and verify state dict, options, text representation and training supervision. [Drive candidates](https://drive.google.com/drive/folders/1rGA6tCRpSlQIWR_5fxagXFsSKrNNrTTR) · [Upstream model zoo](https://github.com/showlab/UniVTG/blob/main/model.md) |

Drive returned authenticated references for the three candidate files. Local staging through those references failed with DNS `ENOTFOUND`. Their tensors and exact configuration remain unverified. No feature channels or training labels were fabricated.

## G. Experiment matrix

The [nine-run JSON](../../research_outputs/d1500_test450_ai_v1/training_matrix.json) holds initialization, feature recipe, seed, selection split, configuration and full planned checkpoint paths. Each checkpoint path is under the new version's `runs/<model>/seed_<seed>/best.ckpt`. Moment-DETR uses the same AdamW, AMP, StepLR and early-stopping recipe across seeds. The blocked rows record upstream feature expectations and carry no executable training recipe.

| Model | Seed | Preflight | Execution |
|---|---:|---|---|
| Moment-DETR | 42 | PASS, CPU | NOT RUN |
| Moment-DETR | 123 | PASS, CPU | NOT RUN |
| Moment-DETR | 2026 | PASS, CPU | NOT RUN |
| QD-DETR | 42 | BLOCKED | NOT RUN |
| QD-DETR | 123 | BLOCKED | NOT RUN |
| QD-DETR | 2026 | BLOCKED | NOT RUN |
| UniVTG | 42 | BLOCKED | NOT RUN |
| UniVTG | 123 | BLOCKED | NOT RUN |
| UniVTG | 2026 | BLOCKED | NOT RUN |

## H. Files and artifacts

- Added `data_process/test450` with the CLI, parent verification, annotation, release, training and evaluation modules. Added regression coverage in `data_process/tests/test_test450.py`.
- Extended `experiments/M1_lumae_ads/cached_runner.py` with an optional feature resolver and optional deterministic training mode. The historical callers retain their original defaults.
- Added the initial audit, this report and the [runbook](d1500_test450_runbook.md). New local evidence, preflights, caches, matrix and Val evaluation outputs are under `research_outputs/d1500_test450_ai_v1`. That directory follows the repository's ignored research-output convention.
- Reverified protected split files, parent canonical data, 150 Test media, 3,000 feature NPZs and the M0 checkpoint. Historical model outputs were read for audit context.

## I. Validation

| Command | Result |
|---|---|
| `git status --short`, `git diff --check` | PASS. Only the scoped source/docs/tests are pending, with a Windows LF/CRLF advisory. |
| `npm test` | PASS. 10 tests. |
| `python -B -m pytest -q -p no:cacheprovider --basetemp research_outputs/d1500_test450_ai_v1/pytest_release data_process/tests/test_test450.py data_process/tests/test_cached_runner.py` | PASS. 6 passed, 3 skipped. Includes a synthetic 450-query publication and byte-preservation check. |
| `python -B -m pytest -q -p no:cacheprovider --basetemp research_outputs/d1500_test450_ai_v1/pytest_full data_process/tests backend/colab` | FAIL observed. The run emitted multiple failures and was interrupted after it stopped emitting progress near the end. |
| `python -B -m pytest -x -q -p no:cacheprovider --basetemp research_outputs/d1500_test450_ai_v1/pytest_first_failure data_process/tests backend/colab` | FAIL. 112 passed, 3 skipped, 1 failed. The existing `test_playwright_clean_blind_dom_blindness` could not bind Streamlit to a local port within 25 seconds. |
| `python -B -m data_process.test450 audit` | PASS. 1,500 VIDs, 1,500 QIDs, 150 Test media and 3,000 pinned NPZs. |
| `python -B -m data_process.test450 expand-test --dry-run` | PASS. Target 150 videos and 300 queries. |
| `python -B -m data_process.test450 expand-test --all --attempt 2` | PASS. 150 evidence packets, zero staging errors. |
| `python -B -m data_process.test450 build-text-cache` | PASS. Two pilot NPZs. |
| `python -B -m data_process.test450 quality` | PASS for the pilot. Two queries screened, zero exact duplicates and zero similarity flags. |
| `python -B -m data_process.test450 validate-benchmark` | BLOCKED. Release absent, 2/300 accepted, 149 unresolved videos. |
| `python -B -m data_process.test450 publish` | Gate triggered on incomplete expansion. No Test-450 manifest was written. |
| `python -B -m data_process.test450 prepare-training-matrix --dry-run` | PASS. Nine entries, three CPU PASS and six BLOCKED. |
| `foreach ($seed in 42,123,2026) { python -B -m data_process.test450 training-preflight --seed $seed }` | PASS. Output shape `[2,10,2]` and finite losses for each seed. |
| `python -B -m data_process.test450 evaluate --split val --run m0_val_cpu_final --moment-detr --bootstrap 50` | PASS. R1@0.5 12.00%, R1@0.7 4.00%, MR mAP 7.19%, mIoU 23.44%, Oracle@5 28.00%, short R1@0.5 0.00%, CPU forward 15.69 ms/query in this run. [Metrics](../../research_outputs/d1500_test450_ai_v1/evaluations/m0_val_cpu_final/report.json) |
| `python -B -m data_process.test450 run-training --seed 42 --execute` | BLOCKED at the CUDA check. No optimizer step or training output. |
| Actual GPU training, peak GPU memory, locked Test evaluation | NOT RUN. |

## J. Scientific limitations

The benchmark remains AI-pseudo-label evaluation. The pilot used distinct reasoning sessions in the same Codex model family. Human agreement and independent cross-model agreement are unmeasured. The frozen campaign audit lists 203 unverified cross-split near-duplicate edges, and Legacy Test-30 has prior evaluation exposure. New queries on a shared video are statistically dependent. The evaluator uses video-level bootstrap resampling and keeps training-seed variability separate. GPU determinism, non-Moment-DETR checkpoint compatibility, feature timeline alignment and remaining annotation quality require verification.

## K. Next research step

Stage the QD-DETR and UniVTG candidates and run strict CPU checkpoint, feature and label preflights. In parallel, complete the 149 remaining blind annotation pairs and resolve diversity flags. Then run the nine validation-selected experiments on verified CUDA, run Ablation Study A and controlled TEF, sampling and data-scaling studies using Validation, and lock the final configurations before evaluating the published Test-450.
