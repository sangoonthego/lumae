# D1500 Test-450 runbook

Run commands from the LUMAE repository root. The local defaults resolve the frozen manifests, historical NPZs, raw media and pinned Moment-DETR snapshot described in the [initial audit](d1500_test450_initial_audit.md). Use Python with NumPy, PyTorch, SciPy, pandas, tqdm, Pillow and the pinned CLIP extractor's dependencies. FFprobe and FFmpeg must be on PATH. Versioned output is under `research_outputs/d1500_test450_ai_v1`.

On another host, stage the repo-relative artifacts at the same relative paths and set `LUMAE_D1500_STORAGE_ROOT` to the staged equivalent of `E:/LUMAE`. Preserve the directory structure below that storage root. A preflight checks staged bytes against the original pins. The release and run manifests record the resolved bindings for the execution host.

## Visual annotation and independent review

```powershell
python -B -m data_process.test450 audit
python -B -m data_process.test450 expand-test --dry-run
python -B -m data_process.test450 expand-test --all
$vid = "lumae_ads_72033bb15133eb17fb2433eb2e81b18e"
python -B -m data_process.test450 expand-test --vid $vid
python -B -m data_process.test450 boundary-evidence --vid $vid --role generation --range 3 5
python -B -m data_process.test450 record-generation --vid $vid generation.json
python -B -m data_process.test450 verify-annotations --vid $vid
python -B -m data_process.test450 record-review --vid $vid review.json
```

Inspect the full generation sheet, make a short event inventory in the generation decision and choose two unique, visually observable events distinct from the original. Use decoded frames around the boundaries and keep the final queries specific. For each proposed interval, record what is visible before, during and after. Request 0.1-second boundary evidence for short events or ambiguous transitions. Frame timestamps come from FFmpeg. The visual estimate is limited by the source frame cadence and sampled evidence.

For this checkout, use `--attempt 2` to resume the overview packets already staged for all 150 videos. The pilot's accepted pair is under attempt 003.

Give the second AI session only the review task and the JPEGs in its review folder. It may request its own `boundary-evidence --role review` ranges. The task contains exact candidate query text, a source SHA, duration and a separate visual sampling grid. Its contents exclude the proposed intervals, generator notes and model predictions. Save both decisions under distinct session IDs. The CLI records evidence hashes and enforces separate declared sessions. The operator is responsible for presenting only the blind packet to the reviewer.

The decision JSON has the following shape. Replace the example values with the inspected task and evidence.

```json
{
  "agent": "visual inspection operator identity",
  "model": "model name and available revision",
  "session": "unique session identifier",
  "timestamp_utc": "2026-11-13T17:00:00Z",
  "task_sha256": "hash printed by the packet command",
  "boundary_packets": ["3.000000_5.000000"],
  "events": [
    {
      "query": "Specific event observed in the video.",
      "decision": "accept",
      "interval": [3.1, 4.8],
      "confidence": 0.9,
      "unique_occurrence": true,
      "note": "Describe the actual visual changes before, inside and after.",
      "evidence": {"before": [3.0], "inside": [3.5, 4.5], "after": [4.9]}
    },
    {
      "query": "Another candidate query.",
      "decision": "abstain",
      "note": "Explain the ambiguity or missing evidence."
    }
  ]
}
```

An attempt is accepted as a pair after two visual acceptances per query, both tIoUs at least 0.70 by default, interval/evidence checks and normalized within-video uniqueness. The final intervals use the generator's visually localized estimates, with both estimates retained. Use `--attempt 2` and increment for fresh review, denser evidence or revised wording. For a later diversity conflict, use `retire-attempt --vid VID --attempt N reason.json` with `agent` and `reason`. The ledger keeps the earlier files and decisions.

```powershell
python -B -m data_process.test450 progress
python -B -m data_process.test450 build-text-cache
python -B -m data_process.test450 quality
python -B -m data_process.test450 publish
python -B -m data_process.test450 validate-benchmark
```

The text cache uses the frozen extractor and weights, 77 positions, at most 32 valid tokens and float32 per-token hidden states. The QID and query hash bind each new NPZ. A changed query can be encoded without overwriting a retired attempt's cache. The quality command checks normalized strings against the original corpus and uses CLIP token-state cosine and word Jaccard for new-query pairs and new-versus-original pairs within one video. Treat these scores as screening signals. Record `diversity_resolutions.json` as an object keyed by the printed pair ID, with `reviewer`, `reason` and `distinct_events: true` after visual inspection. Revise any actual duplicate and rerun both passes. Use the same `--agreement` and `--similarity` values for reports and publication.

Publication requires exactly two accepted rows and two valid new text features per Test VID, diversity resolution and no exact duplicates. The release manifest is the commit point. It binds parent checksums, evidence provenance and original and new cache files. The expanded JSONL begins with the original Test bytes and appends new rows in VID and slot order. Validation rederives and compares the release and verifies cached arrays. The parent Train and Validation remain in the original freeze. Model training uses only their sealed JSONLs and an independently built Train/Val feature index.

## Training matrix and common evaluation

```powershell
python -B -m data_process.test450 prepare-training-matrix --dry-run
foreach ($seed in 42,123,2026) { python -B -m data_process.test450 training-preflight --seed $seed }
python -B -m data_process.test450 prepare-training-matrix
# On a verified CUDA runtime, opt in to a registered Moment-DETR run.
python -B -m data_process.test450 run-training --seed 42 --execute
# Resume that run using its own saved state and matching inputs.
python -B -m data_process.test450 run-training --seed 42 --execute --resume
```

The matrix records the nine registered combinations and their preflight statuses. Moment-DETR reuses the M0 initialization, 2-second video features, TEF, the E0 optimizer/scheduler and Train/Val labels. Python, NumPy, Torch CPU/CUDA and the shuffled DataLoader generator are seeded. Zero loader workers removes worker RNG state from this recipe. CUDA requests deterministic algorithms and records environment and GPU identity. GPU execution parity remains unverified. The selection criterion is Val R1@0.5 followed by Val mIoU. Each run has its own directory, best and latest checkpoints, validation predictions and metrics, and per-epoch timings. QD-DETR and UniVTG remain gated until checkpoint, adapter, stream alignment and training supervision preflights are available.

```powershell
python -B -m data_process.test450 evaluate --split val --run val_m0_example --moment-detr
python -B -m data_process.test450 evaluate --split test450 --run locked_moment_seed42 --moment-detr --lock-test-config --checkpoint research_outputs/d1500_test450_ai_v1/runs/moment_detr/seed_42/best.ckpt
python -B -m data_process.test450 evaluate --split test450 --run locked_univtg_seed42 --predictions univtg_seed42_seconds.jsonl --external-model univtg --external-seed 42 --lock-test-config
python -B -m data_process.test450 aggregate-seeds --name moment_detr_test450 --runs locked_moment_seed42 locked_moment_seed123 locked_moment_seed2026
```

External adapters must export `qid`, `vid` and `pred_relevant_windows` as scored `[start_s,end_s,score]` triples, one row per query including explicit empty predictions. They may supply `--measurements` with checkpoint SHA-256, forward latency per query, device and peak allocated GPU bytes. Moment-DETR converts center/width predictions by video duration, clips to valid times and preserves score ranking. The common evaluator reports historical Test-30, other original Test-120, original Test-150, new Test-300 and combined Test-450. MR mAP uses the pinned upstream AP implementation over IoU 0.50 to 0.95 and the first 10 predictions. Short events are at most 2 seconds. Bootstrap intervals draw whole videos, retaining all their queries. Seed aggregation reports mean and sample standard deviation separately. Test evaluation requires a finalized configuration and an explicit lock flag.
