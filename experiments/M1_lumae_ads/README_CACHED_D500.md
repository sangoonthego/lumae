# D500 E0 cached-feature runner

Source of truth: notebook05 D500 E0 Cells10/11, upstream commit `b7e553ac3b0c898ee6b85e03ee507c064eab89ca`; not notebook03 D48. Original notebooks and frozen inputs are unchanged. The runner imports no video downloader, CLIP extractor or ZIP reader.

## Colab setup

Copy the new `cached_runner.py`, `config_d500_cached.json`, and the existing Python `data_process` validation/schema packages into the LUMAE source checkout. Keep that source revision/patch with every run. Install the dependencies already required by the official Moment-DETR model: Torch/CUDA, NumPy, SciPy, Pandas, tqdm, scikit-learn. No CLIP weights, ffmpeg or raw MP4 are needed for this runner. Do not replace an established environment merely to match an unpinned latest package version.

Use the exact upstream commit, including a clean working tree:

```bash
git clone --no-checkout https://github.com/jayleicn/moment_detr.git /content/moment_detr_pinned
git -C /content/moment_detr_pinned checkout --detach b7e553ac3b0c898ee6b85e03ee507c064eab89ca
```

Alternatively use a verified `SOURCE_MANIFEST.json` snapshot from the versioned P0 validation artifacts. The runner rejects modified/mismatched pinned sources. It never fetches repository HEAD.

Mount Google Drive. Materialize **only the small Train/Val JSONL files** from the existing frozen inputs if they are still stored solely inside the D500 ZIP; this is a one-time setup separate from the entrypoint. Keep their exact SHA. The runner itself never opens/extracts an archive. Direct Train/Val copies can also be uploaded independently. Test JSONL is not an argument.

The feature manifest is the combined `CV_Lumae/artifacts/m1_d500_features/feature_manifest.json`. Its paths reference **both** D200 and incremental D500 cache directories. Set `--feature-root` to `CV_Lumae`, not to `m1_d500_features`. Do not rename the historical D200 `qid...npz` files; resolution follows the manifest. Do not re-encode or overwrite the historical float16 exception: `lumae_ads_008ff6f5e15fae24e9a72bc79deae917.npz`. `.astype(float32)` preserves original notebook loading behavior.

```bash
python -m experiments.M1_lumae_ads.cached_runner preflight \
  --train-jsonl /content/frozen/lumae_ads_d500_train.jsonl \
  --val-jsonl /content/frozen/lumae_ads_d500_val.jsonl \
  --feature-manifest /content/drive/MyDrive/CV_Lumae/artifacts/m1_d500_features/feature_manifest.json \
  --feature-root /content/drive/MyDrive/CV_Lumae \
  --upstream-root /content/moment_detr_pinned \
  --checkpoint /content/drive/MyDrive/CV_Lumae/checkpoints/moment_detr/pretrained/model_best.ckpt \
  --output /content/preflight_unique_run.json
```

After a separate GPU execution authorization, use `train` with the same inputs and a **new** versioned `--output` directory. `--resume` is only for that runner's existing directory with matching full input/config fingerprint, `best.ckpt` and `latest.ckpt`. It cannot resume the original notebook's checkpoint format silently. Historical checkpoints never serve as output targets. Changing cache bytes, manifest or recipe blocks resume.

Recipe: AdamW1e-5, weight_decay1e-4, batch4, AMP,50epochs, patience10, gradient_clip0.1, StepLR400/gamma0.1 stepped **per epoch**, seed20260930. Model/matcher/loss options come from strict SHA-verified QVHighlights M0; saliency loss0. Normalization flags/scheduler values are frozen from actual D500 E0 configuration instead of requiring M1-D200 weights at runtime. Validation selects R1@0.5, then rounded mIoU, with exact tie behavior preserved. All proposals and official evaluator behavior are preserved.

Resume additionally saves NumPy RNG state (the original notebook omitted it). It preserves Python/Torch/CUDA RNGs, DataLoader generator, optimizer, scheduler, scaler, history and patience state. This is a deliberate resume integrity improvement, not a claim of identical interrupted GPU trajectories across runtime versions.

## Validation scope

`test_cached_runner.py` extracts the original loader/evaluator AST without executing the notebook. Inputs/padding/targets require exact equality; predictions/loss use atol1e-6/rtol1e-5; metrics atol1e-8. CPU validation uses current M0,3 real Train samples (including float16) and all30 Val samples, both independent model loads, official matcher/loss/evaluator. All440/30 identifier order is checked. Full940 local NPZ preflight requires the full existing feature cache on Colab; the local test assets intentionally contain only66 NPZ.

No GPU optimizer steps or historical retraining were executed. Full historical checkpoint reproduction remains a separately authorized experiment with the exact historical CUDA/package/runtime conditions.
