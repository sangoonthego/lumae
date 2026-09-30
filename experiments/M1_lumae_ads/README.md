# M1: LUMAE Ads v1 fine-tuning

The frozen handoff contains 48 videos: 18 human-verified temporal labels and 30 unverified AI pseudo labels. Train has 18 human and 16 pseudo labels. Validation and test each have seven pseudo labels. All 30 pseudo intervals have LOW semantic confidence and zero ranking margin. Validation scores therefore measure agreement with pseudo labels, not human-grounded accuracy. Keep test sealed until model selection is complete.

## Colab prerequisites

- CUDA runtime and the portable LUMAE input bundle. Build it locally with `python -m experiments.M1_lumae_ads.train_frozen package --lumae-root . --bundle-path local_data/exports/lumae_ads_v1_colab.zip`, then upload it to Drive. The notebook extracts it under `/content/lumae`. It contains the frozen dataset, handoff, runner, source CSVs, and all 48 videos.
- A local checkout of the [official Moment-DETR repository](https://github.com/jayleicn/moment_detr) with its CLIP-only pretrained checkpoint at `run_on_video/moment_detr_ckpt/model_best.ckpt`. Supply only a checkpoint from a trusted source; PyTorch needs to unpickle its saved options.
- Dependencies required by the upstream checkout: PyTorch, `ffmpeg`/`ffprobe`, `ffmpeg-python`, `ftfy`, `regex`, `tqdm`, `easydict`, `tensorboard`, `tabulate`, `scikit-learn`, `pandas`, and `numpy`. The upstream project was tested on an older Python/PyTorch stack, so confirm compatibility on the selected Colab image before a full run.

Use `03_Lumae_M1_ProductAds.ipynb` for the ordered commands. The equivalent CLI is:

```bash
python -m experiments.M1_lumae_ads.train_frozen preflight --lumae-root /content/lumae
python -m experiments.M1_lumae_ads.train_frozen features --lumae-root /content/lumae --moment-detr-root /content/moment_detr --feature-root /content/m1_features
python -m experiments.M1_lumae_ads.train_frozen train --lumae-root /content/lumae --moment-detr-root /content/moment_detr --checkpoint /content/moment_detr/run_on_video/moment_detr_ckpt/model_best.ckpt --feature-root /content/m1_features --results-root /content/m1_results
```

Run commands with `/content/lumae` on `PYTHONPATH`, or from that directory. The runner verifies handoff hashes and provenance, extracts official 2-second CLIP image/text features, fine-tunes temporal spans with zero saliency loss, and writes a new `model_best.ckpt` plus `result.json`. It never feeds the test split into training or checkpoint selection. No source annotation or frozen dataset file is modified.

The local development host has no CUDA device or upstream checkpoint. The Colab training run and any resulting metrics have not been executed or claimed here.
