# M0 Baseline: Pretrained Moment-DETR (CLIP-only)

## Description
This experiment establishes the zero-shot / baseline performance of the official pretrained Moment-DETR model (`model_best.ckpt`) on:
1. The standard **QVHighlights validation split** (reproducibility reference).
2. The held-out **Lumae Product-Ads test split** (out-of-domain baseline).

## Hardware & Environment
- Evaluated on: Google Colab T4 GPU (or local CUDA).
- Checkpoint: External persistent storage in Google Drive (`Checkpoints/M0/model_best.ckpt`).
- Features: Pre-extracted CLIP ViT-B/32 visual embeddings (~2s intervals) and text embeddings.

## Reference Benchmark Numbers (QVHighlights Val)
- R1@0.5: `53.23`
- R1@0.7: `34.00`
- MR mAP: `30.58`
- HL mAP: `35.51`
- HIT@1: `55.87`
