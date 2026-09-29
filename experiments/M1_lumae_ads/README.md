# M1 Domain Adaptation: Lumae Product-Ads Fine-Tuning

## Description
This experiment fine-tunes the pretrained Moment-DETR model (M0) on the **Lumae Product-Ads Dataset**. The objective is to improve query-conditioned temporal localization on short-form commercial advertising videos (e.g. demonstrating products, showing benefits, problem hooks) without catatrophically degrading generic temporal localization.

## Execution Target
- Platform: Google Colab T4 GPU.
- Parent weights: Initialized from M0 checkpoint (`Checkpoints/M0/model_best.ckpt`).
- Output checkpoint: Target Google Drive storage (`Checkpoints/M1/model_best.ckpt`).

## Evaluation Strategy
1. **Primary Domain Target**: Test performance on `local_data/canonical/lumae_ads/test.jsonl` (R1@0.5, R1@0.7, MR mAP).
2. **Generalization Retention**: Secondary evaluation on `local_data/canonical/qvhighlights/val.jsonl` to quantify out-of-domain degradation / retention.

## Status
- Initial status: **NOT_TRAINED**. Checkpoints and training scripts will be deployed in subsequent GPU execution phases.
