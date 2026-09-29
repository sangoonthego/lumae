"""Tests for temporal grounding metrics and N/A highlight handling."""

from data_process.metrics.temporal import (
    compute_sample_metrics,
    compute_temporal_iou,
    evaluate_grounding_dataset,
)


def test_temporal_iou_exact_match():
    assert compute_temporal_iou([5.0, 10.0], [5.0, 10.0]) == 1.0


def test_temporal_iou_disjoint():
    assert compute_temporal_iou([1.0, 4.0], [6.0, 10.0]) == 0.0


def test_temporal_iou_partial_overlap():
    # Intersection: [5, 8] = 3. Union: [2, 10] = 8. IoU = 3/8 = 0.375
    iou = compute_temporal_iou([2.0, 8.0], [5.0, 10.0])
    assert iou == 0.375


def test_sample_metrics():
    gt_windows = [[4.0, 10.0]]
    pred = [5.0, 11.0]  # Inter: 5 to 10 = 5. Union: 4 to 11 = 7. IoU: 5/7 ~ 0.7143
    res = compute_sample_metrics(pred, gt_windows)
    assert res["top1_tiou"] == 0.7143
    assert res["hit_0_5"] == 1
    assert res["hit_0_7"] == 1


def test_dataset_metrics_lumae_ads_unsupported_highlights_as_na():
    # Synthetic samples for Lumae Ads (no saliency GT)
    sample_results = [
        {"top1_tiou": 0.85, "hit_0_5": 1, "hit_0_7": 1},
        {"top1_tiou": 0.60, "hit_0_5": 1, "hit_0_7": 0},
        {"top1_tiou": 0.30, "hit_0_5": 0, "hit_0_7": 0},
    ]

    report = evaluate_grounding_dataset(sample_results, has_saliency_gt=False)

    assert report["sample_count"] == 3
    assert report["R1_0_5"] == 66.67
    assert report["R1_0_7"] == 33.33
    assert report["mIoU"] == 58.33
    # HL_mAP and HIT_1 MUST be "N/A" (never 0.0)
    assert report["HL_mAP"] == "N/A"
    assert report["HIT_1"] == "N/A"
