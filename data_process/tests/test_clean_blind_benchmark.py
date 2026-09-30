"""Unit tests for STAGE A.2.5F Clean Blind Benchmark & Generalization Evaluation.

Verifies:
1. Human GT freeze manifest and SHA256 verification.
2. Prediction hash verification and mismatch rejection.
3. Human-first freeze ordering rule.
4. Query change classification and fair-comparison logic.
5. Single-window tIoU, multi-window top-1 selection, and boundary errors.
6. Small-sample count reporting (raw counts and fractions).
7. Quality gates: PASS, FAIL, INCONCLUSIVE_QUERY_SHIFT.
8. Source data immutability.
"""

import csv
import hashlib
import json
from pathlib import Path
import pytest

from data_process.adsqa.clean_blind_benchmark import (
    classify_query_alignment,
    compute_temporal_iou,
    evaluate_clean_blind_benchmark,
    sha256_file,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def test_human_gt_freeze_manifest_and_sha256():
    """Verify clean3_human_gt_frozen.csv and clean3_human_gt_freeze.json validity."""
    gt_csv = REPO_ROOT / "local_data" / "annotations" / "lumae_ads" / "blind_eval" / "clean3_human_gt_frozen.csv"
    manifest_path = REPO_ROOT / "local_data" / "manifests" / "clean3_human_gt_freeze.json"

    assert gt_csv.is_file(), f"Missing human GT csv: {gt_csv}"
    assert manifest_path.is_file(), f"Missing human GT manifest: {manifest_path}"

    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest["status"] == "HUMAN_GT_FROZEN_BEFORE_AI_REVEAL"
    assert manifest["sample_ids"] == [
        "lumae_ads_pilot_0008",
        "lumae_ads_pilot_0010",
        "lumae_ads_pilot_0011",
    ]

    actual_sha = sha256_file(gt_csv)
    assert actual_sha == manifest["human_gt_sha256"], "Human GT SHA256 mismatch!"


def test_prediction_hash_verification():
    """Verify sealed predictions match the frozen SHA256 in semantic_v2_clean3_blind_freeze.json."""
    pred_csv = REPO_ROOT / "local_data" / "annotations" / "lumae_ads" / "blind_eval" / "semantic_v2_clean3_predictions.csv"
    manifest_path = REPO_ROOT / "local_data" / "manifests" / "semantic_v2_clean3_blind_freeze.json"

    assert pred_csv.is_file()
    assert manifest_path.is_file()

    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest["status"] == "PREDICTIONS_FROZEN_BEFORE_HUMAN_GT"
    actual_sha = sha256_file(pred_csv)
    assert actual_sha == manifest["prediction_sha256"]


def test_prediction_hash_mismatch_rejection(tmp_path: Path):
    """Verify evaluate_clean_blind_benchmark rejects tampered predictions."""
    fake_root = tmp_path / "lumae_test"
    fake_root.mkdir(parents=True, exist_ok=True)

    # Copy genuine human gt files
    gt_dir = fake_root / "local_data" / "annotations" / "lumae_ads" / "blind_eval"
    gt_dir.mkdir(parents=True, exist_ok=True)
    real_gt_csv = REPO_ROOT / "local_data" / "annotations" / "lumae_ads" / "blind_eval" / "clean3_human_gt_frozen.csv"
    gt_csv = gt_dir / "clean3_human_gt_frozen.csv"
    gt_csv.write_bytes(real_gt_csv.read_bytes())

    m_dir = fake_root / "local_data" / "manifests"
    m_dir.mkdir(parents=True, exist_ok=True)
    real_h_manifest = REPO_ROOT / "local_data" / "manifests" / "clean3_human_gt_freeze.json"
    (m_dir / "clean3_human_gt_freeze.json").write_bytes(real_h_manifest.read_bytes())

    # Create tampered predictions CSV
    pred_csv = gt_dir / "semantic_v2_clean3_predictions.csv"
    pred_csv.write_text("sample_id,tampered\nlumae_ads_pilot_0008,true\n", encoding="utf-8")

    # Genuine manifest expecting different hash
    real_p_manifest = REPO_ROOT / "local_data" / "manifests" / "semantic_v2_clean3_blind_freeze.json"
    (m_dir / "semantic_v2_clean3_blind_freeze.json").write_bytes(real_p_manifest.read_bytes())

    with pytest.raises(RuntimeError, match="BLOCKED — SEALED AI PREDICTION HASH MISMATCH"):
        evaluate_clean_blind_benchmark(repo_root=fake_root)


def test_human_first_freeze_ordering(tmp_path: Path):
    """Verify evaluate_clean_blind_benchmark blocks if human GT is not frozen first."""
    fake_root = tmp_path / "lumae_test_order"
    fake_root.mkdir(parents=True, exist_ok=True)

    with pytest.raises(RuntimeError, match="BLOCKED — HUMAN BLIND GT INCOMPLETE"):
        evaluate_clean_blind_benchmark(repo_root=fake_root)


def test_query_change_classification():
    """Verify classification of query semantic compatibility."""
    # Identical query
    c_type, comp = classify_query_alignment(
        "The demonstrator shows the product.",
        "The demonstrator shows the product."
    )
    assert c_type == "SAME_INTENT"
    assert comp is True

    # Generic narrowed to specific feature
    c_type, comp = classify_query_alignment(
        "The creator presents the key physical features of the product.",
        "The sports glasses' display, projector, and lens features are shown."
    )
    assert c_type == "NARROWED_INTENT"
    assert comp is True


def test_tiou_single_and_multi_window():
    """Verify tIoU computation for exact overlap, partial overlap, and disjoint intervals."""
    # Exact overlap
    assert compute_temporal_iou([10.0, 20.0], [10.0, 20.0]) == 1.0

    # Disjoint
    assert compute_temporal_iou([10.0, 15.0], [20.0, 25.0]) == 0.0

    # Partial overlap: [22.1, 24.1] vs [22.5, 24.4]
    # inter = 1.6, union = 2.3 -> 1.6 / 2.3 = 0.695652...
    tiou = compute_temporal_iou([22.1, 24.1], [22.5, 24.4])
    assert round(tiou, 4) == 0.6957

    # Multi-window selection: picks highest tIoU
    gt = [34.0, 44.5]
    windows = [[31.0, 44.8], [31.0, 52.5]]
    tious = [compute_temporal_iou(w, gt) for w in windows]
    best_tiou = max(tious)
    assert round(best_tiou, 4) == 0.7609


def test_clean_blind_benchmark_execution_and_metrics():
    """Verify live execution of evaluate_clean_blind_benchmark against frozen artifacts."""
    report = evaluate_clean_blind_benchmark()

    # Pre-condition verifications
    assert report["verification"]["prediction_hash_match"] is True
    assert report["verification"]["human_freeze_status"] == "HUMAN_GT_FROZEN_BEFORE_AI_REVEAL"

    # Per sample counts
    assert len(report["per_sample"]) == 3
    s0008 = next(r for r in report["per_sample"] if r["sample_id"] == "lumae_ads_pilot_0008")
    s0010 = next(r for r in report["per_sample"] if r["sample_id"] == "lumae_ads_pilot_0010")
    s0011 = next(r for r in report["per_sample"] if r["sample_id"] == "lumae_ads_pilot_0011")

    assert s0008["tiou"] == 0.6957
    assert s0008["semantic_event_match"] == "CORRECT_EVENT"

    assert s0010["tiou"] == 0.0
    assert s0010["semantic_event_match"] == "WRONG_EVENT"

    assert s0011["tiou"] == 0.7609
    assert s0011["semantic_event_match"] == "CORRECT_EVENT"

    # Aggregate
    agg = report["aggregate"]
    assert agg["comparable_sample_count"] == 3
    assert agg["mean_tiou"] == 0.4855
    assert agg["median_tiou"] == 0.6957
    assert agg["r1_0_5_fraction"] == "2 / 3"
    assert agg["r1_0_7_fraction"] == "1 / 3"
    assert agg["semantic_correct_count"] == 2
    assert agg["semantic_wrong_count"] == 1

    # Quality Gate
    # mean_tiou = 0.4855 < 0.60 -> FAIL_CLEAN3_GENERALIZATION
    assert report["quality_gate"]["gate_result"] == "FAIL_CLEAN3_GENERALIZATION"
    assert report["dev_vs_blind"]["generalization_gap"] == 0.5145


def test_source_data_unchanged():
    """Verify source data files have not been mutated by benchmark."""
    pred_path = REPO_ROOT / "local_data" / "annotations" / "lumae_ads" / "blind_eval" / "semantic_v2_clean3_predictions.csv"
    assert sha256_file(pred_path) == "bfec7d5c91e699ffb52315bcb1a97e9e03b4473304f5644f92203d19190bf8b2"

    sem_v2_path = REPO_ROOT / "local_data" / "annotations" / "lumae_ads" / "semantic_preannotations_v2.csv"
    assert sha256_file(sem_v2_path) == "e3ead17dd58c988c700e80e0800caaa4996bd081112946ed41ad51a03eebf8a1"

    ai_path = REPO_ROOT / "local_data" / "annotations" / "lumae_ads" / "ai_preannotations.csv"
    assert sha256_file(ai_path) == "651ef4661a23deb342c44fe166da1a8612714debeff8a33ec0ab426f8ee0a94f"

    holdout_path = REPO_ROOT / "local_data" / "annotations" / "lumae_ads" / "blind_holdout.json"
    assert sha256_file(holdout_path) == "ec551df79eac67b5984190d6efab730c173e3c394cbeea1761632ab247a52387"
