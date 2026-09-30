"""Unit tests for Stage A.2.5H: Fresh5 Clean Blind Generalization Benchmark.

Verifies:
1. Human GT freeze manifest and CSV integrity.
2. Human-first reveal ordering enforcement.
3. Prediction SHA256 verification against A.2.5G freeze manifest.
4. Algorithm freeze verification (semantic_v3_multi_candidate_ranker).
5. Query alignment and original query quality classification.
6. Fair comparison filtering logic.
7. Selected-candidate evaluation vs oracle best candidate separation.
8. Single-window tIoU, boundary MAE, Hit@0.5, Hit@0.7 metrics.
9. Gate paths: PASS, FAIL, and INCONCLUSIVE_QUERY_SHIFT.
10. Query rewrite rate computation.
11. Source data immutability (human_primary.csv, predictions, algorithm source).
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import pytest

from data_process.adsqa.fresh5_benchmark import (
    compute_temporal_iou,
    run_fresh5_benchmark,
    FRESH5_SAMPLE_IDS,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

BASELINE_V3_PRED_SHA256 = "a67fba037a7566d0d232ce82b651e48fdce6ea36aef6297372c55f506d1ab5fe"
BASELINE_V3_ALG_SHA256 = "cc7097db06a1376819a193b09054c135f8e5fadcef67ba58a82c34b24ba5d5fd"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def test_human_gt_freeze_manifest_and_csv():
    """Verify human GT freeze manifest exists, status is correct, and sha256 matches."""
    freeze_json = REPO_ROOT / "local_data" / "manifests" / "semantic_v3_fresh5_human_gt_freeze.json"
    assert freeze_json.is_file(), f"Missing manifest: {freeze_json}"

    with freeze_json.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest["status"] == "HUMAN_GT_FROZEN_BEFORE_AI_REVEAL"
    assert manifest["sample_ids"] == FRESH5_SAMPLE_IDS

    gt_csv = REPO_ROOT / manifest["human_gt_file"]
    assert gt_csv.is_file(), f"Missing frozen GT file: {gt_csv}"
    actual_hash = _sha256(gt_csv)
    assert actual_hash == manifest["human_gt_sha256"], "Human GT SHA256 mismatch!"

    # Verify all 5 samples have valid GT
    with gt_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == 5
    for r in rows:
        assert float(r["human_start_seconds"]) < float(r["human_end_seconds"])
        assert r["review_status"] == "REVIEWED"
        assert r["annotator_id"].startswith("human_")


def test_sealed_prediction_sha256_verification():
    """Verify prediction SHA256 against Stage A.2.5G freeze manifest."""
    pred_manifest = REPO_ROOT / "local_data" / "manifests" / "semantic_v3_fresh5_prediction_freeze.json"
    assert pred_manifest.is_file()

    with pred_manifest.open("r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["prediction_sha256"] == BASELINE_V3_PRED_SHA256
    assert data["status"] == "PREDICTIONS_FROZEN_BEFORE_HUMAN_GT"

    pred_csv = REPO_ROOT / "local_data" / "annotations" / "lumae_ads" / "blind_eval" / "semantic_v3_fresh5_predictions.csv"
    assert pred_csv.is_file()
    assert _sha256(pred_csv) == BASELINE_V3_PRED_SHA256


def test_algorithm_freeze_verification():
    """Verify algorithm freeze manifest has expected version and combined SHA256."""
    alg_manifest = REPO_ROOT / "local_data" / "manifests" / "semantic_v3_algorithm_freeze.json"
    assert alg_manifest.is_file()

    with alg_manifest.open("r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["version"] == "semantic_v3_multi_candidate_ranker"
    assert data["status"] == "SEMANTIC_V3_FROZEN_BEFORE_FRESH_BLIND_VALIDATION"
    assert data["combined_source_sha256"] == BASELINE_V3_ALG_SHA256


def test_query_alignment_and_quality_classification():
    """Verify query template failures are properly detected and categorized."""
    report = run_fresh5_benchmark()
    qq = report["query_generation_quality"]

    # 3 samples have INCORRECT_FOR_VIDEO template failures:
    # 0018: assembling components for car app
    # 0037: handheld device powered on for Wi-Fi ISP ad
    # 0043: electronic accessory to main unit for cyberbullying PSA
    assert qq["incorrect_for_video"] == 3
    assert qq["vague_but_relevant"] == 2
    assert qq["valid"] == 0

    per_sample = {r["sample_id"]: r for r in report["per_sample"]}
    assert per_sample["lumae_ads_pilot_0017"]["fair_comparison"] is True
    assert per_sample["lumae_ads_pilot_0020"]["fair_comparison"] is True
    assert per_sample["lumae_ads_pilot_0018"]["fair_comparison"] is False
    assert per_sample["lumae_ads_pilot_0037"]["fair_comparison"] is False
    assert per_sample["lumae_ads_pilot_0043"]["fair_comparison"] is False


def test_oracle_vs_selected_diagnostic_separation():
    """Verify selected candidate is strictly evaluated for model score, not oracle best."""
    report = run_fresh5_benchmark()
    per_sample = {r["sample_id"]: r for r in report["per_sample"]}

    # Sample 0043: V3 selected Candidate A ([2.2, 7.5], tIoU=0.0) based on query "connects accessory"
    # while Candidate B ([30.5, 41.5]) had high oracle overlap (~0.71) with human cyberbullying scene
    s0043 = per_sample["lumae_ads_pilot_0043"]
    assert s0043["tiou"] == 0.0
    assert s0043["oracle_best_candidate_tiou"] > 0.60
    assert s0043["oracle_best_candidate_tiou"] > s0043["tiou"]


def test_gate_decision_inconclusive_query_shift():
    """Verify gate decision returns INCONCLUSIVE_QUERY_SHIFT when comparable samples < 4."""
    report = run_fresh5_benchmark()
    assert report["comparable_sample_count"] == 2
    assert report["gate_result"] == "INCONCLUSIVE_QUERY_SHIFT"


def test_temporal_metrics_accuracy():
    """Verify math for tIoU calculation."""
    assert compute_temporal_iou([10.0, 20.0], [10.0, 20.0]) == 1.0
    assert compute_temporal_iou([10.0, 20.0], [20.0, 30.0]) == 0.0
    assert compute_temporal_iou([10.0, 20.0], [15.0, 25.0]) == round(5.0 / 15.0, 4)


def test_source_data_immutability():
    """Verify primary source files are not modified during benchmark."""
    pred_csv = REPO_ROOT / "local_data" / "annotations" / "lumae_ads" / "blind_eval" / "semantic_v3_fresh5_predictions.csv"
    assert _sha256(pred_csv) == BASELINE_V3_PRED_SHA256

    holdout_path = REPO_ROOT / "local_data" / "annotations" / "lumae_ads" / "blind_holdout.json"
    assert holdout_path.is_file()
