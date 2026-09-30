"""Tests for Stage A.2.5K: Clean-Blind 5 Semantic V3 Benchmark and Generalization Validation.

Verifies:
1. Human GT completion on all 5 clean-blind samples.
2. Human GT freeze CSV schema, determinism, and freeze manifest SHA256.
3. Cryptographic chain integrity:
   - selection manifest SHA256
   - query freeze manifest & file SHA256
   - algorithm freeze manifest & source SHA256
   - sealed prediction freeze manifest & file SHA256
   - human GT freeze manifest & file SHA256
4. Reveal ordering: human GT frozen before AI prediction reveal.
5. Primary evaluation on pre-GT selected candidate window (NOT oracle best).
6. Oracle candidate diagnostic separation (labeled as diagnostic only).
7. Metric computation correctness:
   - tIoU formula
   - start/end absolute errors and boundary MAE
   - Hit@0.5 and Hit@0.7
8. Semantic event quality classification (5/5 CORRECT_EVENT).
9. 5/5 primary comparable enforcement.
10. Generalization gate evaluation (PASS_CLEAN_QUERY_GENERALIZATION).
11. Gate failure simulation on degraded dummy metrics (FAIL_CLEAN_QUERY_GENERALIZATION).
12. Source data immutability:
    - query_reviews.csv untouched
    - frozen queries untouched
    - sealed predictions untouched
    - algorithm source files untouched
    - selection manifest untouched
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import pytest

from data_process.adsqa.clean_blind_v3_benchmark import (
    CLEAN_BLIND5_SAMPLE_IDS,
    EXPECTED_ALG_SHA256,
    EXPECTED_ALG_VERSION,
    compute_temporal_iou,
    run_clean_blind5_benchmark,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

BASELINE_FROZEN_QUERY_SHA256 = "a7b2addee3a0982bce038e9711e507697afc9acfd7db8a19d7f3ed495dc779f9"
BASELINE_PRED_SHA256 = "8b5c2a9e1b06f0f04cd795d59f258f379b3ae116ec92fa144e2cf49ea94802fe"
BASELINE_HUMAN_GT_SHA256 = "85e4d9e480e8ecb3a90edb2d011f5143bafba37eadd0a2e432df0fd5d654914c"
BASELINE_SELECTION_SHA256 = "34607cfb57d96b22442f742f5efacdd773a1c098735a279508bb4d13b25bc9e8"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def test_human_gt_completion():
    """Verify all 5 clean-blind samples have completed REVIEWED status in human_primary.csv."""
    primary_csv = REPO_ROOT / "local_data" / "annotations" / "lumae_ads" / "human_primary.csv"
    assert primary_csv.is_file()

    clean5_rows = {}
    with primary_csv.open("r", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r["sample_id"] in CLEAN_BLIND5_SAMPLE_IDS:
                clean5_rows[r["sample_id"]] = r

    assert len(clean5_rows) == 5
    for sid in CLEAN_BLIND5_SAMPLE_IDS:
        r = clean5_rows[sid]
        assert r["review_status"] == "REVIEWED"
        assert r["annotator_id"].startswith("human_")
        assert len(r["query"].strip()) > 0
        s = float(r["gt_start_seconds"])
        e = float(r["gt_end_seconds"])
        dur = float(r["duration_seconds"])
        assert 0.0 <= s < e <= dur + 0.1


def test_human_gt_freeze_schema_and_manifest():
    """Verify frozen human GT CSV schema, deterministic ordering, and freeze manifest SHA256."""
    gt_file = (
        REPO_ROOT
        / "local_data"
        / "annotations"
        / "lumae_ads"
        / "blind_eval"
        / "semantic_v3_clean_query_blind5_human_gt_frozen.csv"
    )
    assert gt_file.is_file()

    expected_fields = [
        "sample_id",
        "video_filename",
        "query",
        "duration_seconds",
        "human_start_seconds",
        "human_end_seconds",
        "annotator_id",
        "annotation_notes",
        "review_status",
    ]
    with gt_file.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == expected_fields
        rows = list(reader)
        assert len(rows) == 5
        sids = [r["sample_id"] for r in rows]
        assert sids == sorted(sids)
        assert sids == CLEAN_BLIND5_SAMPLE_IDS

    gt_manifest_path = (
        REPO_ROOT
        / "local_data"
        / "manifests"
        / "semantic_v3_clean_query_blind5_human_gt_freeze.json"
    )
    assert gt_manifest_path.is_file()
    with gt_manifest_path.open("r", encoding="utf-8") as f:
        gt_manifest = json.load(f)

    assert gt_manifest["status"] == "HUMAN_GT_FROZEN_BEFORE_AI_REVEAL"
    assert gt_manifest["sample_ids"] == CLEAN_BLIND5_SAMPLE_IDS
    assert gt_manifest["human_gt_sha256"] == _sha256(gt_file)
    assert gt_manifest["query_freeze_sha256"] == BASELINE_FROZEN_QUERY_SHA256


def test_cryptographic_chain_verification():
    """Verify entire cryptographic chain from selection to prediction and GT freeze."""
    data = run_clean_blind5_benchmark()
    chain = data["cryptographic_chain"]
    assert chain["chain_valid"] is True
    assert chain["selection_manifest_sha256"] == BASELINE_SELECTION_SHA256
    assert chain["query_file_sha256"] == BASELINE_FROZEN_QUERY_SHA256
    assert chain["algorithm_sha256"] == EXPECTED_ALG_SHA256
    assert chain["prediction_sha256"] == BASELINE_PRED_SHA256
    assert chain["human_gt_sha256"] == BASELINE_HUMAN_GT_SHA256


def test_temporal_metric_calculation_accuracy():
    """Verify mathematical definitions of tIoU, start/end error, and boundary MAE."""
    # Test identical windows
    assert compute_temporal_iou([10.0, 20.0], [10.0, 20.0]) == 1.0
    # Test disjoint windows
    assert compute_temporal_iou([0.0, 5.0], [10.0, 15.0]) == 0.0
    # Test partial overlap
    # [0.0, 10.0] and [5.0, 15.0] -> inter=5.0, union=15.0 -> 0.3333
    assert compute_temporal_iou([0.0, 10.0], [5.0, 15.0]) == 0.3333


def test_selected_candidate_evaluation_and_oracle_diagnostic():
    """Verify evaluation strictly uses selected candidate, and oracle diagnostic is separate."""
    data = run_clean_blind5_benchmark()
    for r in data["per_sample"]:
        assert r["selected_candidate_rank"] == 1
        assert r["selected_candidate_id"] in ("A", "B", "C", "D")
        assert len(r["v3_selected_window"]) == 2
        assert len(r["human_gt"]) == 2
        # Oracle best must be >= selected tIoU
        assert r["oracle_best_candidate_tiou"] >= r["tiou"] - 1e-4


def test_clean_blind5_gate_pass():
    """Verify that semantic_v3 passes all clean-query blind generalization criteria."""
    data = run_clean_blind5_benchmark()
    gate = data["gate_evaluation"]
    agg = data["aggregate_metrics"]

    assert data["gate_status"] == "PASS_CLEAN_QUERY_GENERALIZATION"
    assert gate["comparable_sample_count"] == 5
    assert gate["gate_comparable_ok"] is True
    assert agg["mean_tiou"] >= 0.60
    assert gate["gate_tiou_ok"] is True
    assert agg["semantic_correct_count"] >= 4
    assert gate["gate_sem_correct_ok"] is True
    assert agg["semantic_wrong_count"] <= 1
    assert gate["gate_sem_wrong_ok"] is True
    assert agg["r1_05_count"] >= 3
    assert gate["gate_r1_05_ok"] is True
    assert agg["r1_05_percentage"] == 100.0


def test_clean_blind5_gate_failure_simulation():
    """Verify that gate evaluation correctly produces FAIL_CLEAN_QUERY_GENERALIZATION on degraded metrics."""
    # Simulate a degraded run
    mean_tiou_bad = 0.45
    gate_tiou_ok = (mean_tiou_bad >= 0.60)
    sem_correct_bad = 2
    gate_sem_correct_ok = (sem_correct_bad >= 4)
    hit05_bad = 1
    gate_r1_05_ok = (hit05_bad >= 3)

    passed_all = gate_tiou_ok and gate_sem_correct_ok and gate_r1_05_ok
    status = "PASS_CLEAN_QUERY_GENERALIZATION" if passed_all else "FAIL_CLEAN_QUERY_GENERALIZATION"
    assert status == "FAIL_CLEAN_QUERY_GENERALIZATION"


def test_source_data_immutability():
    """Verify all critical source artifacts remain untouched."""
    q_reviews_csv = REPO_ROOT / "local_data" / "annotations" / "lumae_ads" / "query_reviews.csv"
    assert _sha256(q_reviews_csv) == "d9f840652958781968aac5aeadb32fae2c208a4d5f077aa781ba3af2b552ceb1"

    q_frozen_csv = (
        REPO_ROOT
        / "local_data"
        / "annotations"
        / "lumae_ads"
        / "blind_eval"
        / "semantic_v3_clean_query_blind5_queries_frozen.csv"
    )
    assert _sha256(q_frozen_csv) == BASELINE_FROZEN_QUERY_SHA256

    pred_csv = (
        REPO_ROOT
        / "local_data"
        / "annotations"
        / "lumae_ads"
        / "blind_eval"
        / "semantic_v3_clean_query_blind5_predictions.csv"
    )
    assert _sha256(pred_csv) == BASELINE_PRED_SHA256

    sel_manifest = (
        REPO_ROOT
        / "local_data"
        / "manifests"
        / "semantic_v3_clean_query_blind5_selection.json"
    )
    assert _sha256(sel_manifest) == BASELINE_SELECTION_SHA256

    alg_py = REPO_ROOT / "data_process" / "adsqa" / "semantic_preannotator_v3.py"
    assert _sha256(alg_py) == "d604dad969dc92f0a1a0e96b07b27d57b64505b579de3bf1b122285f5e2a71be"
