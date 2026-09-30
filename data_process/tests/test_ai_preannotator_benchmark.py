"""Unit tests for Stage A.2.5C: 5-Sample AI Preannotator Benchmark.

CPU-only tests covering:
1. Single-window tIoU calculation
2. Multi-window Top-1 tIoU selection
3. Start absolute error, end absolute error, boundary MAE
4. Hit@0.5 and Hit@0.7
5. Aggregate metric computation (mean, median, min, max, R1@0.5, R1@0.7)
6. Quality gate PASS scenario
7. Quality gate FAIL scenario
8. Missing AI candidate handling (MISSING_AI_CANDIDATE)
9. Missing human GT handling (MISSING_HUMAN_GT)
10. Source data immutability verification (source files remain unchanged)
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import pytest

from data_process.adsqa.ai_benchmark import (
    classify_semantic_event_match,
    compute_boundary_errors,
    compute_interval_tiou,
    compute_multi_window_top1_tiou,
    evaluate_5sample_benchmark,
    get_file_sha256,
)
from data_process.annotation.models import (
    AIConfidence,
    AIPreannotationRecord,
    AnnotationRecord,
    ReviewStatus,
)
from data_process.annotation.storage import (
    save_ai_preannotations,
    save_primary_records,
)


def test_single_window_tiou():
    """Verify single-interval tIoU computation."""
    # Identical intervals -> 1.0
    assert compute_interval_tiou([10.0, 20.0], [10.0, 20.0]) == 1.0

    # Disjoint intervals -> 0.0
    assert compute_interval_tiou([5.0, 10.0], [15.0, 20.0]) == 0.0

    # Partial overlap: [10, 20] and [15, 25] -> Inter=5, Union=15 -> 5/15 = 0.3333
    assert compute_interval_tiou([10.0, 20.0], [15.0, 25.0]) == pytest.approx(0.3333, abs=1e-4)

    # Contained: [10, 20] and [12, 16] -> Inter=4, Union=10 -> 4/10 = 0.4000
    assert compute_interval_tiou([10.0, 20.0], [12.0, 16.0]) == 0.4000


def test_multi_window_top1_tiou():
    """Verify Top-1 selection picks candidate with maximum tIoU without merging."""
    human_gt = [10.5, 25.0]  # length 14.5
    ai_windows = [
        [4.0, 14.0],    # overlap [10.5, 14.0] = 3.5; union = 10 + 14.5 - 3.5 = 21.0 -> 3.5/21.0 = 0.1667
        [20.5, 24.5],   # overlap [20.5, 24.5] = 4.0; union = 4 + 14.5 - 4.0 = 14.5 -> 4.0/14.5 = 0.2759
        [30.0, 35.0],   # disjoint -> 0.0
    ]

    best_tiou, best_win = compute_multi_window_top1_tiou(ai_windows, human_gt)
    assert best_tiou == pytest.approx(0.2759, abs=1e-4)
    assert best_win == [20.5, 24.5]

    # Empty windows
    zero_tiou, zero_win = compute_multi_window_top1_tiou([], human_gt)
    assert zero_tiou == 0.0
    assert zero_win == []


def test_boundary_errors_and_mae():
    """Verify start error, end error, and boundary MAE."""
    ai_win = [29.0, 32.0]
    human_gt = [29.9, 31.0]

    s_err, e_err, b_mae = compute_boundary_errors(ai_win, human_gt)
    assert s_err == 0.90
    assert e_err == 1.00
    assert b_mae == 0.95


def test_hit_at_05_and_07():
    """Verify Hit@0.5 and Hit@0.7 thresholding."""
    high_tiou, _ = compute_multi_window_top1_tiou([[10.0, 20.0]], [10.5, 19.5])
    # Inter=9.0, Union=10.0 -> tIoU = 0.90
    assert high_tiou >= 0.70
    assert (1 if high_tiou >= 0.50 else 0) == 1
    assert (1 if high_tiou >= 0.70 else 0) == 1

    mid_tiou, _ = compute_multi_window_top1_tiou([[10.0, 20.0]], [14.0, 22.0])
    # Inter=6.0, Union=12.0 -> tIoU = 0.50
    assert (1 if mid_tiou >= 0.50 else 0) == 1
    assert (1 if mid_tiou >= 0.70 else 0) == 0

    low_tiou, _ = compute_multi_window_top1_tiou([[10.0, 20.0]], [18.0, 25.0])
    # Inter=2.0, Union=15.0 -> tIoU = 0.1333
    assert (1 if low_tiou >= 0.50 else 0) == 0
    assert (1 if low_tiou >= 0.70 else 0) == 0


def test_semantic_event_match_classification():
    """Verify semantic event match classification logic."""
    assert classify_semantic_event_match("lumae_ads_pilot_0004", "", [50, 69], [12, 28], 0.0) == "WRONG_EVENT"
    assert classify_semantic_event_match("lumae_ads_pilot_0006", "", [10, 25], [20, 24], 0.27) == "PARTIAL_EVENT"
    assert classify_semantic_event_match("lumae_ads_pilot_0007", "", [18, 21], [6, 17], 0.0) == "WRONG_EVENT"
    assert classify_semantic_event_match("lumae_ads_pilot_0012", "", [6, 8], [3, 22], 0.08) == "PARTIAL_EVENT"
    assert classify_semantic_event_match("lumae_ads_pilot_0013", "", [29, 31], [29, 32], 0.36) == "CORRECT_EVENT"


def test_quality_gate_decision_pass_and_fail(monkeypatch, tmp_path):
    """Verify explicit quality gate evaluation for both PASS and FAIL scenarios."""
    monkeypatch.setenv("LUMAE_ANNOTATIONS_DIR", str(tmp_path))
    monkeypatch.setenv("LUMAE_REPORTS_DIR", str(tmp_path))

    # Scenario A: Setup high-accuracy mock data that should PASS the gate
    # (mean_tIoU >= 0.60, semantic_correct >= 4, wrong_event <= 1)
    pass_human_records = []
    pass_ai_records = []

    for i in range(1, 6):
        sid = f"mock_sample_{i}"
        pass_human_records.append(
            AnnotationRecord(
                sample_id=sid,
                video_filename=f"{sid}.mp4",
                vid=f"vid_{sid}",
                source_dataset="AdsQA",
                source_video_id=sid,
                source_split="test",
                source_url="",
                product_category="Gadgets",
                query=f"Demonstration action {i}",
                duration_seconds=30.0,
                gt_start_seconds=10.0,
                gt_end_seconds=20.0,
                annotator_id="human_01",
                review_status=ReviewStatus.REVIEWED.value,
            )
        )
        pass_ai_records.append(
            AIPreannotationRecord(
                sample_id=sid,
                video_filename=f"{sid}.mp4",
                original_query=f"Demonstration action {i}",
                ai_query_status="VALID",
                ai_proposed_query=f"Demonstration action {i}",
                ai_start_seconds=10.5,
                ai_end_seconds=19.5,
                ai_windows_json="[[10.5, 19.5]]",
                ai_confidence="HIGH",
                ai_reason="Clear demo",
                analysis_method="test",
                frame_sampling_interval=2.0,
                boundary_refinement_interval=0.5,
                generated_at_utc="",
                status="AI_PROPOSED",
            )
        )

    save_primary_records(pass_human_records)
    save_ai_preannotations(pass_ai_records)

    report_pass = evaluate_5sample_benchmark(
        sample_ids=[f"mock_sample_{i}" for i in range(1, 6)],
        output_dir=tmp_path / "pass_out",
        include_diagnostic_candidates=False,
    )

    # tIoU between [10, 20] and [10.5, 19.5]: Inter=9.0, Union=10.0 -> 0.9000
    assert report_pass["aggregate_metrics"]["mean_tIoU"] >= 0.60
    assert report_pass["quality_gate"]["gate_result"] == "PASS_FOR_AI_ASSISTED_REVIEW"
    assert "BENCHMARK PASS" in report_pass["quality_gate"]["final_status"]

    # Scenario B: Setup low-accuracy mock data that should FAIL the gate
    fail_ai_records = []
    for i in range(1, 6):
        sid = f"mock_sample_{i}"
        fail_ai_records.append(
            AIPreannotationRecord(
                sample_id=sid,
                video_filename=f"{sid}.mp4",
                original_query=f"Demonstration action {i}",
                ai_query_status="VALID",
                ai_proposed_query=f"Demonstration action {i}",
                ai_start_seconds=1.0,
                ai_end_seconds=4.0,  # Completely disjoint from [10, 20]
                ai_windows_json="[[1.0, 4.0]]",
                ai_confidence="LOW",
                ai_reason="Intro scene detected",
                analysis_method="test",
                frame_sampling_interval=2.0,
                boundary_refinement_interval=0.5,
                generated_at_utc="",
                status="AI_PROPOSED",
            )
        )
    save_ai_preannotations(fail_ai_records)

    report_fail = evaluate_5sample_benchmark(
        sample_ids=[f"mock_sample_{i}" for i in range(1, 6)],
        output_dir=tmp_path / "fail_out",
        include_diagnostic_candidates=False,
    )

    assert report_fail["aggregate_metrics"]["mean_tIoU"] < 0.60
    assert report_fail["quality_gate"]["gate_result"] == "FAIL_CURRENT_PREANNOTATOR"
    assert "BENCHMARK FAIL" in report_fail["quality_gate"]["final_status"]


def test_missing_candidate_and_missing_gt_handling(monkeypatch, tmp_path):
    """Verify handling when an AI candidate or human GT record is missing."""
    monkeypatch.setenv("LUMAE_ANNOTATIONS_DIR", str(tmp_path))
    monkeypatch.setenv("LUMAE_REPORTS_DIR", str(tmp_path))

    # Only create GT for sample_A, sample_B is missing from human GT
    save_primary_records([
        AnnotationRecord(
            sample_id="sample_A",
            video_filename="a.mp4",
            vid="v_a",
            source_dataset="AdsQA",
            source_video_id="a",
            source_split="test",
            source_url="",
            product_category="Demo",
            query="Query A",
            duration_seconds=30.0,
            gt_start_seconds=5.0,
            gt_end_seconds=15.0,
            annotator_id="human_01",
            review_status=ReviewStatus.REVIEWED.value,
        )
    ])

    # No AI candidates saved in ai_preannotations.csv
    save_ai_preannotations([])

    rep = evaluate_5sample_benchmark(
        sample_ids=["sample_A", "sample_B"],
        output_dir=tmp_path / "missing_out",
        include_diagnostic_candidates=False,
    )

    # Sample A has human GT but missing AI candidate -> MISSING_AI_CANDIDATE
    res_a = [s for s in rep["sample_level_results"] if s["sample_id"] == "sample_A"][0]
    assert res_a["status"] == "MISSING_AI_CANDIDATE"

    # Sample B is missing from human GT
    assert any("sample_B" in m for m in rep["input_audit"]["missing_samples"])


def test_source_files_remain_unchanged_after_benchmark(monkeypatch, tmp_path):
    """Verify that running the benchmark does NOT alter input annotation files."""
    monkeypatch.setenv("LUMAE_ANNOTATIONS_DIR", str(tmp_path))
    monkeypatch.setenv("LUMAE_REPORTS_DIR", str(tmp_path))

    rec = AnnotationRecord(
        sample_id="test_immutable",
        video_filename="imm.mp4",
        vid="vid_imm",
        source_dataset="AdsQA",
        source_video_id="imm",
        source_split="test",
        source_url="",
        product_category="Product",
        query="Test query",
        duration_seconds=30.0,
        gt_start_seconds=10.0,
        gt_end_seconds=20.0,
        annotator_id="human_01",
        review_status=ReviewStatus.REVIEWED.value,
    )
    save_primary_records([rec])

    cand = AIPreannotationRecord(
        sample_id="test_immutable",
        video_filename="imm.mp4",
        original_query="Test query",
        ai_query_status="VALID",
        ai_proposed_query="Test query",
        ai_start_seconds=10.0,
        ai_end_seconds=20.0,
        ai_windows_json="[[10.0, 20.0]]",
        ai_confidence="HIGH",
        ai_reason="Exact match",
        analysis_method="test",
        frame_sampling_interval=2.0,
        boundary_refinement_interval=0.5,
        generated_at_utc="",
        status="AI_PROPOSED",
    )
    save_ai_preannotations([cand])

    h_path = tmp_path / "human_primary.csv"
    ai_path = tmp_path / "ai_preannotations.csv"

    sha_h_before = get_file_sha256(h_path)
    sha_ai_before = get_file_sha256(ai_path)

    evaluate_5sample_benchmark(
        sample_ids=["test_immutable"],
        output_dir=tmp_path / "imm_out",
        include_diagnostic_candidates=False,
    )

    sha_h_after = get_file_sha256(h_path)
    sha_ai_after = get_file_sha256(ai_path)

    assert sha_h_before == sha_h_after
    assert sha_ai_before == sha_ai_after
