"""Unit tests for Stage A.2.5 Human Review Tooling, Validation, Storage, and Freezing."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from unittest.mock import patch
import pytest

from data_process.adsqa.annotation_qc import compute_temporal_iou
from data_process.annotation.models import (
    AnnotationMode,
    AnnotationRecord,
    QueryStatus,
    ReviewLogEntry,
    ReviewStatus,
    TemporalWindow,
)
from data_process.annotation.storage import (
    append_review_audit_log,
    atomic_write_text,
    generate_secondary_worklist,
    load_multi_windows,
    load_primary_records,
    save_multi_window,
)
from data_process.annotation.validation import (
    validate_multi_windows,
    validate_query,
    validate_temporal_boundaries,
)
from data_process.cli.freeze_lumae_ads_pilot import freeze_lumae_ads_pilot


def test_unreviewed_gt_fields_remain_blank():
    records = load_primary_records()
    unreviewed = [r for r in records if not r.is_human_reviewed() and not r.is_excluded()]
    assert len(unreviewed) > 0
    for r in unreviewed:
        assert r.gt_start_seconds is None
        assert r.gt_end_seconds is None


def test_existing_human_reviewed_rows_preserved():
    records = load_primary_records()
    by_id = {r.sample_id: r for r in records}

    # Verify lumae_ads_pilot_0004
    r4 = by_id.get("lumae_ads_pilot_0004")
    assert r4 is not None
    assert r4.is_human_reviewed()
    assert r4.annotator_id == "human_01"
    assert r4.gt_start_seconds == 50.0
    assert r4.gt_end_seconds == 69.0
    assert r4.query == "The demonstrator shows how the product functions."

    # Verify lumae_ads_pilot_0006
    r6 = by_id.get("lumae_ads_pilot_0006")
    assert r6 is not None
    assert r6.is_human_reviewed()
    assert r6.annotator_id == "human_01"
    assert r6.gt_start_seconds == 10.5
    assert r6.gt_end_seconds == 25.0
    assert r6.query == "People carry various Samsung products out of the store."


def test_agent_draft_annotations_never_used_for_prefilling():
    records = load_primary_records()
    # Check that sample 1 has blank GT even though draft pilot_annotations.csv has draft timestamps
    r1 = next(r for r in records if r.sample_id == "lumae_ads_pilot_0001")
    assert r1.gt_start_seconds is None
    assert r1.gt_end_seconds is None


def test_valid_temporal_boundaries():
    ok, err = validate_temporal_boundaries(5.0, 15.0, 30.0)
    assert ok is True
    assert err is None


def test_invalid_temporal_boundary_rejection():
    # start >= end
    ok, err = validate_temporal_boundaries(10.0, 5.0, 30.0)
    assert ok is False
    assert "precede" in str(err)

    # start < 0
    ok2, err2 = validate_temporal_boundaries(-1.0, 5.0, 30.0)
    assert ok2 is False
    assert "negative" in str(err2)

    # window too small (<0.1s)
    ok3, err3 = validate_temporal_boundaries(5.0, 5.05, 30.0)
    assert ok3 is False
    assert "0.1s" in str(err3)


def test_end_exceeds_duration_rejection():
    ok, err = validate_temporal_boundaries(10.0, 35.5, 30.0)
    assert ok is False
    assert "exceeds" in str(err).lower()


def test_query_validation():
    # Valid
    ok, err = validate_query("The presenter tests the smartphone camera in sunlight.", "VALID")
    assert ok is True
    assert err is None

    # Empty
    ok_empty, err_empty = validate_query("   ", "VALID")
    assert ok_empty is False
    assert "empty" in str(err_empty).lower()

    # Disallowed subjective phrase
    ok_subj, err_subj = validate_query("Watch the best part of the ad right here.", "VALID")
    assert ok_subj is False
    assert "disallowed" in str(err_subj).lower()


def test_multi_window_validation():
    # Valid non-overlapping
    windows = [[2.0, 8.0], [12.0, 18.0]]
    ok, err = validate_multi_windows(windows, 30.0)
    assert ok is True
    assert err is None

    # Overlapping windows
    windows_overlap = [[2.0, 10.0], [8.0, 15.0]]
    ok_ov, err_ov = validate_multi_windows(windows_overlap, 30.0)
    assert ok_ov is False
    assert "overlap" in str(err_ov).lower()


def test_atomic_write_and_backup(tmp_path: Path):
    target_file = tmp_path / "test.csv"
    target_file.write_text("initial content", encoding="utf-8")

    atomic_write_text(target_file, "updated content", make_backup=True)

    assert target_file.read_text(encoding="utf-8") == "updated content"
    backups = list((tmp_path / "backups").glob("test_*.csv"))
    assert len(backups) == 1
    assert backups[0].read_text(encoding="utf-8") == "initial content"


def test_audit_log_append(tmp_path: Path):
    log_file = tmp_path / "test_log.jsonl"
    entry = ReviewLogEntry(
        sample_id="test_001",
        video_filename="v1.mp4",
        original_query="Original query text",
        final_query="Final revised query text",
        query_status="NEEDS_EDIT",
        windows=[[2.5, 8.0]],
        annotator_id="human_01",
        notes="Corrected product name",
        reviewed_at_utc="2026-09-30T00:00:00Z",
        previous_review_status="DRAFT",
    )

    with patch("data_process.annotation.storage.get_annotations_dir", return_value=tmp_path):
        append_review_audit_log(entry)

    log_path = tmp_path / "human_review_log.jsonl"
    assert log_path.is_file()
    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert parsed["sample_id"] == "test_001"
    assert parsed["query_status"] == "NEEDS_EDIT"
    assert parsed["windows"] == [[2.5, 8.0]]


def test_multi_window_jsonl_storage(tmp_path: Path):
    with patch("data_process.annotation.storage.get_annotations_dir", return_value=tmp_path):
        save_multi_window("sample_mw_1", [[1.0, 5.0], [10.0, 15.0]])
        mapping = load_multi_windows()
        assert "sample_mw_1" in mapping
        assert mapping["sample_mw_1"] == [[1.0, 5.0], [10.0, 15.0]]


def test_secondary_worklist_contains_blank_gt(tmp_path: Path):
    records = [
        AnnotationRecord(
            sample_id="s1",
            video_filename="v1.mp4",
            vid="vid1",
            source_dataset="AdsQA",
            source_video_id="h1",
            source_split="train",
            source_url="http://example.com/1",
            product_category="Gadgets",
            query="Verified final query 1",
            duration_seconds=20.0,
            gt_start_seconds=4.0,
            gt_end_seconds=12.0,
            annotator_id="human_01",
            annotation_notes="Primary notes revealing boundary 4-12s",
            review_status="REVIEWED",
        ),
        AnnotationRecord(
            sample_id="s2",
            video_filename="v2.mp4",
            vid="vid2",
            source_dataset="AdsQA",
            source_video_id="h2",
            source_split="test",
            source_url="http://example.com/2",
            product_category="Appliances",
            query="Verified final query 2",
            duration_seconds=30.0,
            gt_start_seconds=5.0,
            gt_end_seconds=15.0,
            annotator_id="human_01",
            annotation_notes="Primary notes",
            review_status="REVIEWED",
        ),
    ]

    with patch("data_process.annotation.storage.get_annotations_dir", return_value=tmp_path):
        rows = generate_secondary_worklist(records, fraction=1.0, seed=42)
        assert len(rows) == 2
        for r in rows:
            # Must contain verified final query
            assert "Verified final query" in r["query"]
            # Must have BLANK GT
            assert r["gt_start_seconds"] == ""
            assert r["gt_end_seconds"] == ""
            # Must NOT expose primary boundary notes
            assert r["annotation_notes"] == ""
            assert r["annotator_id"] == "human_02"


def test_pairwise_tiou():
    w1 = [[3.0, 9.0]]
    w2 = [[3.0, 9.0]]
    assert compute_temporal_iou(w1, w2) == 1.0

    w_partial = [[5.0, 11.0]]
    assert compute_temporal_iou(w1, w_partial) == 0.5

    w_disjoint = [[15.0, 20.0]]
    assert compute_temporal_iou(w1, w_disjoint) == 0.0


def test_freeze_blocked_with_incomplete_primary_review(tmp_path: Path):
    # Running freeze on current workspace (where 46 rows are still DRAFT) must raise ValueError
    with pytest.raises(ValueError, match="Gate 1 Failed"):
        freeze_lumae_ads_pilot(force=False)


def test_freeze_blocked_with_insufficient_secondary_coverage(tmp_path: Path):
    # Mock primary fully reviewed, but zero secondary reviews
    mock_prim = [
        AnnotationRecord(
            sample_id=f"s_{i}",
            video_filename=f"v_{i}.mp4",
            vid=f"vid_{i}",
            source_dataset="AdsQA",
            source_video_id=f"h_{i}",
            source_split="train",
            source_url="http://example.com",
            product_category="Gadgets",
            query="Valid test query sentence",
            duration_seconds=30.0,
            gt_start_seconds=5.0,
            gt_end_seconds=15.0,
            annotator_id="human_01",
            review_status="REVIEWED",
        )
        for i in range(10)
    ]

    with patch("data_process.cli.freeze_lumae_ads_pilot.load_primary_records", return_value=mock_prim), \
         patch("data_process.cli.freeze_lumae_ads_pilot.load_secondary_records", return_value=[]), \
         patch("data_process.cli.freeze_lumae_ads_pilot.load_adjudicated_records", return_value=[]), \
         patch("data_process.cli.freeze_lumae_ads_pilot.load_multi_windows", return_value={}):
        with pytest.raises(ValueError, match="Gate 3 Failed"):
            freeze_lumae_ads_pilot(force=False, min_secondary_fraction=0.20)


def test_freeze_blocked_with_unresolved_adjudication(tmp_path: Path):
    # Mock primary and secondary with tIoU < 0.70 and no adjudication
    mock_prim = [
        AnnotationRecord(
            sample_id="s_1",
            video_filename="v_1.mp4",
            vid="vid_1",
            source_dataset="AdsQA",
            source_video_id="h_1",
            source_split="train",
            source_url="http://example.com",
            product_category="Gadgets",
            query="Valid test query sentence",
            duration_seconds=30.0,
            gt_start_seconds=2.0,
            gt_end_seconds=8.0,
            annotator_id="human_01",
            review_status="REVIEWED",
        )
    ]
    mock_sec = [
        AnnotationRecord(
            sample_id="s_1",
            video_filename="v_1.mp4",
            vid="vid_1",
            source_dataset="AdsQA",
            source_video_id="h_1",
            source_split="train",
            source_url="http://example.com",
            product_category="Gadgets",
            query="Valid test query sentence",
            duration_seconds=30.0,
            gt_start_seconds=15.0,
            gt_end_seconds=25.0,  # Completely disjoint -> tIoU = 0.0
            annotator_id="human_02",
            review_status="REVIEWED",
        )
    ]

    with patch("data_process.cli.freeze_lumae_ads_pilot.load_primary_records", return_value=mock_prim), \
         patch("data_process.cli.freeze_lumae_ads_pilot.load_secondary_records", return_value=mock_sec), \
         patch("data_process.cli.freeze_lumae_ads_pilot.load_adjudicated_records", return_value=[]), \
         patch("data_process.cli.freeze_lumae_ads_pilot.load_multi_windows", return_value={}):
        with pytest.raises(ValueError, match="Gate 4 Failed"):
            freeze_lumae_ads_pilot(force=False, min_secondary_fraction=0.20)


def test_successful_freeze_and_manifest_generation(tmp_path: Path):
    mock_prim = [
        AnnotationRecord(
            sample_id="s_1",
            video_filename="v_1.mp4",
            vid="vid_1",
            source_dataset="AdsQA",
            source_video_id="h_1",
            source_split="train",
            source_url="http://example.com",
            product_category="Gadgets",
            query="The presenter tests the gadget controls.",
            duration_seconds=30.0,
            gt_start_seconds=5.0,
            gt_end_seconds=15.0,
            annotator_id="human_01",
            review_status="REVIEWED",
        )
    ]
    mock_sec = [
        AnnotationRecord(
            sample_id="s_1",
            video_filename="v_1.mp4",
            vid="vid_1",
            source_dataset="AdsQA",
            source_video_id="h_1",
            source_split="train",
            source_url="http://example.com",
            product_category="Gadgets",
            query="The presenter tests the gadget controls.",
            duration_seconds=30.0,
            gt_start_seconds=5.2,
            gt_end_seconds=15.0,  # tIoU > 0.95 -> high agreement
            annotator_id="human_02",
            review_status="REVIEWED",
        )
    ]

    ann_dir = tmp_path / "annotations"
    rep_dir = tmp_path / "reports"
    can_dir = tmp_path / "canonical"
    man_dir = tmp_path / "manifests"
    for d in [ann_dir, rep_dir, can_dir, man_dir]:
        d.mkdir(parents=True, exist_ok=True)

    with patch("data_process.cli.freeze_lumae_ads_pilot.get_annotations_dir", return_value=ann_dir), \
         patch("data_process.cli.freeze_lumae_ads_pilot.get_reports_dir", return_value=rep_dir), \
         patch("data_process.cli.freeze_lumae_ads_pilot.get_repo_root", return_value=tmp_path), \
         patch("data_process.cli.freeze_lumae_ads_pilot.load_primary_records", return_value=mock_prim), \
         patch("data_process.cli.freeze_lumae_ads_pilot.load_secondary_records", return_value=mock_sec), \
         patch("data_process.cli.freeze_lumae_ads_pilot.load_adjudicated_records", return_value=[]), \
         patch("data_process.cli.freeze_lumae_ads_pilot.load_multi_windows", return_value={"s_1": [[5.0, 10.0], [12.0, 15.0]]}):

        manifest = freeze_lumae_ads_pilot(force=False, min_secondary_fraction=0.20)
        assert manifest["status"] if "status" in manifest else manifest["ground_truth"]["status"] == "FROZEN"
        assert manifest["total_samples"] == 1
        assert manifest["usable_sample_count"] == 1

        # Check files created
        assert (ann_dir / "frozen_gt.csv").is_file()
        assert (rep_dir / "annotation_agreement_human.json").is_file()
        # Verify multi-window was exported to canonical sample
        jsonl_files = list(tmp_path.glob("**/pilot_v1_human_verified.jsonl"))
        assert len(jsonl_files) == 1
        lines = jsonl_files[0].read_text(encoding="utf-8").strip().splitlines()
        sample = json.loads(lines[0])
        assert sample["relevant_windows"] == [[5.0, 10.0], [12.0, 15.0]]
        assert sample.get("saliency_scores") is None
        assert sample["source"] == "lumae_product_ads"

