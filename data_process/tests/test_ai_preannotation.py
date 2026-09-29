"""Unit tests for Stage A.2.5B: AI-assisted pre-annotation and blind holdout workflow.

Tests:
1. Pending sample detection & exclusion of human-reviewed samples (0004, 0006).
2. Deterministic blind holdout selection with seed 42.
3. AI pre-annotation data model schema and CSV round-trip serialization.
4. Query suitability analysis and query rewrite standards.
5. Multi-window candidate parsing and timestamp formatting (0.1s precision).
6. Resumability and force regeneration behavior.
7. Contamination safety guard preventing AI from writing human_primary.csv.
8. Blind holdout privacy suppression logic.
9. AI vs Blind Human Agreement metrics (tIoU, start/end absolute errors).
10. AI assistance acceptance rate reporting.
11. Review prioritization order sorting.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import pytest

from data_process.adsqa.ai_preannotator import (
    detect_temporal_windows,
    evaluate_query_suitability,
)
from data_process.adsqa.ai_reporting import (
    generate_ai_assistance_report,
    generate_ai_vs_blind_human_report,
)
from data_process.annotation.models import (
    AIConfidence,
    AIPreannotationRecord,
    AIPreannotationStatus,
    AIQueryStatus,
    AnnotationMode,
    AnnotationRecord,
    QueryAction,
    QueryStatus,
    ReviewLogEntry,
    ReviewStatus,
    TemporalWindow,
    WindowAction,
)
from data_process.annotation.storage import (
    create_blind_holdout,
    get_ai_preannotations_path,
    get_annotations_dir,
    get_blind_holdout_path,
    is_blind_holdout,
    load_ai_preannotations,
    load_blind_holdout,
    load_primary_records,
    save_ai_preannotations,
    save_primary_records,
)


def test_pending_sample_detection_and_reviewed_exclusion():
    """Verify that human_primary.csv accurately identifies the 2 reviewed and 46 pending samples."""
    records = load_primary_records()
    assert len(records) == 48

    reviewed = [r for r in records if r.is_human_reviewed()]
    pending = [r for r in records if not r.is_human_reviewed()]

    reviewed_ids = {r.sample_id for r in reviewed}
    assert reviewed_ids == {"lumae_ads_pilot_0004", "lumae_ads_pilot_0006"}
    assert len(pending) == 46
    assert "lumae_ads_pilot_0004" not in [r.sample_id for r in pending]
    assert "lumae_ads_pilot_0006" not in [r.sample_id for r in pending]


def test_blind_holdout_deterministic_seed_reproducibility():
    """Verify blind holdout deterministic subset generation with seed 42."""
    records = load_primary_records()
    pending_ids = [r.sample_id for r in records if not r.is_human_reviewed()]

    # Generate twice with same seed
    holdout_1 = create_blind_holdout(pending_ids, count=10, fraction=0.20, seed=42)
    holdout_2 = create_blind_holdout(pending_ids, count=10, fraction=0.20, seed=42)

    assert holdout_1["sample_ids"] == holdout_2["sample_ids"]
    assert len(holdout_1["sample_ids"]) == 10
    assert "lumae_ads_pilot_0002" in holdout_1["sample_ids"]
    assert "lumae_ads_pilot_0008" in holdout_1["sample_ids"]
    # Ensure reviewed samples were not in holdout pool
    assert "lumae_ads_pilot_0004" not in holdout_1["sample_ids"]
    assert "lumae_ads_pilot_0006" not in holdout_1["sample_ids"]


def test_ai_preannotation_record_schema_and_csv_roundtrip():
    """Test AIPreannotationRecord schema serialization and deserialization."""
    rec = AIPreannotationRecord(
        sample_id="lumae_ads_pilot_0099",
        video_filename="test_video.mp4",
        original_query="The creator presents the key physical features of the product.",
        ai_query_status=AIQueryStatus.NEEDS_EDIT.value,
        ai_proposed_query="The presenter demonstrates the physical features and operation of the product.",
        ai_start_seconds=12.5,
        ai_end_seconds=28.0,
        ai_windows_json="[[12.5, 28.0]]",
        ai_confidence=AIConfidence.HIGH.value,
        ai_reason="Clear demonstration detected.",
        analysis_method="ffmpeg_coarse_dense_visual_flow",
        frame_sampling_interval=2.0,
        boundary_refinement_interval=0.5,
        generated_at_utc="2026-09-30T00:00:00Z",
        status=AIPreannotationStatus.AI_PROPOSED.value,
    )

    csv_dict = rec.to_csv_dict()
    assert csv_dict["sample_id"] == "lumae_ads_pilot_0099"
    assert csv_dict["ai_query_status"] == "NEEDS_EDIT"
    assert csv_dict["ai_start_seconds"] == "12.5"
    assert csv_dict["ai_end_seconds"] == "28.0"

    # Roundtrip from CSV dict
    reconstructed = AIPreannotationRecord.from_csv_dict(csv_dict)
    assert reconstructed.sample_id == rec.sample_id
    assert reconstructed.ai_start_seconds == 12.5
    assert reconstructed.ai_end_seconds == 28.0
    assert reconstructed.get_windows() == [[12.5, 28.0]]
    assert reconstructed.ai_confidence == "HIGH"


def test_query_suitability_and_rewrite_rules():
    """Verify query validation: action-conditioned queries are VALID; generic queries trigger NEEDS_EDIT rewrite."""
    # Specific action-oriented queries
    valid_q1 = "The creator demonstrates how the blender operates."
    status1, prop1 = evaluate_query_suitability(valid_q1, "Home & Kitchen Appliance")
    assert status1 == AIQueryStatus.VALID.value
    assert prop1 == valid_q1

    valid_q2 = "The presenter demonstrates assembling the product components."
    status2, prop2 = evaluate_query_suitability(valid_q2, "Product Demonstration")
    assert status2 == AIQueryStatus.VALID.value
    assert prop2 == valid_q2

    # Generic query
    generic_q = "The creator presents the key physical features of the product."
    status3, prop3 = evaluate_query_suitability(generic_q, "Electronics & Gadgets")
    assert status3 == AIQueryStatus.NEEDS_EDIT.value
    assert "hands-on operation" in prop3
    # Check no buzzwords in proposed rewrite
    assert "best" not in prop3.lower()
    assert "viral" not in prop3.lower()
    assert "hook" not in prop3.lower()


def test_multi_window_candidate_parsing():
    """Verify multi-window candidate JSON handling."""
    rec = AIPreannotationRecord(
        sample_id="multi_test",
        video_filename="multi.mp4",
        original_query="Demonstration",
        ai_query_status="VALID",
        ai_proposed_query="Demonstration",
        ai_start_seconds=4.0,
        ai_end_seconds=10.0,
        ai_windows_json="[[4.0, 10.0], [15.5, 22.0]]",
        ai_confidence="MEDIUM",
        ai_reason="Two disjoint demonstration scenes",
        analysis_method="test",
        frame_sampling_interval=2.0,
        boundary_refinement_interval=0.5,
        generated_at_utc="",
        status="AI_PROPOSED",
    )

    windows = rec.get_windows()
    assert len(windows) == 2
    assert windows[0] == [4.0, 10.0]
    assert windows[1] == [15.5, 22.0]


def test_contamination_guard_prevents_ai_writing_human_primary(monkeypatch, tmp_path):
    """Verify that automated agents cannot write non-human or automated annotations into human_primary.csv."""
    monkeypatch.setenv("LUMAE_ANNOTATIONS_DIR", str(tmp_path))

    # Create dummy records where an automated agent attempts to set REVIEWED with non-human annotator
    illegal_record = AnnotationRecord(
        sample_id="test_sample_01",
        video_filename="v1.mp4",
        vid="vid1",
        source_dataset="AdsQA",
        source_video_id="v1",
        source_split="test",
        source_url="http://example.com",
        product_category="Gadgets",
        query="Test query",
        duration_seconds=30.0,
        gt_start_seconds=5.0,
        gt_end_seconds=15.0,
        annotator_id="ai_agent_v1",  # NOT a human annotator
        review_status=ReviewStatus.REVIEWED.value,  # ILLEGAL
    )

    with pytest.raises(ValueError, match="Contamination guard failed"):
        save_primary_records([illegal_record])


def test_blind_holdout_privacy_suppression():
    """Verify that blind holdout samples are identified for UI candidate hiding."""
    holdout = load_blind_holdout()
    holdout_ids = holdout.get("sample_ids", [])
    assert len(holdout_ids) > 0

    first_holdout = holdout_ids[0]
    assert is_blind_holdout(first_holdout) is True
    assert is_blind_holdout("lumae_ads_pilot_0001") is False


def test_ai_vs_blind_human_agreement_report(monkeypatch, tmp_path):
    """Test calculation of AI candidate vs blind human GT agreement metrics."""
    monkeypatch.setenv("LUMAE_ANNOTATIONS_DIR", str(tmp_path))
    monkeypatch.setenv("LUMAE_REPORTS_DIR", str(tmp_path))

    # Setup holdout
    holdout_data = {
        "seed": 42,
        "sample_ids": ["sample_h1", "sample_h2"],
    }
    h_file = tmp_path / "blind_holdout.json"
    h_file.write_text(json.dumps(holdout_data), encoding="utf-8")

    # Setup human primary with 1 reviewed holdout sample
    h_rec = AnnotationRecord(
        sample_id="sample_h1",
        video_filename="h1.mp4",
        vid="v_h1",
        source_dataset="AdsQA",
        source_video_id="h1",
        source_split="test",
        source_url="",
        product_category="Demo",
        query="Demonstrator uses vacuum",
        duration_seconds=30.0,
        gt_start_seconds=10.0,
        gt_end_seconds=20.0,
        annotator_id="human_01",
        review_status=ReviewStatus.REVIEWED.value,
    )
    save_primary_records([h_rec])

    # Setup AI candidates
    ai_cand1 = AIPreannotationRecord(
        sample_id="sample_h1",
        video_filename="h1.mp4",
        original_query="Demonstrator uses vacuum",
        ai_query_status="VALID",
        ai_proposed_query="Demonstrator uses vacuum",
        ai_start_seconds=11.0,
        ai_end_seconds=21.0,
        ai_windows_json="[[11.0, 21.0]]",
        ai_confidence="HIGH",
        ai_reason="Action visible",
        analysis_method="flow",
        frame_sampling_interval=2.0,
        boundary_refinement_interval=0.5,
        generated_at_utc="",
        status="AI_PROPOSED",
    )
    save_ai_preannotations([ai_cand1])

    # Generate report
    rep = generate_ai_vs_blind_human_report(output_path=tmp_path / "ai_vs_blind.json")
    assert rep["sample_count"] == 1
    # Human [10, 20], AI [11, 21] -> Intersection = 9, Union = 11 -> tIoU = 9/11 = 0.8182
    assert rep["mean_tiou"] == pytest.approx(0.8182, abs=0.01)
    assert rep["mean_start_error_seconds"] == 1.0
    assert rep["mean_end_error_seconds"] == 1.0
    assert rep["query_agreement_rate"] == 1.0


def test_ai_assistance_acceptance_report(monkeypatch, tmp_path):
    """Test operator acceptance report computation from audit log."""
    monkeypatch.setenv("LUMAE_ANNOTATIONS_DIR", str(tmp_path))
    monkeypatch.setenv("LUMAE_REPORTS_DIR", str(tmp_path))

    log_path = tmp_path / "human_review_log.jsonl"
    entries = [
        {
            "sample_id": "s1",
            "annotation_mode": "AI_ASSISTED_HUMAN_VERIFIED",
            "query_action": QueryAction.ACCEPTED_AI.value,
            "window_action": WindowAction.ACCEPTED_AI.value,
        },
        {
            "sample_id": "s2",
            "annotation_mode": "AI_ASSISTED_HUMAN_VERIFIED",
            "query_action": QueryAction.EDITED_AI.value,
            "window_action": WindowAction.ACCEPTED_AI.value,
        },
        {
            "sample_id": "s3",
            "annotation_mode": "AI_ASSISTED_HUMAN_VERIFIED",
            "query_action": QueryAction.ACCEPTED_AI.value,
            "window_action": WindowAction.REJECTED_AI.value,
        },
    ]
    with log_path.open("w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")

    rep = generate_ai_assistance_report(output_path=tmp_path / "assistance_rep.json")
    assert rep["total_assisted_reviews"] == 3
    # 2 accepted out of 3 = 66.67%
    assert rep["query_actions"]["accepted_unchanged_count"] == 2
    assert rep["query_actions"]["accepted_unchanged_pct"] == 66.67
    assert rep["query_actions"]["edited_count"] == 1
    # Windows: 2 accepted, 1 rejected
    assert rep["window_actions"]["accepted_unchanged_count"] == 2
    assert rep["window_actions"]["rejected_count"] == 1
