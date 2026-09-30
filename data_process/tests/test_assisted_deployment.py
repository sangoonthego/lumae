"""Stage A.2.6 deployment gates and explicit human decisions."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from data_process.annotation import assisted_deployment as deployment


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def pilot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    ann, _, videos = deployment._paths(tmp_path)
    videos.mkdir(parents=True)
    (videos / "video.mp4").write_bytes(b"local video fixture")
    write_csv(ann / "human_primary.csv", [dict(
        sample_id="sample_1", video_filename="video.mp4", vid="vid_1", source_dataset="AdsQA",
        source_video_id="video", source_split="test", source_url="", product_category="", query="Draft query",
        duration_seconds="10", gt_start_seconds="", gt_end_seconds="", annotator_id="ann_lead",
        annotation_notes="", review_status="DRAFT")])
    write_csv(ann / "query_reviews.csv", [dict(
        sample_id="sample_1", video_filename="video.mp4", original_query="Draft query",
        ai_suggested_query="", human_final_query="", query_review_status="PENDING_QUERY_REVIEW",
        original_query_quality="", query_change_type="", query_localizable="", query_observable="",
        reviewer_id="", review_notes="", reviewed_at_utc="", query_version="1")])
    write_csv(ann / "query_sanitation_worklist.csv", [dict(sample_id="sample_1")])
    manifest = tmp_path / "local_data/manifests/temporal_exposure_registry.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"total_pilot_samples": 1}), encoding="utf-8")
    monkeypatch.setattr(deployment, "verify_frozen_ranker", lambda root: None)
    return tmp_path


def approve(root: Path) -> None:
    path = deployment._paths(root)[0] / "query_reviews.csv"
    row = deployment._read_csv(path)[0]
    row.update(human_final_query="A person turns the dial.", query_review_status="HUMAN_EDITED",
               query_localizable="TRUE", query_observable="TRUE", reviewer_id="human_01")
    deployment._atomic_csv(path, list(row), [row])


def add_candidates(root: Path) -> None:
    path = deployment._paths(root)[0] / "semantic_v3_deployment_candidate_pools.json"
    path.write_text(json.dumps({"sample_1": [dict(
        candidate_id="A", start_coarse=1, end_coarse=4, visible_subject="person",
        visible_action="turns dial", visible_object="dial", supporting_visual_evidence="Person's hand turns dial in inspected frames"),
        dict(candidate_id="B", start_coarse=5, end_coarse=9, visible_subject="person",
             visible_action="walks away", visible_object="door", supporting_visual_evidence="Person walks to door in inspected frames")]}), encoding="utf-8")


def test_suggestion_does_not_approve_query_or_enter_gt(pilot: Path) -> None:
    deployment.prepare(pilot)
    ann = deployment._paths(pilot)[0]
    before_primary = (ann / "human_primary.csv").read_bytes()
    deployment.save_ai_suggestions([dict(sample_id="sample_1", video_filename="video.mp4",
        ai_suggested_query="A person turns a dial.", query_quality_suggestion="INCORRECT_FOR_VIDEO",
        observable_evidence="Hand and dial visible", inspection_method="sampled video frames",
        sampled_times_seconds="[1, 3, 5]", generated_at_utc="2026-09-30T00:00:00Z")], pilot)
    assert not deployment.approved_query(deployment.load_state(pilot)["queries"]["sample_1"])
    assert (ann / "human_primary.csv").read_bytes() == before_primary
    assert deployment.precompute(pilot)["generated"] == 0
    assert deployment._read_csv(ann / "semantic_v3_deployment_preannotations.csv") == []


@pytest.mark.parametrize("field,value", [
    ("reviewer_id", "ann_lead"), ("human_final_query", ""), ("query_localizable", "FALSE"),
    ("query_observable", "FALSE"), ("query_review_status", "PENDING_QUERY_REVIEW"),
])
def test_query_approval_gate_rejects_missing_requirement(pilot: Path, field: str, value: str) -> None:
    approve(pilot)
    q = deployment.load_state(pilot)["queries"]["sample_1"]
    q[field] = value
    assert not deployment.approved_query(q)


def test_precompute_uses_verified_query_and_refuses_generic_fallback(pilot: Path) -> None:
    approve(pilot)
    deployment.prepare(pilot)
    assert deployment.precompute(pilot)["missing_candidate_pool"] == 1
    add_candidates(pilot)
    result = deployment.precompute(pilot)
    assert result["generated"] == 1
    row = deployment._read_csv(deployment._paths(pilot)[0] / "semantic_v3_deployment_preannotations.csv")[0]
    assert row["verified_query"] == "A person turns the dial."
    assert row["verified_query"] != "Draft query"
    assert row["human_review_status"] == "PENDING_TEMPORAL_REVIEW"
    assert row["algorithm_sha256"] == deployment.ALGORITHM_SHA256
    assert deployment._read_csv(deployment._paths(pilot)[0] / "human_primary.csv")[0]["review_status"] == "DRAFT"


@pytest.mark.parametrize("decision,bounds,provenance", [
    ("ACCEPT_AI", (None, None), "AI_ACCEPTED"),
    ("EDIT_AI", (2, 4), "AI_EDITED"),
    ("REJECT_AI", (6, 8), "AI_REJECTED"),
    ("EXCLUDE", (None, None), "EXCLUDED"),
])
def test_human_decision_audit_and_readiness(pilot: Path, decision: str, bounds: tuple, provenance: str) -> None:
    approve(pilot)
    deployment.prepare(pilot)
    add_candidates(pilot)
    deployment.precompute(pilot)
    deployment.submit_temporal_decision("sample_1", decision, "human_02", *bounds, notes="Reviewed video", root=pilot)
    ann = deployment._paths(pilot)[0]
    primary = deployment._read_csv(ann / "human_primary.csv")[0]
    audit = deployment._read_csv(ann / "assisted_annotation_audit.csv")[0]
    pre = deployment._read_csv(ann / "semantic_v3_deployment_preannotations.csv")[0]
    assert primary["annotator_id"] == "human_02"
    assert provenance in primary["annotation_notes"]
    assert audit["temporal_decision"] == pre["human_review_status"] == decision
    report = deployment.readiness(pilot)
    assert report["READY_FOR_DATASET_FREEZE"] == "YES"
    assert report["pending"] == 0


def test_human_and_temporal_validation(pilot: Path) -> None:
    approve(pilot)
    deployment.prepare(pilot)
    add_candidates(pilot)
    deployment.precompute(pilot)
    for reviewer, start, end in [("ann_lead", 2, 4), ("human_02", -1, 3),
                                 ("human_02", 5, 5), ("human_02", 1, 11)]:
        with pytest.raises(ValueError):
            deployment.submit_temporal_decision("sample_1", "EDIT_AI", reviewer, start, end, root=pilot)
    assert deployment._read_csv(deployment._paths(pilot)[0] / "assisted_annotation_audit.csv") == []


def test_atomic_csv_preserves_existing_on_replace_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "data.csv"
    path.write_text("old contents", encoding="utf-8")
    monkeypatch.setattr(deployment.os, "replace", lambda *_: (_ for _ in ()).throw(OSError("replace failed")))
    with pytest.raises(OSError):
        deployment._atomic_csv(path, ("sample_id",), [{"sample_id": "new"}])
    assert path.read_text(encoding="utf-8") == "old contents"
    assert list(tmp_path.iterdir()) == [path]


def test_frozen_ranker_source_hashes_match_manifest() -> None:
    deployment.verify_frozen_ranker(deployment.ROOT)
    manifest = json.loads((deployment.ROOT / "local_data/manifests/semantic_v3_algorithm_freeze.json").read_text())
    for name, digest in manifest["source_file_hashes"].items():
        assert hashlib.sha256((deployment.ROOT / name).read_bytes()).hexdigest() == digest


def test_readiness_blocks_frozen_provenance_mismatch(pilot: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(deployment, "frozen_artifact_errors", lambda root: ["Frozen selection hash mismatch"])
    report = deployment.readiness(pilot)
    assert report["READY_FOR_DATASET_FREEZE"] == "NO"
    assert "Frozen selection hash mismatch" in report["validation_errors"]


def test_assisted_streamlit_modes_show_video_suggestion_and_gate() -> None:
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(deployment.ROOT / "data_process/annotation/app.py"), default_timeout=15).run()
    assert not app.exception
    app.radio[1].set_value("QUERY_SANITATION").run()
    assert not app.exception
    assert any("A man rinses vegetables" in item.value for item in app.info)
    checks = {item.label: item.value for item in app.checkbox}
    assert checks["I watched this video"] is False
    assert checks["The final action has a localizable start and end"] is False
    assert checks["The final action is visibly observable"] is False
    accept = next(button for button in app.button if button.label == "Accept Suggestion · Lock & Predict")
    reviews_path = deployment.ROOT / "local_data/annotations/lumae_ads/query_reviews.csv"
    before = hashlib.sha256(reviews_path.read_bytes()).hexdigest()
    accept.click().run()
    assert not app.exception
    assert any("actual human_ reviewer ID" in item.value for item in app.error)
    assert hashlib.sha256(reviews_path.read_bytes()).hexdigest() == before
    app.radio[0].set_value("ASSISTED_TEMPORAL_REVIEW").run()
    assert not app.exception
    assert any("Human query approval is required" in item.value for item in app.warning)
    assert len(app.get("video")) == 1
