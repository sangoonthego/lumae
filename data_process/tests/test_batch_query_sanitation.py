"""Stage A.2.6Q must emit advice without changing human or temporal data."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from data_process.annotation import batch_query_sanitation as batch
from data_process.annotation.assisted_deployment import SUGGESTION_FIELDS, _paths, _read_csv


def _csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def pilot(tmp_path: Path) -> Path:
    ann, _, videos = _paths(tmp_path)
    videos.mkdir(parents=True)
    writer = cv2.VideoWriter(str(videos / "playable.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                             5, (32, 32))
    assert writer.isOpened()
    for _ in range(5):
        writer.write(np.zeros((32, 32, 3), dtype=np.uint8))
    writer.release()
    _csv(ann / "human_primary.csv", [
        dict(sample_id="draft_playable", video_filename="playable.mp4", duration_seconds="1",
             query="Generic demonstration.", review_status="DRAFT"),
        dict(sample_id="draft_missing", video_filename="missing.mp4", duration_seconds="1",
             query="Another generic demonstration.", review_status="DRAFT"),
        dict(sample_id="human_reviewed", video_filename="playable.mp4", duration_seconds="1",
             query="Human reviewed query.", review_status="REVIEWED"),
    ])
    _csv(ann / "query_reviews.csv", [dict(sample_id="human_reviewed",
                                         human_final_query="Human approved event.",
                                         query_review_status="HUMAN_EDITED",
                                         reviewer_id="human_01")])
    (ann / "semantic_v3_deployment_preannotations.csv").write_text("sample_id,verified_query\n", encoding="utf-8")
    (ann / "assisted_annotation_audit.csv").write_text("sample_id,reviewer_id\n", encoding="utf-8")
    return tmp_path


def _proposal() -> dict[str, object]:
    return dict(original_query_quality="INCORRECT_FOR_VIDEO",
                ai_suggested_query="A hand moves a red object across the table.",
                query_change_type="CHANGED_INTENT", query_localizable=True,
                query_observable=True, visual_evidence="Decoded frames show a hand moving the object.",
                inspection_method="whole_video_frames", sampled_times_seconds=[0, 0.5])


def test_local_video_resolution_and_missing_handling(pilot: Path) -> None:
    videos = _paths(pilot)[2]
    assert batch.video_status(videos, "playable.mp4") == "PLAYABLE"
    assert batch.video_status(videos, "missing.mp4") == "VIDEO_MISSING"
    (videos / "broken.mp4").write_bytes(b"not video data")
    assert batch.video_status(videos, "broken.mp4") == "VIDEO_FAILED"
    with pytest.raises(ValueError, match="Unsafe video filename"):
        batch.video_status(videos, "../outside.mp4")


def test_batch_targets_draft_only_and_requires_inspection(pilot: Path) -> None:
    with pytest.raises(ValueError, match="DRAFT"):
        batch.build_rows(pilot, {"human_reviewed": _proposal()}, "2026-09-30T00:00:00Z")
    with pytest.raises(ValueError, match="lacks a suggestion"):
        batch.build_rows(pilot, {}, "2026-09-30T00:00:00Z")
    bad = _proposal()
    bad["reviewer_id"] = "human_01"
    with pytest.raises(ValueError, match="human or temporal fields"):
        batch.build_rows(pilot, {"draft_playable": bad}, "2026-09-30T00:00:00Z")


def test_write_batch_preserves_human_and_temporal_artifacts(pilot: Path) -> None:
    ann, reports, _ = _paths(pilot)
    before = {name: (ann / name).read_bytes() for name in batch.PROTECTED_FILES}
    proposals_path = pilot / "proposals.json"
    proposals_path.write_text(json.dumps({"draft_playable": _proposal()}), encoding="utf-8")

    summary = batch.write_batch(pilot, proposals_path)

    assert summary["total"] == 2
    assert summary["playable"] == 1
    assert summary["missing"] == 1
    assert {name: (ann / name).read_bytes() for name in batch.PROTECTED_FILES} == before
    suggestions = _read_csv(ann / "ai_query_suggestions.csv")
    assert {r["sample_id"] for r in suggestions} == {"draft_playable", "draft_missing"}
    assert set(SUGGESTION_FIELDS).issubset(suggestions[0])
    proposed = next(r for r in suggestions if r["sample_id"] == "draft_playable")
    assert proposed["status"] == "AI_SUGGESTED"
    assert proposed["query_localizable"] == proposed["query_observable"] == "TRUE"
    assert not {"reviewer_id", "human_final_query", "query_review_status",
                "gt_start_seconds", "gt_end_seconds"} & proposed.keys()
    assert next(r for r in suggestions if r["sample_id"] == "draft_missing")["status"] == "VIDEO_MISSING"
    assert len(_read_csv(reports / "batch_query_sanitation_review.csv")) == 2
    assert _read_csv(ann / "semantic_v3_deployment_preannotations.csv") == []
