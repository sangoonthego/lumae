"""Write video-inspected AI query advice without granting human approval."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import cv2

from data_process.annotation.assisted_deployment import (
    ROOT, SUGGESTION_FIELDS, _atomic_csv, _atomic_json, _paths, _read_csv,
    _unique,
)

QUALITY = {"VALID", "VAGUE_BUT_RELEVANT", "INCORRECT_FOR_VIDEO", "INVALID_OR_UNLOCALIZABLE"}
CHANGE = {"SAME_INTENT", "NARROWED_INTENT", "CHANGED_INTENT"}
REVIEW_FIELDS = ("sample_id", "original_query", "quality", "ai_suggested_query",
                 "change_type", "localizable", "visual_evidence")
PROTECTED_FILES = ("human_primary.csv", "query_reviews.csv",
                   "semantic_v3_deployment_preannotations.csv",
                   "assisted_annotation_audit.csv")
FORBIDDEN_PROPOSAL_FIELDS = {"human_final_query", "reviewer_id", "reviewed_at_utc",
                             "query_review_status", "gt_start_seconds", "gt_end_seconds"}


def _sha256(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def video_status(videos: Path, filename: str) -> str:
    """Resolve only a direct local filename and verify that a frame decodes."""
    if not filename or Path(filename).name != filename:
        raise ValueError(f"Unsafe video filename: {filename!r}")
    path = videos / filename
    if not path.is_file():
        return "VIDEO_MISSING"
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            return "VIDEO_FAILED"
        ok, _ = capture.read()
        return "PLAYABLE" if ok else "VIDEO_FAILED"
    finally:
        capture.release()


def build_rows(root: Path, proposals: dict[str, dict[str, object]],
               generated_at_utc: str) -> tuple[list[dict[str, str]], dict[str, int]]:
    ann, _, videos = _paths(root)
    primary = _unique(_read_csv(ann / "human_primary.csv"), "human_primary.csv")
    drafts = {sid: row for sid, row in primary.items() if row["review_status"] == "DRAFT"}
    if not set(proposals).issubset(drafts):
        raise ValueError("Proposals may target DRAFT samples only")
    rows: list[dict[str, str]] = []
    counts: Counter[str] = Counter()
    for sid, source in sorted(drafts.items()):
        state = video_status(videos, source["video_filename"])
        proposal = proposals.get(sid)
        if state != "PLAYABLE":
            if proposal is not None:
                raise ValueError(f"Cannot suggest for an unplayable video: {sid}")
            row = {key: "" for key in SUGGESTION_FIELDS}
            row.update(sample_id=sid, video_filename=source["video_filename"],
                       original_query=source["query"], status=state,
                       visual_evidence="Local video could not be inspected")
            rows.append(row)
            counts[state] += 1
            continue
        if proposal is None:
            raise ValueError(f"Playable DRAFT sample lacks a suggestion: {sid}")
        if FORBIDDEN_PROPOSAL_FIELDS & set(proposal):
            raise ValueError(f"Proposal contains human or temporal fields: {sid}")
        quality = str(proposal.get("original_query_quality", ""))
        change = str(proposal.get("query_change_type", ""))
        query = str(proposal.get("ai_suggested_query", "")).strip()
        evidence = str(proposal.get("visual_evidence", "")).strip()
        method = str(proposal.get("inspection_method", "")).strip()
        times = proposal.get("sampled_times_seconds")
        if quality not in QUALITY or change not in CHANGE:
            raise ValueError(f"Invalid quality or change classification: {sid}")
        if not query or not evidence or not method or not isinstance(times, list) or not times:
            raise ValueError(f"Incomplete video-inspected suggestion: {sid}")
        if any(not isinstance(t, (int, float)) or t < 0 or t > float(source["duration_seconds"]) + 1 for t in times):
            raise ValueError(f"Invalid sampled video time: {sid}")
        if proposal.get("query_localizable") is not True or proposal.get("query_observable") is not True:
            raise ValueError(f"Suggestion must be localizable and observable: {sid}")
        if quality == "INCORRECT_FOR_VIDEO" and change != "CHANGED_INTENT":
            raise ValueError(f"Incorrect original needs changed intent: {sid}")
        row = {key: "" for key in SUGGESTION_FIELDS}
        row.update(sample_id=sid, video_filename=source["video_filename"],
                   original_query=source["query"], original_query_quality=quality,
                   ai_suggested_query=query, query_change_type=change,
                   query_localizable="TRUE", query_observable="TRUE",
                   visual_evidence=evidence, inspection_method=method,
                   generated_at_utc=generated_at_utc, status="AI_SUGGESTED",
                   sampled_times_seconds=json.dumps(times),
                   query_quality_suggestion=quality, observable_evidence=evidence)
        rows.append(row)
        counts[quality] += 1
        counts["AI_SUGGESTED"] += 1
    return rows, dict(counts)


def write_batch(root: Path = ROOT, proposals_path: Path | None = None) -> dict[str, object]:
    """Validate the entire batch, then write advice and review artifacts only."""
    if proposals_path is None:
        proposals_path = root / "local_data/manifests/a26q_audit/curated_suggestions.json"
    proposals = json.loads(proposals_path.read_text(encoding="utf-8"))
    if not isinstance(proposals, dict):
        raise ValueError("Proposals must be keyed by sample_id")
    ann, reports, _ = _paths(root)
    before = {name: _sha256(ann / name) for name in PROTECTED_FILES}
    now = datetime.now(timezone.utc).isoformat()
    rows, counts = build_rows(root, proposals, now)
    review = [dict(sample_id=row["sample_id"], original_query=row["original_query"],
                   quality=row["original_query_quality"],
                   ai_suggested_query=row["ai_suggested_query"],
                   change_type=row["query_change_type"],
                   localizable=row["query_localizable"],
                   visual_evidence=row["visual_evidence"]) for row in rows]
    summary = dict(total=len(rows), playable=counts.get("AI_SUGGESTED", 0),
                   missing=counts.get("VIDEO_MISSING", 0), failed=counts.get("VIDEO_FAILED", 0),
                   quality={name: counts.get(name, 0) for name in sorted(QUALITY)},
                   rewrite_recommended_count=sum(row["ai_suggested_query"] != row["original_query"] for row in rows if row["status"] == "AI_SUGGESTED"),
                   change_intent_count=sum(row["query_change_type"] == "CHANGED_INTENT" for row in rows),
                   narrowed_intent_count=sum(row["query_change_type"] == "NARROWED_INTENT" for row in rows),
                   generated_at_utc=now, protected_hashes=before)
    _atomic_csv(ann / "ai_query_suggestions.csv", SUGGESTION_FIELDS, rows)
    _atomic_csv(reports / "batch_query_sanitation_review.csv", REVIEW_FIELDS, review)
    _atomic_json(reports / "batch_query_sanitation_summary.json", summary)
    after = {name: _sha256(ann / name) for name in PROTECTED_FILES}
    if after != before:
        raise RuntimeError("Protected human or temporal artifact changed during suggestion batch")
    return summary


if __name__ == "__main__":
    print(json.dumps(write_batch(), indent=2))
