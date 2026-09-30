"""Per-sample human query lock, frozen-v3 advice, and human temporal commit.

The candidate generator uses only local video frame changes and prior whole-video
visual evidence. Its intervals are advisory; frozen semantic_v3 ranking is not
modified or fitted on this pilot.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile

from data_process.adsqa.semantic_preannotator_v3 import (
    SemanticCandidateEvent, decompose_query, rank_and_select_candidates,
)
from data_process.annotation import assisted_deployment as deployment
from data_process.annotation.human_query_review import (
    _paths as review_paths, decision_of, progress as query_progress,
    submit_decision, worklist,
)
from data_process.annotation.query_sanitation import load_query_reviews


ROOT = Path(__file__).resolve().parents[2]
FRAME_STEP_SECONDS = 1.0
MAX_CANDIDATES = 5
PROPOSAL_HALF_WIDTH_SECONDS = 2.5
PROPOSAL_MIN_GAP_SECONDS = 3.0


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_path(root: Path, sid: str, version: str, kind: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_]+", sid) or not re.fullmatch(r"[0-9]+", version):
        raise ValueError("Unsafe sample ID or query version")
    folder = "a26_fastlane_query_locks" if kind == "query_lock" else "a26_fastlane_preannotations"
    return root / "local_data" / "manifests" / folder / f"{sid}.v{version}.{kind}.json"


def _create_immutable_json(path: Path, payload: dict) -> dict:
    data = (json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f"Immutable artifact collision: {path}")
        return payload
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = None
    try:
        with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as stream:
            tmp = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(tmp, path)
        except FileExistsError:
            if path.read_bytes() != data:
                raise ValueError(f"Immutable artifact collision: {path}")
    finally:
        if tmp is not None and tmp.exists():
            tmp.unlink()
    return payload


def _current(root: Path, sid: str) -> tuple[dict[str, str], object, dict[str, str]]:
    target = {r["sample_id"]: r for r in worklist(root)}.get(sid)
    if target is None:
        raise ValueError("Sample is outside the fast-lane cohort")
    query = load_query_reviews(review_paths(root)[2]).get(sid)
    if query is None or not decision_of(query):
        raise ValueError("Explicit human query decision is required")
    primary = deployment._unique(deployment._read_csv(review_paths(root)[0]), "human_primary.csv")[sid]
    return target, query, primary


def lock_query(root: Path, sid: str) -> dict:
    """Create an immutable versioned lock from an explicit human approval."""
    target, query, primary = _current(root, sid)
    if query.query_review_status not in {"HUMAN_EDITED", "VALID_AS_IS"}:
        raise ValueError("Only approved queries can be locked")
    if not deployment.approved_query(query.to_csv_dict()) or not query.reviewed_at_utc:
        raise ValueError("Incomplete human query approval")
    if primary["review_status"] != "DRAFT":
        raise ValueError("Cannot create a new lock after temporal review")
    deployment.verify_frozen_ranker(root)
    payload = {
        "sample_id": sid, "video_filename": target["video_filename"],
        "human_final_query": query.human_final_query,
        "query_version": query.query_version, "reviewer_id": query.reviewer_id,
        "reviewed_at_utc": query.reviewed_at_utc,
        "original_query": query.original_query,
        "ai_suggested_query": query.ai_suggested_query,
        "query_review_status": query.query_review_status,
        "review_notes": query.review_notes,
        "query_sha256": _sha(query.human_final_query.encode("utf-8")),
    }
    return _create_immutable_json(_manifest_path(root, sid, query.query_version, "query_lock"), payload)


def current_lock(root: Path, sid: str) -> dict | None:
    try:
        _, query, _ = _current(root, sid)
    except ValueError:
        return None
    path = _manifest_path(root, sid, query.query_version, "query_lock")
    if not path.is_file():
        return None
    lock = json.loads(path.read_text(encoding="utf-8"))
    if (lock.get("human_final_query") != query.human_final_query
            or lock.get("query_sha256") != _sha(query.human_final_query.encode("utf-8"))
            or lock.get("reviewer_id") != query.reviewer_id
            or lock.get("reviewed_at_utc") != query.reviewed_at_utc):
        raise ValueError(f"Current query lock mismatch: {sid}")
    return lock


def _video_candidates(video: Path, duration: float, evidence: str) -> tuple[list[SemanticCandidateEvent], dict]:
    """Fixed frame-change proposals; no sample-specific tuning or GT input."""
    try:
        import cv2
    except ImportError as exc:
        raise ValueError("OpenCV is required for local video candidate extraction") from exc

    if not video.is_file() or duration <= 0:
        raise ValueError("Local video is unavailable or has invalid duration")
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f"Cannot decode local video: {video.name}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps > 0 and frame_count > 0:
        actual_duration = frame_count / fps
        if abs(actual_duration - duration) > max(1.0, duration * 0.05):
            capture.release()
            raise ValueError(f"Local video duration differs from primary metadata: {video.name}")
    observations: list[tuple[float, float]] = []
    previous = None
    try:
        t = 0.0
        while t < duration:
            capture.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ok, frame = capture.read()
            if ok:
                gray = cv2.cvtColor(cv2.resize(frame, (64, 36)), cv2.COLOR_BGR2GRAY)
                score = float(cv2.absdiff(gray, previous).mean()) if previous is not None else 0.0
                observations.append((round(t, 2), score))
                previous = gray
            t += FRAME_STEP_SECONDS
    finally:
        capture.release()
    if len(observations) < 2:
        raise ValueError(f"Insufficient decodable frames for {video.name}")
    ranked = sorted(observations[1:], key=lambda entry: (-entry[1], entry[0]))
    selected: list[tuple[float, float]] = []
    for center, score in ranked:
        if all(abs(center - old) >= PROPOSAL_MIN_GAP_SECONDS for old, _ in selected):
            selected.append((center, score))
            if len(selected) == MAX_CANDIDATES:
                break
    if not selected:
        raise ValueError("No video-derived candidate intervals")
    candidates = []
    for idx, (center, motion) in enumerate(selected):
        start = max(0.0, round(center - PROPOSAL_HALF_WIDTH_SECONDS, 1))
        end = min(duration, round(center + PROPOSAL_HALF_WIDTH_SECONDS, 1))
        if end - start < 0.1:
            continue
        candidates.append(SemanticCandidateEvent(
            candidate_id=chr(ord("A") + idx), start_coarse=start, end_coarse=end,
            visible_subject="video scene", visible_action="visible frame change",
            visible_object="objects in the local video",
            supporting_visual_evidence=(
                f"Local video frame-change magnitude {motion:.3f} at {center:.1f}s. "
                f"Whole-video AI visual evidence (not localized): {evidence}"
            ),
        ))
    if not candidates:
        raise ValueError("Video-derived windows are too short")
    return candidates, {"method": "fixed_frame_difference_v1", "frame_step_seconds": FRAME_STEP_SECONDS,
                        "sampled_frame_count": len(observations),
                        "motion_peak_times_seconds": [round(t, 2) for t, _ in selected]}


def current_prediction(root: Path, sid: str) -> dict | None:
    lock = current_lock(root, sid)
    if lock is None:
        return None
    path = _manifest_path(root, sid, lock["query_version"], "preannotation")
    if not path.is_file():
        return None
    pred = json.loads(path.read_text(encoding="utf-8"))
    if (pred.get("query_sha256") != lock["query_sha256"]
            or pred.get("query_version") != lock["query_version"]
            or pred.get("semantic_v3_algorithm_sha256") != deployment.ALGORITHM_SHA256
            or pred.get("status") != "AI_PREANNOTATION"):
        raise ValueError(f"Current temporal preannotation is stale or invalid: {sid}")
    return pred


def _sync_deployment_preannotation(root: Path, pred: dict) -> None:
    ann = deployment._paths(root)[0]
    path = ann / "semantic_v3_deployment_preannotations.csv"
    rows = deployment._unique(deployment._read_csv(path), path.name)
    sid = pred["sample_id"]
    row = {
        "sample_id": sid, "video_filename": pred["video_filename"],
        "verified_query": pred["human_final_query"],
        "candidate_windows_json": json.dumps(pred["candidate_windows"]),
        "selected_candidate_window": json.dumps([pred["pred_start_seconds"], pred["pred_end_seconds"]]),
        "selected_candidate_rank": "1", "candidate_count": str(len(pred["candidate_windows"])),
        "semantic_confidence": pred["semantic_confidence"],
        "ranking_margin": str(pred["ranking_margin"]),
        "semantic_reason": pred["semantic_reason"],
        "algorithm_version": deployment.ALGORITHM_VERSION,
        "algorithm_sha256": deployment.ALGORITHM_SHA256,
        "query_version": pred["query_version"],
        "generated_at_utc": pred["generated_at_utc"],
        "human_review_status": "PENDING_TEMPORAL_REVIEW",
    }
    old = rows.get(sid)
    if old and old.get("human_review_status") not in {"PENDING_TEMPORAL_REVIEW", "SUPERSEDED"}:
        if old.get("query_version") != row["query_version"]:
            raise ValueError("Cannot replace a human-reviewed temporal preannotation")
        return
    if old != row:
        rows[sid] = row
        deployment._atomic_csv(path, deployment.PREANNOTATION_FIELDS,
                               [rows[key] for key in sorted(rows)])


def ensure_prediction(root: Path, sid: str) -> dict:
    """Lock then infer for one approved sample; never writes human temporal GT."""
    target, query, primary = _current(root, sid)
    if primary["review_status"] != "DRAFT":
        raise ValueError("Sample is already temporally resolved")
    lock = current_lock(root, sid) or lock_query(root, sid)
    deployment.verify_frozen_ranker(root)
    existing = current_prediction(root, sid)
    if existing is not None:
        _sync_deployment_preannotation(root, existing)
        return existing
    video = deployment._paths(root)[2] / target["video_filename"]
    duration = float(primary["duration_seconds"])
    candidates, proposal_meta = _video_candidates(video, duration, target["suggestion_visual_evidence"])
    top, margin, confidence, reason = rank_and_select_candidates(
        decompose_query(lock["human_final_query"]), candidates)
    start, end = top.get_effective_window()
    if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end <= duration):
        raise ValueError("Frozen ranker returned an invalid temporal interval")
    payload = {
        "sample_id": sid, "video_filename": target["video_filename"],
        "human_final_query": lock["human_final_query"],
        "query_version": lock["query_version"],
        "query_sha256": lock["query_sha256"],
        "query_lock_sha256": _file_sha(_manifest_path(root, sid, lock["query_version"], "query_lock")),
        "semantic_v3_algorithm_sha256": deployment.ALGORITHM_SHA256,
        "pred_start_seconds": start, "pred_end_seconds": end,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "AI_PREANNOTATION",
        "candidate_windows": [c.get_effective_window() for c in candidates],
        "candidate_scores": [{"candidate_id": c.candidate_id, "score": c.query_match_score}
                             for c in candidates],
        "selected_candidate_id": top.candidate_id,
        "ranking_margin": margin, "semantic_confidence": confidence,
        "semantic_reason": reason,
        "semantic_phase_information": {"query_decomposition": vars(decompose_query(lock["human_final_query"])),
                                       "proposal": proposal_meta},
        "video_sha256": _file_sha(video),
        "score_notice": "Ranking scores are uncalibrated and the frame-change candidates are advisory.",
    }
    _create_immutable_json(_manifest_path(root, sid, lock["query_version"], "preannotation"), payload)
    _sync_deployment_preannotation(root, payload)
    return payload


def submit_query_and_predict(root: Path, **decision_args) -> dict | None:
    """UI button handler: human decision -> lock -> one-sample inference."""
    rec = submit_decision(root, freeze_batch=False, **decision_args)
    if rec.query_review_status == "EXCLUDE":
        submit_temporal(root, rec.sample_id, "EXCLUDE", rec.reviewer_id,
                        note=decision_args.get("note", ""), confirm_exclude=True)
        return None
    return ensure_prediction(root, rec.sample_id)


def revise_query_and_predict(root: Path, **decision_args) -> dict:
    sid = decision_args["sample_id"]
    _, old, primary = _current(root, sid)
    if primary["review_status"] != "DRAFT" or old.query_review_status == "EXCLUDE":
        raise ValueError("Only unresolved, approved queries can be revised")
    if not decision_args.get("note", "").strip():
        raise ValueError("Query revision requires a human reason")
    audit = deployment._unique(deployment._read_csv(
        deployment._paths(root)[0] / "assisted_annotation_audit.csv"), "assisted audit")
    if sid in audit:
        raise ValueError("Finish the pending human temporal commit before revising the query")
    # Preserve the previous human-approved version even if its prediction failed.
    lock_query(root, sid)
    rec = submit_decision(root, allow_revision=True, freeze_batch=False, **decision_args)
    if rec.query_review_status == "EXCLUDE":
        raise ValueError("Use the explicit temporal exclusion action")
    tombstone = (root / "local_data/manifests/a26_fastlane_preannotations"
                 / f"{sid}.v{old.query_version}.superseded.json")
    _create_immutable_json(tombstone, {
        "sample_id": sid, "superseded_query_version": old.query_version,
        "superseded_query_sha256": _sha(old.human_final_query.encode("utf-8")),
        "superseded_by_query_version": rec.query_version,
        "superseded_at_utc": rec.reviewed_at_utc,
        "reviewer_id": rec.reviewer_id,
        "human_reason": decision_args["note"].strip(),
        "old_prediction_valid_for_current_query": False,
    })
    return ensure_prediction(root, sid)


def submit_temporal(root: Path, sid: str, decision: str, reviewer_id: str,
                    start: float | None = None, end: float | None = None,
                    note: str = "", confirm_exclude: bool = False) -> None:
    """A human-click handler. Existing deployment audit is durable before GT."""
    reviewer = reviewer_id.strip()
    if not re.fullmatch(r"human_[A-Za-z0-9_-]+", reviewer):
        raise ValueError("Enter your actual human_ temporal reviewer ID")
    if decision not in deployment.DECISIONS:
        raise ValueError("Choose an explicit temporal decision")
    _, query, primary = _current(root, sid)
    if primary["review_status"] != "DRAFT":
        raise ValueError("Sample is already temporally resolved")
    if decision == "EXCLUDE":
        if not confirm_exclude or not note.strip():
            raise ValueError("Temporal exclusion requires confirmation and a human reason")
    else:
        if current_lock(root, sid) is None:
            raise ValueError("Human query lock is required before temporal review")
        pred = current_prediction(root, sid)
        if pred is None:
            raise ValueError("Current semantic_v3 preannotation is required")
        video = deployment._paths(root)[2] / primary["video_filename"]
        if _file_sha(video) != pred["video_sha256"]:
            raise ValueError("Source video changed after temporal preannotation")
        duration = float(primary["duration_seconds"])
        if decision == "ACCEPT_AI":
            if start is not None or end is not None:
                raise ValueError("Accept must use the AI interval exactly")
        else:
            if start is None or end is None or not (math.isfinite(float(start)) and math.isfinite(float(end))
                                                    and 0 <= float(start) < float(end) <= duration):
                raise ValueError("Manual interval must satisfy 0 <= start < end <= duration")
            if (float(start), float(end)) == (pred["pred_start_seconds"], pred["pred_end_seconds"]):
                raise ValueError("Edit or reject requires manually changed boundaries")
        csv_pre = deployment._unique(deployment._read_csv(
            deployment._paths(root)[0] / "semantic_v3_deployment_preannotations.csv"), "preannotations")
        row = csv_pre.get(sid)
        if (not row or row["query_version"] != pred["query_version"]
                or row["verified_query"] != query.human_final_query
                or row["human_review_status"] != "PENDING_TEMPORAL_REVIEW"):
            raise ValueError("Deployment preannotation is stale")
    provenance_note = (
        "Human verified AI temporal preannotation" if decision == "ACCEPT_AI"
        else "Human corrected AI temporal preannotation" if decision in {"EDIT_AI", "REJECT_AI"}
        else "Human excluded sample"
    )
    full_note = provenance_note + (f"; {note.strip()}" if note.strip() else "")
    deployment.submit_temporal_decision(sid, decision, reviewer, start, end, full_note, root=root)
    from data_process.annotation.fastlane_freeze import freeze_if_complete
    freeze_if_complete(root)


def recover_incomplete_commits(root: Path) -> int:
    """Finish a human-audited commit interrupted between atomic CSV writes.

    The audit is written first by submit_temporal_decision. It records the
    explicit human action, reviewer, timestamp, query version/hash, and final
    interval. Recovery only replays that durable action; it makes no decision.
    """
    ann = deployment._paths(root)[0]
    audit = deployment._unique(deployment._read_csv(ann / "assisted_annotation_audit.csv"), "audit")
    rows = deployment._read_csv(ann / "human_primary.csv")
    primary = deployment._unique(rows, "human_primary.csv")
    pre_path = ann / "semantic_v3_deployment_preannotations.csv"
    pre = deployment._unique(deployment._read_csv(pre_path), "preannotations")
    cohort = {r["sample_id"] for r in worklist(root)}
    recovered = 0
    primary_changed = pre_changed = False
    for sid in sorted(cohort & set(audit)):
        entry = audit[sid]
        row = primary[sid]
        decision = entry["temporal_decision"]
        expected_status = "EXCLUDED" if decision == "EXCLUDE" else "REVIEWED"
        if (decision not in deployment.DECISIONS or not entry["reviewer_id"].startswith("human_")
                or not entry["reviewed_at_utc"] or not entry.get("review_notes", "").strip()):
            raise ValueError(f"Human temporal audit cannot be recovered: {sid}")
        _, query, _ = _current(root, sid)
        if entry.get("query_version") != query.query_version:
            raise ValueError(f"Recovery query version mismatch: {sid}")
        if decision != "EXCLUDE":
            lock, pred = current_lock(root, sid), current_prediction(root, sid)
            if (not lock or not pred or entry.get("query_sha256") != lock["query_sha256"]
                    or entry["algorithm_sha256"] != deployment.ALGORITHM_SHA256):
                raise ValueError(f"Recovery prediction provenance mismatch: {sid}")
            video = deployment._paths(root)[2] / row["video_filename"]
            if (_file_sha(video) != pred["video_sha256"]
                    or [float(entry["ai_start"]), float(entry["ai_end"])] != [
                        pred["pred_start_seconds"], pred["pred_end_seconds"]]):
                raise ValueError(f"Recovery AI candidate mismatch: {sid}")
            start, end = float(entry["human_start"]), float(entry["human_end"])
            duration = float(row["duration_seconds"])
            if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end <= duration):
                raise ValueError(f"Recovery interval invalid: {sid}")
        if row["review_status"] == "DRAFT":
            row["review_status"] = expected_status
            row["annotator_id"] = entry["reviewer_id"]
            provenance = {"ACCEPT_AI": "AI_ACCEPTED", "EDIT_AI": "AI_EDITED",
                          "REJECT_AI": "AI_REJECTED", "EXCLUDE": "EXCLUDED"}[decision]
            row["annotation_notes"] = provenance + "; " + entry["review_notes"]
            if decision == "EXCLUDE":
                row["gt_start_seconds"] = row["gt_end_seconds"] = ""
            else:
                row["query"] = query.human_final_query
                row["gt_start_seconds"], row["gt_end_seconds"] = str(start), str(end)
            primary_changed = True
            recovered += 1
        elif row["review_status"] != expected_status:
            raise ValueError(f"Recovery primary status conflicts with human audit: {sid}")
        if sid in pre and pre[sid]["human_review_status"] != decision:
            pre[sid]["human_review_status"] = decision
            pre_changed = True
    if primary_changed:
        deployment._atomic_csv(ann / "human_primary.csv", list(rows[0]), rows)
    if pre_changed:
        deployment._atomic_csv(pre_path, deployment.PREANNOTATION_FIELDS,
                               [pre[s] for s in sorted(pre)])
    if recovered:
        from data_process.annotation.fastlane_freeze import freeze_if_complete
        freeze_if_complete(root)
    return recovered


def progress(root: Path) -> dict:
    qstate = query_progress(root)
    primary = deployment._unique(deployment._read_csv(review_paths(root)[0]), "human_primary.csv")
    cohort = {item["sample_id"] for item in qstate["items"]}
    statuses = {}
    approved = preannotated = reviewed = excluded = 0
    for sid in sorted(cohort):
        row = primary[sid]
        if row["review_status"] == "REVIEWED":
            statuses[sid] = "TEMPORAL_REVIEWED"
            reviewed += 1
        elif row["review_status"] == "EXCLUDED":
            statuses[sid] = "EXCLUDED"
            excluded += 1
        else:
            rec = load_query_reviews(review_paths(root)[2]).get(sid)
            if rec and rec.query_review_status in {"HUMAN_EDITED", "VALID_AS_IS"} and decision_of(rec):
                approved += 1
                lock = current_lock(root, sid)
                if lock is None:
                    statuses[sid] = "QUERY_APPROVED"
                elif current_prediction(root, sid) is None:
                    statuses[sid] = "QUERY_LOCKED"
                else:
                    statuses[sid] = "AI_PREANNOTATED"
                    preannotated += 1
            elif rec and decision_of(rec) == "EXCLUDE":
                statuses[sid] = "EXCLUSION_PENDING_COMMIT"
            else:
                statuses[sid] = "DRAFT"
    # Include completed cohort members in cumulative counts.
    queries = load_query_reviews(review_paths(root)[2])
    approved = sum(sid in queries and decision_of(queries[sid]) in {
        "ACCEPT_SUGGESTION", "EDIT_QUERY", "KEEP_ORIGINAL"} for sid in cohort)
    preannotated = sum(_manifest_path(root, sid, queries[sid].query_version, "preannotation").is_file()
                       for sid in cohort if sid in queries and decision_of(queries[sid]) in {
                           "ACCEPT_SUGGESTION", "EDIT_QUERY", "KEEP_ORIGINAL"})
    audit = deployment._read_csv(deployment._paths(root)[0] / "assisted_annotation_audit.csv")
    decisions = {d: sum(r["sample_id"] in cohort and r["temporal_decision"] == d for r in audit)
                 for d in deployment.DECISIONS}
    return {
        "total_pilot": len(primary),
        "reviewed_before_fastlane": sum(sid not in cohort and row["review_status"] == "REVIEWED"
                                        for sid, row in primary.items()),
        "fastlane_total": len(cohort), "query_approved": approved,
        "preannotated": preannotated, "temporal_reviewed": reviewed,
        "excluded": excluded, "remaining": sum(primary[sid]["review_status"] == "DRAFT" for sid in cohort),
        "statuses": statuses, "query_decisions": qstate["counts"],
        "temporal_decisions": decisions,
    }
