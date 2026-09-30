"""Stage A.2.6 pilot deployment. AI output is advisory until a human decision."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

from data_process.adsqa.semantic_preannotator_v3 import (
    SemanticCandidateEvent, decompose_query, rank_and_select_candidates,
)

ROOT = Path(__file__).resolve().parents[2]
ALGORITHM_VERSION = "semantic_v3_multi_candidate_ranker"
ALGORITHM_SHA256 = "cc7097db06a1376819a193b09054c135f8e5fadcef67ba58a82c34b24ba5d5fd"
WORKLIST_FIELDS = ("sample_id", "video_filename", "source_video_id", "original_query", "duration_seconds", "query_review_status", "query_reviewer")
SUGGESTION_FIELDS = (
    "sample_id", "video_filename", "original_query", "original_query_quality",
    "ai_suggested_query", "query_change_type", "query_localizable",
    "query_observable", "visual_evidence", "inspection_method",
    "generated_at_utc", "status", "sampled_times_seconds",
    # Existing A.2.6 consumers still read these aliases.
    "query_quality_suggestion", "observable_evidence",
)
PREANNOTATION_FIELDS = ("sample_id", "video_filename", "verified_query", "candidate_windows_json", "selected_candidate_window", "selected_candidate_rank", "candidate_count", "semantic_confidence", "ranking_margin", "semantic_reason", "algorithm_version", "algorithm_sha256", "query_version", "generated_at_utc", "human_review_status")
AUDIT_FIELDS = ("sample_id", "query_status", "query_reviewer", "temporal_decision", "ai_start", "ai_end", "human_start", "human_end", "start_delta", "end_delta", "reviewer_id", "reviewed_at_utc", "algorithm_version", "algorithm_sha256", "review_notes", "query_version", "query_sha256")
DECISIONS = {"ACCEPT_AI", "EDIT_AI", "REJECT_AI", "EXCLUDE"}


def _paths(root: Path) -> tuple[Path, Path, Path]:
    return (root / "local_data/annotations/lumae_ads", root / "local_data/reports/lumae_ads", root / "local_data/raw/adsqa/videos")


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _atomic_csv(path: Path, fields: tuple[str, ...] | list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=path.parent, delete=False) as stream:
            tmp = Path(stream.name)
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        if tmp is not None and tmp.exists():
            tmp.unlink()


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
            tmp = Path(stream.name)
            json.dump(payload, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        if tmp is not None and tmp.exists():
            tmp.unlink()


def _unique(rows: list[dict[str, str]], label: str) -> dict[str, dict[str, str]]:
    result = {}
    for row in rows:
        sid = row.get("sample_id", "").strip()
        if not sid or sid in result:
            raise ValueError(f"Missing or duplicate sample_id in {label}: {sid!r}")
        result[sid] = row
    return result


def _true(value: object) -> bool:
    return value is True or str(value).strip().upper() == "TRUE"


def approved_query(row: dict[str, str] | None) -> bool:
    """The only gate for deployment temporal inference."""
    return bool(row and row.get("query_review_status") in {"VALID_AS_IS", "HUMAN_EDITED"}
                and row.get("human_final_query", "").strip()
                and _true(row.get("query_localizable", ""))
                and _true(row.get("query_observable", ""))
                and row.get("reviewer_id", "").startswith("human_"))


def load_state(root: Path = ROOT) -> dict:
    ann, _, videos = _paths(root)
    primary = _unique(_read_csv(ann / "human_primary.csv"), "human_primary.csv")
    queries = _unique(_read_csv(ann / "query_reviews.csv"), "query_reviews.csv")
    sanitation = _unique(_read_csv(ann / "query_sanitation_worklist.csv"), "query_sanitation_worklist.csv")
    registry_path = root / "local_data/manifests/temporal_exposure_registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8")) if registry_path.is_file() else {}
    if registry and registry.get("total_pilot_samples") != len(primary):
        raise ValueError("Exposure registry and human_primary.csv pilot totals differ")
    if not set(sanitation).issubset(primary) or not set(queries).issubset(primary):
        raise ValueError("Query records contain a sample outside the pilot")
    completed = {sid for sid, r in primary.items() if r["review_status"] == "REVIEWED"}
    excluded = {sid for sid, r in primary.items() if r["review_status"] == "EXCLUDED"}
    pending = set(primary) - completed - excluded
    if any(primary[sid]["review_status"] != "DRAFT" for sid in pending):
        raise ValueError("Unknown pilot review status")
    eligible = {sid for sid in pending if (videos / primary[sid]["video_filename"]).is_file()}
    approved = {sid for sid in pending if approved_query(queries.get(sid))}
    invalid = {sid for sid in pending if queries.get(sid, {}).get("query_review_status") in {"INVALID_VIDEO", "EXCLUDE"}}
    return dict(primary=primary, queries=queries, sanitation=sanitation, registry=registry,
                completed=completed, excluded=excluded, pending=pending, eligible=eligible,
                approved=approved, invalid=invalid, missing_video=pending - eligible)


def deployment_worklist(root: Path = ROOT) -> list[dict[str, str]]:
    state = load_state(root)
    rows = []
    for sid in sorted(state["eligible"] - state["invalid"]):
        p = state["primary"][sid]
        q = state["queries"].get(sid, {})
        rows.append(dict(sample_id=sid, video_filename=p["video_filename"],
                         source_video_id=p["source_video_id"],
                         original_query=q.get("original_query") or p["query"],
                         duration_seconds=p["duration_seconds"],
                         query_review_status=q.get("query_review_status", "PENDING_QUERY_REVIEW"),
                         query_reviewer=q.get("reviewer_id", "")))
    return rows


def save_ai_suggestions(suggestions: list[dict[str, str]], root: Path = ROOT) -> None:
    """Write video-inspection advice without modifying human query approvals."""
    state = load_state(root)
    seen = _unique(suggestions, "AI suggestions")
    if not set(seen).issubset(state["eligible"] - state["invalid"]):
        raise ValueError("AI suggestion includes a completed or unavailable sample")
    for sid, row in seen.items():
        if row.get("video_filename") != state["primary"][sid]["video_filename"]:
            raise ValueError(f"Suggestion video mismatch: {sid}")
        if not (row.get("ai_suggested_query", "").strip() and row.get("observable_evidence", "").strip()
                and row.get("inspection_method", "").strip() and row.get("sampled_times_seconds", "").strip()):
            raise ValueError(f"AI suggestion lacks video inspection evidence: {sid}")
        if row.get("query_quality_suggestion") not in {"VALID", "VAGUE_BUT_RELEVANT", "INCORRECT_FOR_VIDEO", "INVALID_OR_UNLOCALIZABLE"}:
            raise ValueError(f"Invalid quality suggestion: {sid}")
    path = _paths(root)[0] / "ai_query_suggestions.csv"
    _atomic_csv(path, SUGGESTION_FIELDS, [seen[s] for s in sorted(seen)])


def verify_frozen_ranker(root: Path = ROOT) -> None:
    manifest = json.loads((root / "local_data/manifests/semantic_v3_algorithm_freeze.json").read_text(encoding="utf-8"))
    if manifest.get("version") != ALGORITHM_VERSION or manifest.get("combined_source_sha256") != ALGORITHM_SHA256:
        raise ValueError("Frozen semantic_v3 manifest mismatch")
    for name, digest in manifest["source_file_hashes"].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Frozen semantic_v3 source changed: {name}")


def frozen_artifact_errors(root: Path = ROOT) -> list[str]:
    """Verify historical freeze references without writing any historical artifact."""
    manifest_dir = root / "local_data/manifests"
    query_manifest_path = manifest_dir / "semantic_v3_clean_query_blind5_query_freeze.json"
    if not query_manifest_path.is_file():
        return []  # Small isolated test pilots have no historical experiment.
    query_manifest = json.loads(query_manifest_path.read_text(encoding="utf-8"))
    prediction_manifest = json.loads((manifest_dir / "semantic_v3_clean_query_blind5_prediction_freeze.json").read_text(encoding="utf-8"))
    gt_manifest = json.loads((manifest_dir / "semantic_v3_clean_query_blind5_human_gt_freeze.json").read_text(encoding="utf-8"))
    checks = (
        (manifest_dir / "semantic_v3_clean_query_blind5_selection.json", query_manifest["selection_manifest_sha256"]),
        (root / query_manifest["query_file"], query_manifest["query_file_sha256"]),
        (root / prediction_manifest["prediction_file"], prediction_manifest["prediction_sha256"]),
        (root / gt_manifest["human_gt_file"], gt_manifest["human_gt_sha256"]),
    )
    errors = [f"Frozen artifact hash mismatch: {path.relative_to(root).as_posix()}"
              for path, expected in checks
              if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected]
    try:
        verify_frozen_ranker(root)
    except (ValueError, FileNotFoundError) as exc:
        errors.append(str(exc))
    return errors


def _candidate_pool(root: Path, sid: str, duration: float) -> list[SemanticCandidateEvent]:
    """Require separately inspected video candidates; never use the ranker's generic fallback."""
    pool_path = _paths(root)[0] / "semantic_v3_deployment_candidate_pools.json"
    pools = json.loads(pool_path.read_text(encoding="utf-8")) if pool_path.is_file() else {}
    items = pools.get(sid, [])
    if not items:
        raise ValueError(f"No video-inspected candidate pool for {sid}")
    candidates = []
    for item in items:
        if not item.get("supporting_visual_evidence", "").strip():
            raise ValueError(f"Candidate has no visual evidence: {sid}")
        candidate = SemanticCandidateEvent(**item)
        start, end = candidate.get_effective_window()
        if not (0 <= start < end <= duration):
            raise ValueError(f"Invalid candidate window for {sid}: {start}, {end}")
        candidates.append(candidate)
    return candidates


def precompute(root: Path = ROOT) -> dict[str, int]:
    """Run the frozen ranking function only after query approval and video inspection."""
    verify_frozen_ranker(root)
    state = load_state(root)
    ann, _, _ = _paths(root)
    target = ann / "semantic_v3_deployment_preannotations.csv"
    existing = _unique(_read_csv(target), target.name)
    fastlane_ids = {r["sample_id"] for r in _read_csv(ann / "ai_query_suggestions.csv")}
    generated = 0
    no_pool = 0
    for sid in sorted(state["eligible"] & state["approved"]):
        q = state["queries"][sid]
        p = state["primary"][sid]
        query = q["human_final_query"].strip()
        if sid.startswith("lumae_ads_pilot_") and sid in fastlane_ids:
            raise ValueError(f"Fast-lane sample requires per-sample video-derived inference: {sid}")
        old = existing.get(sid)
        if old and old["verified_query"] == query and old["query_version"] == q["query_version"]:
            continue
        try:
            candidates = _candidate_pool(root, sid, float(p["duration_seconds"]))
        except ValueError as exc:
            if str(exc).startswith("No video-inspected candidate pool"):
                no_pool += 1
                continue
            raise
        top, margin, confidence, reason = rank_and_select_candidates(decompose_query(query), candidates)
        existing[sid] = dict(sample_id=sid, video_filename=p["video_filename"], verified_query=query,
                             candidate_windows_json=json.dumps([c.get_effective_window() for c in candidates]),
                             selected_candidate_window=json.dumps(top.get_effective_window()),
                             selected_candidate_rank="1", candidate_count=str(len(candidates)),
                             semantic_confidence=confidence, ranking_margin=f"{margin:.4f}",
                             semantic_reason=reason, algorithm_version=ALGORITHM_VERSION,
                             algorithm_sha256=ALGORITHM_SHA256, query_version=q["query_version"],
                             generated_at_utc=datetime.now(timezone.utc).isoformat(),
                             human_review_status="PENDING_TEMPORAL_REVIEW")
        generated += 1
    _atomic_csv(target, PREANNOTATION_FIELDS, [existing[s] for s in sorted(existing)])
    return {"approved_eligible": len(state["eligible"] & state["approved"]),
            "generated": generated, "missing_candidate_pool": no_pool,
            "blocked_query_review": len(state["eligible"] - state["approved"] - state["invalid"])}


def submit_temporal_decision(sample_id: str, decision: str, reviewer_id: str,
                             start: float | None = None, end: float | None = None,
                             notes: str = "", root: Path = ROOT) -> None:
    """Explicit human decision; AI output never writes primary GT on its own."""
    if decision not in DECISIONS or not reviewer_id.startswith("human_"):
        raise ValueError("A valid human reviewer and decision are required")
    state = load_state(root)
    if sample_id not in state["pending"]:
        raise ValueError("Sample is not pending temporal review")
    ann, _, videos = _paths(root)
    p = state["primary"][sample_id]
    if not (videos / p["video_filename"]).is_file():
        raise ValueError("Source video is missing")
    q = state["queries"].get(sample_id)
    if decision != "EXCLUDE" and not approved_query(q):
        raise ValueError("Human query approval gate is not satisfied")
    fastlane_prediction = None
    if decision != "EXCLUDE" and sample_id.startswith("lumae_ads_pilot_") and sample_id in {
        r["sample_id"] for r in _read_csv(ann / "ai_query_suggestions.csv")
    }:
        from data_process.annotation.fastlane import current_lock, current_prediction
        lock, fastlane_prediction = current_lock(root, sample_id), current_prediction(root, sample_id)
        if lock is None or fastlane_prediction is None or fastlane_prediction["query_sha256"] != lock["query_sha256"]:
            raise ValueError("Current per-sample query lock and prediction are required")
    pre_path = ann / "semantic_v3_deployment_preannotations.csv"
    pre = _unique(_read_csv(pre_path), pre_path.name)
    ai = pre.get(sample_id)
    ai_start = ai_end = None
    if decision != "EXCLUDE":
        if not ai or ai["human_review_status"] != "PENDING_TEMPORAL_REVIEW":
            raise ValueError("A pending semantic_v3 preannotation is required")
        if ai["verified_query"] != q["human_final_query"].strip() or ai["query_version"] != q["query_version"]:
            raise ValueError("Preannotation query does not match the approved query")
        if ai["algorithm_sha256"] != ALGORITHM_SHA256:
            raise ValueError("Preannotation algorithm hash mismatch")
        if fastlane_prediction is not None and json.loads(ai["selected_candidate_window"]) != [
            fastlane_prediction["pred_start_seconds"], fastlane_prediction["pred_end_seconds"]
        ]:
            raise ValueError("Deployment interval differs from immutable fast-lane prediction")
        ai_start, ai_end = map(float, json.loads(ai["selected_candidate_window"]))
        if decision == "ACCEPT_AI":
            if start is not None or end is not None:
                raise ValueError("ACCEPT_AI must use the unchanged AI interval")
            start, end = ai_start, ai_end
        else:
            if start is None or end is None:
                raise ValueError("Human-supplied boundaries are required")
            if (float(start), float(end)) == (ai_start, ai_end):
                raise ValueError(f"{decision} requires a human-supplied interval different from the AI interval")
    elif not (q and q.get("query_review_status") in {"EXCLUDE", "INVALID_VIDEO"}) and not notes.strip():
        raise ValueError("EXCLUDE requires a human reason")
    if decision != "EXCLUDE":
        duration = float(p["duration_seconds"])
        if not (math.isfinite(float(start)) and math.isfinite(float(end)) and 0 <= float(start) < float(end) <= duration):
            raise ValueError("Temporal interval must satisfy 0 <= start < end <= duration")
    primary_path = ann / "human_primary.csv"
    primary_rows = _read_csv(primary_path)
    row = next(r for r in primary_rows if r["sample_id"] == sample_id)
    row["annotator_id"] = reviewer_id
    provenance = {"ACCEPT_AI": "AI_ACCEPTED", "EDIT_AI": "AI_EDITED", "REJECT_AI": "AI_REJECTED", "EXCLUDE": "EXCLUDED"}[decision]
    row["annotation_notes"] = provenance + (f"; {notes.strip()}" if notes.strip() else "")
    row["review_status"] = "EXCLUDED" if decision == "EXCLUDE" else "REVIEWED"
    row["query"] = q["human_final_query"].strip() if decision != "EXCLUDE" else row["query"]
    row["gt_start_seconds"] = "" if decision == "EXCLUDE" else str(float(start))
    row["gt_end_seconds"] = "" if decision == "EXCLUDE" else str(float(end))
    audit_path = ann / "assisted_annotation_audit.csv"
    audit = _read_csv(audit_path)
    if any(r["sample_id"] == sample_id for r in audit):
        raise ValueError("Duplicate assisted annotation audit entry")
    audit.append(dict(sample_id=sample_id, query_status=q.get("query_review_status", "") if q else "",
                      query_reviewer=q.get("reviewer_id", "") if q else "",
                      temporal_decision=decision, ai_start="" if ai_start is None else str(ai_start),
                      ai_end="" if ai_end is None else str(ai_end),
                      human_start="" if start is None or decision == "EXCLUDE" else str(float(start)),
                      human_end="" if end is None or decision == "EXCLUDE" else str(float(end)),
                      start_delta="" if ai_start is None or decision == "EXCLUDE" else str(round(float(start)-ai_start, 4)),
                      end_delta="" if ai_end is None or decision == "EXCLUDE" else str(round(float(end)-ai_end, 4)),
                      reviewer_id=reviewer_id, reviewed_at_utc=datetime.now(timezone.utc).isoformat(),
                      algorithm_version=ALGORITHM_VERSION if ai else "",
                      algorithm_sha256=ALGORITHM_SHA256 if ai else "",
                      review_notes=notes.strip(), query_version=q.get("query_version", "") if q else "",
                      query_sha256=hashlib.sha256(q["human_final_query"].encode("utf-8")).hexdigest()
                      if q and q.get("human_final_query") else ""))
    if ai:
        ai["human_review_status"] = decision
    # Each file is replaced atomically. The audit is written first, so a failed primary
    # write is visible as an incomplete transaction and cannot silently create GT.
    _atomic_csv(audit_path, AUDIT_FIELDS, audit)
    _atomic_csv(primary_path, list(primary_rows[0]), primary_rows)
    _atomic_csv(pre_path, PREANNOTATION_FIELDS, [pre[s] for s in sorted(pre)])


def readiness(root: Path = ROOT) -> dict:
    state = load_state(root)
    ann, _, videos = _paths(root)
    audit = _read_csv(ann / "assisted_annotation_audit.csv")
    errors = frozen_artifact_errors(root)
    if len(audit) != len({r["sample_id"] for r in audit}):
        errors.append("Duplicate assisted audit sample_id")
    for sid in sorted(state["completed"]):
        p = state["primary"][sid]
        try:
            start, end, duration = float(p["gt_start_seconds"]), float(p["gt_end_seconds"]), float(p["duration_seconds"])
            if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end <= duration):
                errors.append(f"Invalid temporal interval: {sid}")
        except (ValueError, TypeError):
            errors.append(f"Missing temporal interval: {sid}")
        if not p["query"].strip() or not p["annotator_id"].startswith("human_"):
            errors.append(f"Missing human query/reviewer: {sid}")
        if not p["source_video_id"].strip() or p["source_video_id"] != Path(p["video_filename"]).stem:
            errors.append(f"Invalid source video ID: {sid}")
        if not (videos / p["video_filename"]).is_file():
            errors.append(f"Missing video: {sid}")
    preannotations = _unique(_read_csv(ann / "semantic_v3_deployment_preannotations.csv"), "semantic_v3_deployment_preannotations.csv")
    for row in audit:
        sid = row["sample_id"]
        if row["temporal_decision"] != "EXCLUDE" and not approved_query(state["queries"].get(sid)):
            errors.append(f"Assisted GT lacks verified query: {sid}")
        if not row["reviewer_id"].startswith("human_"):
            errors.append(f"Assisted GT lacks human reviewer: {sid}")
        if sid not in state["completed"] | state["excluded"]:
            errors.append(f"Audit decision not committed to primary: {sid}")
        if row["temporal_decision"] != "EXCLUDE":
            if state["primary"].get(sid, {}).get("query") != state["queries"].get(sid, {}).get("human_final_query"):
                errors.append(f"Assisted GT query differs from approved query: {sid}")
            if preannotations.get(sid, {}).get("human_review_status") != row["temporal_decision"]:
                errors.append(f"Preannotation review status differs from audit: {sid}")
    decisions = {d: sum(r["temporal_decision"] == d for r in audit) for d in DECISIONS}
    decided = sum(decisions[d] for d in ("ACCEPT_AI", "EDIT_AI", "REJECT_AI"))
    corrections = [r for r in audit if r["temporal_decision"] in {"EDIT_AI", "REJECT_AI"}]
    start_mean = sum(abs(float(r["start_delta"])) for r in corrections)/len(corrections) if corrections else None
    end_mean = sum(abs(float(r["end_delta"])) for r in corrections)/len(corrections) if corrections else None
    durations = [float(r["duration_seconds"]) for r in state["primary"].values()]
    qualities = {}
    for q in state["queries"].values():
        quality = q.get("original_query_quality", "") or "UNREVIEWED"
        qualities[quality] = qualities.get(quality, 0) + 1
    missing_historical_queries = len(state["completed"] - set(state["queries"]))
    if missing_historical_queries:
        qualities["HISTORICAL_HUMAN_REVIEW_NO_QUERY_REVIEW_ROW"] = missing_historical_queries
    return dict(total=len(state["primary"]), reviewed=len(state["completed"]), excluded=len(state["excluded"]),
                pending=len(state["pending"]), query_reviewed_only=len(state["approved"]),
                pending_query_review=len(state["pending"] - state["approved"] - state["invalid"]),
                pending_temporal_review=len(state["pending"]), invalid=len(state["invalid"]),
                unique_videos=len({r["video_filename"] for r in state["primary"].values()}),
                unique_queries=len({r["query"].strip() for r in state["primary"].values() if r["query"].strip()}),
                duration_stats={"min": min(durations), "max": max(durations), "mean": sum(durations)/len(durations)} if durations else {},
                query_quality_stats=qualities,
                ai_acceptance_stats={**decisions, "acceptance_rate": decisions["ACCEPT_AI"]/decided if decided else None,
                                     "edit_rate": decisions["EDIT_AI"]/decided if decided else None,
                                     "rejection_rate": decisions["REJECT_AI"]/decided if decided else None,
                                     "mean_absolute_start_correction": start_mean,
                                     "mean_absolute_end_correction": end_mean,
                                     "mean_boundary_correction": (start_mean+end_mean)/2 if start_mean is not None else None},
                validation_errors=errors,
                READY_FOR_DATASET_FREEZE="YES" if not state["pending"] and not errors else "NO")


def prepare(root: Path = ROOT) -> dict:
    state = load_state(root)
    ann, reports, _ = _paths(root)
    worklist = deployment_worklist(root)
    _atomic_csv(ann / "semantic_v3_deployment_worklist.csv", WORKLIST_FIELDS, worklist)
    for name, fields in (("semantic_v3_deployment_preannotations.csv", PREANNOTATION_FIELDS),
                         ("assisted_annotation_audit.csv", AUDIT_FIELDS),
                         ("ai_query_suggestions.csv", SUGGESTION_FIELDS)):
        path = ann / name
        if not path.exists():
            _atomic_csv(path, fields, [])
    result = readiness(root)
    _atomic_json(reports / "pilot_dataset_readiness.json", result)
    return {"eligible": len(worklist), "already_completed": len(state["completed"]),
            "excluded": len(state["excluded"]), "missing_video": len(state["missing_video"]),
            "readiness": result}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "precompute", "readiness"))
    args = parser.parse_args()
    result = {"prepare": prepare, "precompute": precompute, "readiness": readiness}[args.action]()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
