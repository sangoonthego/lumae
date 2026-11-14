"""Resume from the immutable generation and review decisions on disk."""

from collections import Counter
from datetime import datetime, timezone

from data_process.lumae_scale.models import atomic_json
from .annotation import progress
from .parent import VERSION, duration_distribution
from .release import build_text_cache, quality, text_feature

RETRY_LIMIT = 3
RETRY_STATES = {"GENERATION_REJECTED", "REVIEW_REJECTED", "BOUNDARY_DISAGREEMENT", "DUPLICATE_QUERY", "RETIRED"}


def queue(states):
    pending = {}
    for vid, state in states.items():
        if state["status"] == "ACCEPTED_VISUAL_REVIEW":
            continue
        attempts = state["attempts"]
        submitted = [attempt for attempt in attempts if "generation_decisions" in attempt]
        current = attempts[-1] if attempts else {"attempt": 1, "status": "WAITING_GENERATION"}
        status = current["status"]
        attempt = current["attempt"]
        if status in RETRY_STATES:
            status = "RETRY_REQUIRED" if len(submitted) < RETRY_LIMIT else "BLOCKED"
            attempt += status == "RETRY_REQUIRED"
        if status == "WAITING_GENERATION":
            status = "RETRY_REQUIRED" if submitted else "PENDING"
        pending[vid] = {"status": status, "attempt": attempt,
                        "submitted_attempts": len(submitted), "last_outcome": current["status"]}
    return pending


def checkpoint(parent, store, agreement, batch_size, encode):
    rows, sidecar, states = store.accepted(agreement)
    pending = queue(states)
    summary = progress(rows, sidecar, states)
    summary.pop("per_video")
    existing, feature_errors = {}, {}
    extraction_error = None
    if encode:
        try:
            build_text_cache(store.work, rows)
        except (OSError, RuntimeError, ValueError) as error:
            extraction_error = str(error)
    for row in rows:
        try:
            feature = text_feature(store.work, row)
            existing[row["qid"]] = {"sha256": feature.sha256, "shape": list(feature.array().shape)}
        except (OSError, ValueError) as error:
            feature_errors[row["qid"]] = str(error)
    diversity_report = None
    if not feature_errors:
        _, diversity_report = quality(parent, store.work, rows, 0.94)
    released = (store.work / "release/manifest.json").is_file()
    if released:
        from .release import validate_release

        validate_release(parent, store)
    report = {
        "dataset_version": VERSION, "checkpoint_utc": datetime.now(timezone.utc).isoformat(),
        "status": "COMPLETE_AND_SEALED" if released else "INCOMPLETE_NOT_PUBLISHED",
        **summary,
        "parent_split_sha256": parent.hashes,
        "original_test_queries": len(parent.test),
        "new_queries_target": 2 * len(parent.test),
        "new_queries_remaining": 2 * len(parent.test) - len(rows),
        "completed_vids": sorted(set(parent.test) - set(pending)),
        "pending_vids": list(pending),
        "next_batch": [vid for vid, item in pending.items() if item["status"] != "BLOCKED"][:batch_size],
        "queue_counts": dict(Counter(item["status"] for item in pending.values())),
        "queue": pending, "retry_limit": RETRY_LIMIT,
        "validated_text_features": len(existing), "text_features": existing,
        "feature_errors": feature_errors, "extraction_error": extraction_error,
        "diversity": diversity_report,
        "duration_buckets": {
            **{split: duration_distribution(parent.rows[split]) for split in ("train", "val", "test")},
            "accepted_new_queries": duration_distribution(rows),
            "original_plus_accepted": duration_distribution(parent.rows["test"] + rows),
        },
        "per_video": states,
    }
    atomic_json(store.work / "progress/execution_state.json", report)
    return {key: value for key, value in report.items()
            if key not in {"per_video", "queue", "pending_vids", "completed_vids",
                           "text_features", "parent_split_sha256"}} | {
        "checkpoint_path": str(store.work / "progress/execution_state.json"),
    }
