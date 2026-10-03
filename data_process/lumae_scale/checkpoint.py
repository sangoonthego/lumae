"""Write a reproducible integrity snapshot at a D500 annotation milestone."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone

from .freeze import accepted_records, validate_annotation
from .incremental import ProtectedInputs, verify_parent
from .layout import BuildLayout
from .models import atomic_json, digest, load_jsonl, read_json
from .telemetry import aggregate


def create_checkpoint(expected_new: int, layout: BuildLayout | None = None) -> dict:
    if expected_new not in (100, 200, 300):
        raise ValueError("Checkpoint must be at 100, 200, or 300 new samples")
    layout = layout or BuildLayout()
    records = accepted_records(layout.accepted)
    if len(records) != expected_new:
        raise ValueError(f"Expected {expected_new} accepted samples; found {len(records)}")

    parent = verify_parent(layout)
    protected = ProtectedInputs(layout).verify(full=True)
    base = load_jsonl(layout.base / f"{layout.base_version}_all.jsonl")
    parent_ids = {row["vid"].removeprefix("lumae_ads_") for row in base}
    source_ids = [row["source_video_id"] for row in records]
    queries = [row["query"].strip().casefold() for row in base + records]
    hashes = [row["source_video_sha256"] for row in records]
    hashes += [item["sha256"] for item in parent["source_video_sha256"].values()]
    if len(set(source_ids)) != expected_new or set(source_ids) & parent_ids:
        raise ValueError("New source IDs are not unique and disjoint from D200")
    if len(set(queries)) != 200 + expected_new:
        raise ValueError("Duplicate query in combined dataset")
    if len(set(hashes)) != 200 + expected_new:
        raise ValueError("Duplicate source content in combined dataset")
    for record in records:
        validate_annotation(record, visual_cache=layout.visual_cache)

    state = read_json(layout.selection)
    states = Counter(row["status"] for row in state["candidates"])
    telemetry = aggregate(layout.work / "telemetry")
    if telemetry["accepted_attempts"] != expected_new:
        raise ValueError("Telemetry accepted count differs from accepted records")
    attempts = [read_json(layout.work / "attempts" / (vid + ".json")) for vid in source_ids]
    long_idle = sorted(
        ({"source_video_id": row["source_video_id"], "seconds": row["total_ms"] / 1000}
         for row in attempts if row["total_ms"] >= 600000),
        key=lambda item: item["seconds"], reverse=True,
    )
    report = {
        "checkpoint": f"D500_PHASE_A_{expected_new}_ACCEPTED",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "accepted_new_exact": expected_new,
        "base_frozen": 200,
        "current_valid_total": 200 + expected_new,
        "remaining_to_accept": 300 - expected_new,
        "source_states": dict(states),
        "rejected_by_failure_kind": dict(Counter(
            row.get("failure_kind") for row in state["candidates"] if row["status"] == "rejected"
        )),
        "integrity": {
            "protected_inputs": protected,
            "d200_parent_checksums": "PASS",
            "d200_val_sha256": digest(layout.base / f"{layout.base_version}_val.jsonl"),
            "d200_test_sha256": digest(layout.base / f"{layout.base_version}_test.jsonl"),
            "new_ids_disjoint_from_D200": True,
            "new_source_ids_unique": True,
            "all_content_hashes_unique": True,
            "all_queries_unique": True,
            "annotation_schema_valid": expected_new,
            "d500_frozen": (layout.dataset / f"{layout.version}_all.jsonl").is_file(),
        },
        "telemetry": telemetry,
        "long_idle_or_pause_attempts": long_idle,
        "timing_scope": "Active attempt wall includes human/agent review pauses; producer preparation is separate and overlapping.",
        "phase_b_external": "D200 feature cache, training features/checkpoints, and Colab recipe remain outside Phase A.",
    }
    path = layout.reports / f"checkpoint_{expected_new}.json"
    atomic_json(path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("expected_new", type=int, choices=(100, 200, 300))
    args = parser.parse_args()
    report = create_checkpoint(args.expected_new)
    print(f"PASS: {report['checkpoint']} ({report['integrity']['protected_inputs']['protected_files']} protected files)")


if __name__ == "__main__":
    main()
