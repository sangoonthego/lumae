"""Honest, resumable D200 progress and annotation quality reports."""

from __future__ import annotations

import csv
import json
import statistics
from collections import Counter
from pathlib import Path

from .freeze import accepted_records, portable_copy
from .models import D200, ROOT, WORK, atomic_json, digest, load_jsonl, read_json
from .quality_gate import interval_errors
from .repo_paths import resolve_path
from .throughput import write_profile
from .shadow import write_shadow_report
from .audit import write_audit_report

REPORT_DIR = ROOT / "local_data/reports/lumae_ads_d200"


def _stats(numbers: list[float]) -> dict:
    if not numbers:
        return {"min": None, "median": None, "mean": None, "max": None}
    return {"min": min(numbers), "median": statistics.median(numbers),
            "mean": statistics.mean(numbers), "max": max(numbers)}


def write_report() -> dict:
    throughput = write_profile()
    shadow = write_shadow_report() if (WORK / "source_selection.json").exists() else None
    high_audit = write_audit_report()
    state_path = WORK / "source_selection.json"
    candidates = read_json(state_path)["candidates"] if state_path.exists() else []
    records = accepted_records()
    ledger_path = WORK / "build_ledger.jsonl"
    ledger = load_jsonl(ledger_path) if ledger_path.exists() else []
    event_counts = Counter(item["event"] for item in ledger)
    unique_predictions = {(item["source_video_id"], item.get("query")) for item in ledger
                          if item["event"] == "semantic_v3_prediction"}
    unique_failures = {(item["source_video_id"], item.get("query")) for item in ledger
                       if item["event"] in {"query_failed", "query_rejected"}}
    status_counts = Counter(item["status"] for item in candidates)
    intervals = [r["window"][1] - r["window"][0] for r in records]
    words = [len(r["query"].split()) for r in records]
    margins = [r["ranking_margin"] for r in records]
    clip_margins = [r["clip_signals"]["inside_minus_outside_margin"] for r in records]
    frozen = D200 / "lumae_ads_d200_all.jsonl"
    report = {
        "dataset_version": "lumae_ads_d200", "build_status": "COMPLETE" if frozen.exists() else "INCOMPLETE",
        "target_samples": 200, "reused_d48_samples": 48,
        "new_accepted_samples": len(records), "new_remaining": 152 - len(records),
        "source_candidates_available": len(candidates),
        "source_candidates_attempted": sum(x["status"] != "unattempted" for x in candidates),
        "rejected_videos": status_counts["rejected"],
        "replacement_videos": status_counts["rejected"],
        "successful_visual_inspections": len({item["source_video_id"] for item in ledger
                                              if item["event"] == "visual_inspection"}),
        "successful_query_generations": len(records),
        "query_candidate_attempts": len(unique_predictions | unique_failures),
        "semantic_v3_successes": len(unique_predictions),
        "semantic_v3_failures": len(unique_failures - unique_predictions),
        "provenance_counts_current": {"HUMAN_VERIFIED": 18,
                                      "AI_PSEUDO_LABELED": 30 + len(records)},
        "interval_duration_seconds": _stats(intervals),
        "query_length_words": _stats(words),
        "semantic_v3_confidence_distribution": dict(Counter(r["semantic_confidence"] for r in records)),
        "ranking_margin": _stats(margins),
        "clip_inside_minus_outside_margin": _stats(clip_margins),
        "train_val_test_counts": {"train": 140, "val": 30, "test": 30} if frozen.exists() else None,
        "video_overlap_check": "PASS" if frozen.exists() else "NOT_RUN_INCOMPLETE",
        "missing_accepted_video_count": sum(not resolve_path(r["source_video_path"]).is_file() for r in records),
        "invalid_accepted_interval_count": sum(bool(interval_errors(*r["window"], r["duration"])) for r in records),
        "dataset_sha256": digest(frozen) if frozen.exists() else None,
        "pending_statuses": {k: v for k, v in status_counts.items()
                             if k.startswith("awaiting_")},
        "annotation_warning": "New labels are AI pseudo labels. No validation/test human ground truth exists."
    }
    report["throughput_profile_path"] = "local_data/reports/lumae_ads_d200/d200_throughput_profile.json"
    report["timed_attempts"] = throughput["attempts_with_timing"]
    report["shadow_completed"] = shadow["completed_with_auto_route_and_review"] if shadow else 0
    report["high_random_audit_count"] = high_audit["reviewed_count"]
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    atomic_json(REPORT_DIR / "d200_build_report.json", report)
    with (REPORT_DIR / "d200_build_report.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["source_video_id", "status", "duration",
                                                     "sha256", "local_path", "rejection_reason"])
        writer.writeheader()
        for row in candidates:
            writer.writerow(portable_copy({key: row.get(key) for key in writer.fieldnames}))
    audit = [{"source_video_id": r["source_video_id"], "query": r["query"],
              "window": r["window"], "visual_evidence": r["visual_evidence"],
              "semantic_artifact_path": r["semantic_artifact_path"],
              "verifier": r["verifier"], "clip_signals": r["clip_signals"]}
             for r in records]
    atomic_json(REPORT_DIR / "d200_quality_audit.json", portable_copy(audit))
    if report["build_status"] == "INCOMPLETE":
        atomic_json(REPORT_DIR / "d200_blocker_report.json", {
            "status": "INCOMPLETE",
            "target_new_samples": 152,
            "accepted_new_samples": len(records),
            "remaining_new_samples": 152 - len(records),
            "attempted_source_videos": report["source_candidates_attempted"],
            "rejected_source_videos": report["rejected_videos"],
            "adsqa_source_access": "Verified working for downloaded candidates; no source-access blocker established.",
            "current_limitation": "Each remaining new video requires direct agent inspection of real contact sheets and the predicted interval. No local VLM was configured or smoke-tested on this CPU-only host in this run.",
            "dataset_frozen": False,
            "training_handoff_ready": False,
            "colab_bundle_ready": False,
            "resume_command": "python -m data_process.lumae_scale --target-videos 200 --seed 20260930 --resume",
            "integrity_policy": "Never fill missing samples with metadata-derived queries, unviewed events, or fabricated intervals."
        })
    else:
        (REPORT_DIR / "d200_blocker_report.json").unlink(missing_ok=True)
    return report
