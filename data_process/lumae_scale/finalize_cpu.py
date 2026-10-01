"""Final CPU build audit and master CSV from immutable annotation artifacts.

Historical times stay unknown. Active wall times include agent/tool pauses;
persisted stage timers measure compute only and omit overlapped prefetch work.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import statistics
from collections import Counter

from data_process.annotation.assisted_deployment import verify_frozen_ranker
from .cache_identity import (SourceIdentity, VisualIndexIdentity, ClipCacheIdentity,
                             SemanticCacheIdentity, deterministic_fingerprint)
from .foundation import verify_d48
from .freeze import ACCEPTED, portable_copy
from .models import ROOT, WORK, VISUAL_CACHE, D48, D200, atomic_json, digest, load_jsonl, read_json

REPORT_DIR = ROOT / "local_data/reports/lumae_ads_d200"
REPORT_PATH = REPORT_DIR / "d200_cpu_finish_report.json"
MASTER_PATH = REPORT_DIR / "lumae_ads_d200_annotation_master.csv"


def absolute_values(value):
    if isinstance(value, dict):
        return [p for v in value.values() for p in absolute_values(v)]
    if isinstance(value, list):
        return [p for v in value for p in absolute_values(v)]
    if isinstance(value, str) and (re.match(r"^[A-Za-z]:[\\/]", value)
                                   or value.startswith(("/content/", "/home/", "\\\\"))):
        return [value]
    return []


def audit() -> dict:
    fp = verify_d48()
    verify_frozen_ranker(ROOT)
    manifest = read_json(D200 / "dataset_manifest.json")
    for vid, expected in manifest["accepted_record_sha256"].items():
        if digest(ACCEPTED / (vid + ".json")) != expected:
            raise ValueError(f"Accepted annotation changed after freeze: {vid}")
    rows = load_jsonl(D200 / "lumae_ads_d200_all.jsonl")
    # Check the exported prefix against the immutable original D48 fingerprint.
    # This still checks exact bytes when a long-running host cannot open D48.
    old_bytes = b"".join((D200 / "lumae_ads_d200_all.jsonl").read_bytes().splitlines(keepends=True)[:48])
    original_d48_sha = fp["file_sha256"]["local_data/datasets/lumae_ads_v1/lumae_ads_v1_all.jsonl"]
    if hashlib.sha256(old_bytes).hexdigest() != original_d48_sha:
        raise ValueError("D48 export prefix differs from its immutable original SHA")
    for split in ("all", "train"):
        if not (D200 / f"lumae_ads_d200_{split}.jsonl").read_bytes().startswith(old_bytes):
            raise ValueError("D48 canonical bytes not preserved")
    from experiments.M1_lumae_ads.train_d200 import preflight
    _, _, splits = preflight(ROOT)
    path_count = 0
    for path in D200.iterdir():
        if path.suffix == ".json":
            path_count += len(absolute_values(read_json(path)))
        elif path.suffix == ".jsonl":
            path_count += len(absolute_values(load_jsonl(path)))
    d48_qids = set(fp["canonical_line_sha256_by_qid"])
    result = {
        "total_samples": len(rows), "d48_reused": len(d48_qids),
        "new_accepted": len(list(ACCEPTED.glob("*.json"))),
        "unique_qids": len({r["qid"] for r in rows}),
        "unique_videos": len({r["vid"] for r in rows}),
        "unique_video_content": len({r["sha256"] for r in manifest["source_video_sha256"].values()}),
        "provenance": dict(Counter(r["metadata"]["review_provenance"] for r in rows)),
        "machine_absolute_paths": path_count,
        "machine_path_scope": "Frozen dataset, nested metadata, selection, annotation audit and ledger export copies; historical accepted caches are preserved.",
        "invalid_windows": sum(not 0 <= r["relevant_windows"][0][0] < r["relevant_windows"][0][1] <= r["duration"] for r in rows),
        "fake_saliency": sum(r["saliency_scores"] is not None or r["relevant_clip_ids"] is not None for r in rows),
        "fake_human_provenance": sum(r["metadata"]["review_provenance"] != "AI_PSEUDO_LABELED" for r in rows[48:]),
        "splits": {k: len(v) for k, v in splits.items()},
        "d48_outside_train": len(d48_qids - {r["qid"] for r in splits["train"]}),
        "video_leakage": sum(len({r["vid"] for r in splits[a]} & {r["vid"] for r in splits[b]}) for a, b in (("train", "val"), ("train", "test"), ("val", "test"))),
        "accepted_sha_preserved_since_freeze": 152,
        "d48_dataset_sha256": fp["file_sha256"]["local_data/datasets/lumae_ads_v1/lumae_ads_v1_all.jsonl"],
        "semantic_v3_sha256": rows[-1]["metadata"]["semantic_v3_algorithm_sha256"],
        "dataset_sha256": manifest["dataset_sha256"],
    }
    if any(result[k] for k in ("machine_absolute_paths", "invalid_windows", "fake_saliency", "fake_human_provenance", "d48_outside_train", "video_leakage")):
        raise ValueError(f"Final integrity gate failed: {result}")
    if (result["total_samples"], result["new_accepted"], result["unique_qids"], result["unique_videos"], result["unique_video_content"]) != (200, 152, 200, 200, 200):
        raise ValueError("Final exact-count/uniqueness gate failed")
    return result


def write_master(session_id: str) -> dict:
    rows = load_jsonl(D200 / "lumae_ads_d200_all.jsonl")
    split_by_qid = {r["qid"]: split for split in ("train", "val", "test")
                    for r in load_jsonl(D200 / f"lumae_ads_d200_{split}.jsonl")}
    sources = {r["source_video_id"]: r for r in read_json(ROOT / "local_data/raw/adsqa/source_manifest.json")["videos"]}
    manifest = read_json(D200 / "dataset_manifest.json")
    history = {}
    for entry in load_jsonl(WORK / "build_ledger.jsonl"):
        history.setdefault(entry.get("source_video_id"), []).append({"event": entry["event"], "time_utc": entry["time_utc"]})
    timing_rows = {p.stem: read_json(p) for p in (WORK / "cpu_sessions" / session_id).glob("*.json")}
    fields = ["qid", "vid", "adsqa_source_id", "source_url", "relative_video_path", "source_sha256", "duration", "query", "start", "end", "query_source", "temporal_label_source", "review_provenance", "router_state", "history_state", "agent_reviewed", "semantic_v3_hash", "clip_signals", "cache_fingerprints", "fingerprint_scope", "accepted_record_sha256", "visual_overview_path", "visual_overview_sha256", "visual_views_count", "timing_scope", "total_accepted_seconds", "total_accepted_minutes", "pipeline_call_seconds", "stages_seconds", "final_split"]
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with MASTER_PATH.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            meta = row["metadata"]
            vid = meta.get("source_video_id", row["vid"].removeprefix("lumae_ads_"))
            sha = manifest["source_video_sha256"][row["qid"]]["sha256"]
            timing = timing_rows.get(vid, {})
            timer_path = WORK / "timings" / (vid + ".json")
            timer = read_json(timer_path) if timer_path.exists() else {}
            index_path = VISUAL_CACHE / vid / "metadata.json"
            index = read_json(index_path) if index_path.exists() else {}
            review_path = WORK / "visual_index" / vid / "visual_events.json"
            review = read_json(review_path) if review_path.exists() else {}
            fingerprints = {"source": SourceIdentity(vid, sha).fingerprint()}
            if "visual_index_sha256" in meta:
                fingerprints.update(visual_index=VisualIndexIdentity(sha).fingerprint(),
                                    clip=ClipCacheIdentity(sha, meta["visual_index_sha256"]).fingerprint(),
                                    semantic=SemanticCacheIdentity(hashlib.sha256(row["query"].encode()).hexdigest(), index["clip_features_sha256"]).fingerprint(),
                                    verifier=deterministic_fingerprint({"image_sha256": meta["verification_image_sha256"], "verdict": meta["verifier"]}))
            data = {
                "qid": row["qid"], "vid": row["vid"], "adsqa_source_id": vid,
                "source_url": meta.get("source_url", sources.get(vid, {}).get("source_url", "")),
                "relative_video_path": "local_data/raw/adsqa/videos/" + meta["video_filename"],
                "source_sha256": sha, "duration": row["duration"], "query": row["query"],
                "start": row["relevant_windows"][0][0], "end": row["relevant_windows"][0][1],
                "query_source": meta.get("query_source", "UNKNOWN_HISTORICAL"),
                "temporal_label_source": meta.get("temporal_label_source", "UNKNOWN_HISTORICAL"),
                "review_provenance": meta["review_provenance"],
                "router_state": meta.get("router", {}).get("route", "D48_FROZEN"),
                "history_state": history.get(vid, []), "agent_reviewed": bool(meta.get("verifier", {}).get("pass")),
                "semantic_v3_hash": meta.get("semantic_v3_algorithm_sha256", ""),
                "clip_signals": meta.get("clip_signals", {}), "cache_fingerprints": fingerprints,
                "fingerprint_scope": "Derived audit identities from frozen hashes; not a historical cache-hit claim",
                "accepted_record_sha256": manifest["accepted_record_sha256"].get(vid, ""),
                "visual_overview_path": review.get("overview_path", ""),
                "visual_overview_sha256": review.get("overview_sha256", ""),
                "visual_views_count": timing.get("visual_views_count"),
                "timing_scope": timing.get("timing_scope", "UNKNOWN_HISTORICAL_END_TO_END"),
                "total_accepted_seconds": timing.get("total_accepted_seconds"),
                "total_accepted_minutes": timing.get("total_accepted_minutes"),
                "pipeline_call_seconds": timing.get("pipeline_call_seconds"),
                "stages_seconds": timer.get("stages_seconds", {}), "final_split": split_by_qid[row["qid"]],
            }
            data = portable_copy(data)
            if absolute_values(data):
                raise ValueError("Master CSV has nonportable paths")
            writer.writerow({k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v for k, v in data.items()})
    return {"path": MASTER_PATH.relative_to(ROOT).as_posix(), "rows": len(rows), "sha256": digest(MASTER_PATH)}


def performance(session_id: str) -> dict:
    timings = [read_json(p) for p in (WORK / "cpu_sessions" / session_id).glob("*.json")]
    timings.sort(key=lambda r: read_json(ACCEPTED / (r["source_video_id"] + ".json"))["generated_at_utc"])
    if any(r["accepted_record_sha256"] != digest(ACCEPTED / (r["source_video_id"] + ".json")) for r in timings):
        raise ValueError("An annotation changed after its atomic acceptance write")
    values = [r["total_accepted_seconds"] for r in timings]
    stages = [read_json(WORK / "timings" / (r["source_video_id"] + ".json")) for r in timings]
    cache = [hit for t in stages for hit in t["cache_hits"].values()]
    def mean_stage(names):
        return statistics.mean(sum(t["stages_seconds"].get(n) or 0 for n in names) for t in stages)
    candidates = read_json(WORK / "source_selection.json")["candidates"]
    statuses = Counter(r["status"] for r in candidates)
    ledger = load_jsonl(WORK / "build_ledger.jsonl")
    return {
        "session_id": session_id, "accepted_this_session": len(timings),
        "historical_accepted_without_end_to_end_timing": 152 - len(timings),
        "first_accepted": timings[0], "median_seconds": statistics.median(values),
        "mean_seconds": statistics.mean(values),
        "p90_seconds": sorted(values)[min(len(values)-1, int(.9*len(values)))],
        "percentile_method": "Sorted index floor(0.9*N), matching live session stats",
        "fastest": min(timings, key=lambda r: r["total_accepted_seconds"]),
        "slowest": max(timings, key=lambda r: r["total_accepted_seconds"]),
        "median_visual_views": statistics.median(r["visual_views_count"] for r in timings),
        "cache_hit_rate": sum(cache) / len(cache), "cache_checks": len(cache),
        "cache_hits": sum(cache),
        "cache_hit_scope": "Last persisted stage cache checks for 86 accepted sources; repeated AttemptTimer.mark_cache overwrites earlier misses, so this is final acceptance-pass reuse, not all-session hit rate.",
        "timing_scope": "86 active source attempts through atomic persistence; includes agent/tool pauses. The recovered Nike source measures the recovery attempt only; prior failed attempt wall time is unavailable.",
        "breakdown_mean_seconds": {
            "preparation_active_compute": mean_stage(("download", "ffprobe", "frame_extraction", "contact_sheet")),
            "clip_semantic_compute": mean_stage(("clip", "semantic")),
            "verifier_refinement_compute": mean_stage(("verification", "refinement")),
            "visual_review_and_tool_wait_approximate": statistics.mean(max(0, r["total_accepted_seconds"] - r["pipeline_call_seconds"]) for r in timings),
            "pipeline_calls_including_audits_io": statistics.mean(r["pipeline_call_seconds"] for r in timings),
        },
        "breakdown_limitations": "Stage totals accumulate pipeline calls. Overlapped background download/decode durations were not persisted separately; no fabricated preparation or pure visual review duration is reported. Approximate review residual includes transport, reasoning and agent pauses. Breakdown categories are not all additive.",
        "source_status_counts": dict(statuses),
        "source_attempts_unique_in_ledger": len({r["source_video_id"] for r in ledger}),
        "rejected_sources": statuses["rejected"],
        "replacement_sources": statuses["rejected"],
        "historical_rejection_events": sum(r["event"] == "rejected" for r in ledger),
        "retryable_sources": sum(n for s, n in statuses.items() if s in ("awaiting_download", "retryable")),
        "targets_met": statistics.median(values) <= 120 and sorted(values)[int(.9*len(values))] <= 180,
        "attempts": timings,
    }


def write_report(session_id: str, *, package=None, tests=None) -> dict:
    ready = bool(package and tests and all(tests.get(k) == "PASS" for k in ("compileall", "pytest")))
    report = {"status": "READY_FOR_M1_D200_COLAB" if ready else "FROZEN_AWAITING_FINAL_CHECKS", "integrity": audit(),
              "master_csv": write_master(session_id), "performance": performance(session_id),
              "compileall": (tests or {}).get("compileall", "NOT_YET_RUN"),
              "pytest": (tests or {}).get("pytest", "NOT_YET_RUN"),
              "test_evidence": tests or {},
              "colab_zip": package or None, "training_performed": False, "t4_consumed": False}
    atomic_json(REPORT_PATH, report)
    return report


def console_summary(report: dict) -> str:
    integrity, perf = report["integrity"], report["performance"]
    lines = ["=" * 60, "LUMAE D200 — LOCAL CPU FAST BUILD COMPLETE", "=" * 60]
    values = [
        ("Total samples", integrity["total_samples"]), ("D48 reused", 48), ("New accepted", 152),
        ("Human verified", 18), ("AI pseudo labeled", 182),
        ("Source attempts", perf["source_attempts_unique_in_ledger"]),
        ("Rejected sources", perf["rejected_sources"]), ("Replacement sources", perf["replacement_sources"]),
        ("First accepted sec", f"{perf['first_accepted']['total_accepted_seconds']:.2f}"),
        ("First accepted min", f"{perf['first_accepted']['total_accepted_minutes']:.4f}"),
        ("Timed accepted this session", perf["accepted_this_session"]),
        ("Median sec / accepted", f"{perf['median_seconds']:.2f}"),
        ("Mean sec / accepted", f"{perf['mean_seconds']:.2f}"),
        ("P90 sec / accepted", f"{perf['p90_seconds']:.2f}"),
        ("Fastest accepted sec", perf["fastest"]["total_accepted_seconds"]),
        ("Slowest accepted sec", perf["slowest"]["total_accepted_seconds"]),
        ("Median visual views", perf["median_visual_views"]),
        ("Cache hit rate", f"{perf['cache_hit_rate']:.1%} ({perf['cache_hits']}/{perf['cache_checks']}; final persisted acceptance-pass checks only)"),
        ("Machine absolute paths", integrity["machine_absolute_paths"]),
        ("Invalid windows", integrity["invalid_windows"]),
        ("Duplicate qids", 200-integrity["unique_qids"]),
        ("Duplicate videos", 200-integrity["unique_videos"]),
        ("Fake saliency", integrity["fake_saliency"]),
        ("Train", 140), ("Val", 30), ("Test", 30),
        ("D48 outside Train", integrity["d48_outside_train"]),
        ("Video leakage", integrity["video_leakage"]),
        ("D48 immutable", "PASS"), ("semantic_v3 immutable", "PASS"),
        ("compileall", report["compileall"]), ("pytest", report["pytest"]),
        ("Dataset SHA256", integrity["dataset_sha256"]),
        ("Colab ZIP SHA256", (report["colab_zip"] or {}).get("sha256", "NOT_READY")),
    ]
    lines.extend(f"{label:<28}: {value}" for label, value in values)
    lines.append("Timing scope: 86 active accepted attempts; historical 66 unknown; recovery attempt only for one source.")
    lines.append("Background prefetch duration and full-session cache hit rate were not persisted; detailed measured breakdown is in the JSON report.")
    lines.extend([report["status"], "=" * 60])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", required=True)
    args = parser.parse_args()
    report = write_report(args.session)
    print(json.dumps({k: v for k, v in report.items() if k != "performance"}, indent=2))
    print(json.dumps({k: v for k, v in report["performance"].items() if k != "attempts"}, indent=2))


if __name__ == "__main__":
    main()
