"""CPU-only dataset audit CLI for inspecting canonical JSONL datasets."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any

from ..io.jsonl import read_jsonl
from ..validation.temporal import validate_duration, validate_windows


def _calc_stats(numbers: list[float]) -> dict[str, float]:
    if not numbers:
        return {"min": 0.0, "mean": 0.0, "median": 0.0, "max": 0.0}
    return {
        "min": round(min(numbers), 3),
        "mean": round(statistics.mean(numbers), 3),
        "median": round(statistics.median(numbers), 3),
        "max": round(max(numbers), 3),
    }


def audit_dataset_file(input_path: Path | str) -> dict[str, Any]:
    """Audit a JSONL file and return descriptive statistics."""
    path = Path(input_path)
    if not path.is_file():
        raise FileNotFoundError(f"Input file does not exist: {path}")

    total_samples = 0
    qid_counter: Counter[str | int] = Counter()
    vid_set: set[str] = set()
    query_set: set[str] = set()
    vid_query_pairs: Counter[tuple[str, str]] = Counter()

    durations: list[float] = []
    window_counts: list[float] = []
    window_lengths: list[float] = []

    invalid_windows = 0
    invalid_durations = 0
    empty_queries = 0
    samples_with_saliency = 0
    samples_with_clips = 0

    for idx, record in enumerate(read_jsonl(path)):
        total_samples += 1

        # QID
        qid = record.get("qid")
        if qid is not None:
            qid_counter[qid] += 1

        # VID
        vid = str(record.get("vid", "")).strip()
        if vid:
            vid_set.add(vid)

        # Query
        query = str(record.get("query", "")).strip()
        if query:
            query_set.add(query)
        else:
            empty_queries += 1

        if vid and query:
            vid_query_pairs[(vid, query)] += 1

        # Duration
        dur = record.get("duration")
        dur_ok, _ = validate_duration(dur)  # type: ignore[arg-type]
        if dur_ok:
            dur_float = float(dur)
            durations.append(dur_float)

            # Windows
            windows = record.get("relevant_windows", [])
            if isinstance(windows, list):
                window_counts.append(len(windows))
                win_ok, _ = validate_windows(windows, dur_float, tolerance=1.5, allow_empty=True)
                if not win_ok:
                    invalid_windows += 1
                for w in windows:
                    if isinstance(w, (list, tuple)) and len(w) == 2:
                        s, e = float(w[0]), float(w[1])
                        if 0 <= s < e:
                            window_lengths.append(e - s)
            else:
                invalid_windows += 1
        else:
            invalid_durations += 1

        # Optional fields
        if record.get("saliency_scores") is not None:
            samples_with_saliency += 1
        if record.get("relevant_clip_ids") is not None:
            samples_with_clips += 1

    duplicate_qids = {qid: c for qid, c in qid_counter.items() if c > 1}
    duplicate_vid_queries = {f"{v} | {q}": c for (v, q), c in vid_query_pairs.items() if c > 1}

    return {
        "file": str(path.resolve()),
        "total_samples": total_samples,
        "unique_qids": len(qid_counter),
        "unique_vids": len(vid_set),
        "unique_queries": len(query_set),
        "duplicate_qids_count": len(duplicate_qids),
        "duplicate_vid_queries_count": len(duplicate_vid_queries),
        "empty_queries_count": empty_queries,
        "invalid_durations_count": invalid_durations,
        "invalid_windows_count": invalid_windows,
        "duration_seconds": _calc_stats(durations),
        "windows_per_sample": _calc_stats(window_counts),
        "window_length_seconds": _calc_stats(window_lengths),
        "optional_features": {
            "saliency_presence_pct": round(samples_with_saliency / max(1, total_samples) * 100, 2),
            "relevant_clips_presence_pct": round(samples_with_clips / max(1, total_samples) * 100, 2),
        },
    }


def print_report(results: dict[str, Any]) -> None:
    """Print human-readable formatted audit report to console."""
    print("=" * 60)
    print(f"LUMAE DATASET AUDIT REPORT: {Path(results['file']).name}")
    print("=" * 60)
    print(f"Total Samples           : {results['total_samples']}")
    print(f"Unique QIDs             : {results['unique_qids']}")
    print(f"Unique Videos (vids)    : {results['unique_vids']}")
    print(f"Unique Queries          : {results['unique_queries']}")
    print("-" * 60)
    print(f"Duplicate QIDs          : {results['duplicate_qids_count']}")
    print(f"Duplicate (vid, query)  : {results['duplicate_vid_queries_count']}")
    print(f"Empty Queries           : {results['empty_queries_count']}")
    print(f"Invalid Durations       : {results['invalid_durations_count']}")
    print(f"Invalid Windows         : {results['invalid_windows_count']}")
    print("-" * 60)
    d = results["duration_seconds"]
    print(f"Video Duration (s)      : min={d['min']}, mean={d['mean']}, med={d['median']}, max={d['max']}")
    w = results["windows_per_sample"]
    print(f"Windows per Query       : min={w['min']}, mean={w['mean']}, med={w['median']}, max={w['max']}")
    wl = results["window_length_seconds"]
    print(f"Window Length (s)       : min={wl['min']}, mean={wl['mean']}, med={wl['median']}, max={wl['max']}")
    print("-" * 60)
    feat = results["optional_features"]
    print(f"Saliency Scores Present : {feat['saliency_presence_pct']}% of samples")
    print(f"Clip IDs Present        : {feat['relevant_clips_presence_pct']}% of samples")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit canonical JSONL temporal grounding datasets.")
    parser.add_argument("--input", "-i", required=True, help="Path to input JSONL dataset file.")
    parser.add_argument("--output", "-o", help="Optional path to output audit summary as JSON.")
    args = parser.parse_args()

    try:
        report = audit_dataset_file(args.input)
        print_report(report)
        if args.output:
            out_path = Path(args.output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(f"Saved JSON report to: {out_path}")
    except Exception as exc:
        print(f"Audit failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
