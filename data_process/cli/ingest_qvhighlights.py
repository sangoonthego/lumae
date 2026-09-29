"""CLI tool for ingesting and validating official QVHighlights annotations."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import sys
from typing import Any

from ..adapters.qvhighlights import QVHighlightsAdapter
from ..io.jsonl import read_jsonl, write_jsonl
from ..schemas.temporal_sample import CanonicalTemporalSample, RejectedSampleRecord
from ..validation.leakage import find_video_leakage


def _calc_stats(numbers: list[float]) -> dict[str, float]:
    if not numbers:
        return {"min": 0.0, "mean": 0.0, "median": 0.0, "max": 0.0}
    return {
        "min": round(min(numbers), 3),
        "mean": round(statistics.mean(numbers), 3),
        "median": round(statistics.median(numbers), 3),
        "max": round(max(numbers), 3),
    }


def profile_split(
    samples: list[CanonicalTemporalSample],
    split_name: str,
) -> dict[str, Any]:
    """Calculate comprehensive dataset profiling metrics for a split."""
    total_samples = len(samples)
    qids = [s.qid for s in samples]
    vids = [s.vid for s in samples]
    queries = [s.query for s in samples]

    durations: list[float] = [s.duration for s in samples]
    window_counts: list[float] = [len(s.relevant_windows) for s in samples]
    window_lengths: list[float] = [
        w[1] - w[0] for s in samples for w in s.relevant_windows if w[1] > w[0]
    ]

    clip_counts: list[float] = [
        len(s.relevant_clip_ids) for s in samples if s.relevant_clip_ids is not None
    ]

    # Saliency score distribution and annotator disagreement
    all_annotator_scores: list[float] = []
    clip_spreads: list[float] = []
    grid_aligned_windows = 0
    total_windows = 0

    for s in samples:
        for w in s.relevant_windows:
            total_windows += 1
            # Check 2-second alignment (modulo 2 < 0.05 or > 1.95)
            if abs(w[0] % 2.0) < 0.05 or abs(w[0] % 2.0 - 2.0) < 0.05:
                grid_aligned_windows += 1

        if s.saliency_scores:
            for clip_scores in s.saliency_scores:
                for score in clip_scores:
                    all_annotator_scores.append(score)
                spread = max(clip_scores) - min(clip_scores)
                clip_spreads.append(spread)

    score_counter = Counter(all_annotator_scores)
    spread_counter = Counter(clip_spreads)

    qid_counter = Counter(qids)
    duplicate_qids = {str(k): v for k, v in qid_counter.items() if v > 1}

    vid_query_pairs = Counter([(s.vid, s.query) for s in samples])
    duplicate_pairs = {f"{k[0]} | {k[1]}": v for k, v in vid_query_pairs.items() if v > 1}

    return {
        "split": split_name,
        "sample_count": total_samples,
        "unique_qids": len(set(qids)),
        "unique_vids": len(set(vids)),
        "unique_queries": len(set(queries)),
        "duplicate_qids_count": len(duplicate_qids),
        "duplicate_vid_query_pairs_count": len(duplicate_pairs),
        "video_duration_seconds": _calc_stats(durations),
        "windows_per_query": _calc_stats(window_counts),
        "window_length_seconds": _calc_stats(window_lengths),
        "relevant_clips_per_query": _calc_stats(clip_counts),
        "temporal_grid": {
            "total_windows": total_windows,
            "aligned_to_2s_grid_count": grid_aligned_windows,
            "aligned_pct": round(grid_aligned_windows / max(1, total_windows) * 100, 2),
        },
        "saliency_statistics": {
            "total_annotator_ratings": len(all_annotator_scores),
            "score_distribution": {str(int(k) if k.is_integer() else k): v for k, v in sorted(score_counter.items())},
            "annotator_disagreement_spread": {
                "total_relevant_clips": len(clip_spreads),
                "spread_distribution": {str(int(k) if k.is_integer() else k): v for k, v in sorted(spread_counter.items())},
                "mean_spread": round(statistics.mean(clip_spreads), 3) if clip_spreads else 0.0,
                "median_spread": round(statistics.median(clip_spreads), 3) if clip_spreads else 0.0,
                "max_spread": round(max(clip_spreads), 3) if clip_spreads else 0.0,
            },
        },
    }


def ingest_qvhighlights(
    train_source_path: Path | str,
    val_source_path: Path | str,
    output_canonical_dir: Path | str,
    output_reports_dir: Path | str,
    namespace_vid: bool = False,
) -> dict[str, Any]:
    """Ingest, validate, profile, and canonicalize QVHighlights dataset."""
    train_file = Path(train_source_path)
    val_file = Path(val_source_path)
    out_dir = Path(output_canonical_dir)
    rep_dir = Path(output_reports_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    rep_dir.mkdir(parents=True, exist_ok=True)

    adapter = QVHighlightsAdapter(namespace_vid=namespace_vid, strict_annotator_count=True)

    print(f"=== Ingesting QVHighlights Train: {train_file} ===")
    train_samples, train_rejected = adapter.process_file(train_file)
    print(f"  Valid Train Samples   : {len(train_samples):,}")
    print(f"  Rejected Train Records: {len(train_rejected):,}")

    print(f"=== Ingesting QVHighlights Val: {val_file} ===")
    val_samples, val_rejected = adapter.process_file(val_file)
    print(f"  Valid Val Samples     : {len(val_samples):,}")
    print(f"  Rejected Val Records  : {len(val_rejected):,}")

    # Write Canonical Outputs
    train_canon_path = out_dir / "train.jsonl"
    val_canon_path = out_dir / "val.jsonl"
    write_jsonl(train_canon_path, train_samples)
    write_jsonl(val_canon_path, val_samples)
    print(f"Saved canonical train JSONL: {train_canon_path}")
    print(f"Saved canonical val JSONL  : {val_canon_path}")

    # Write Rejections if any
    all_rejected = train_rejected + val_rejected
    if all_rejected:
        rejections_path = rep_dir / "rejected_records.jsonl"
        write_jsonl(rejections_path, all_rejected)
        print(f"Saved {len(all_rejected)} rejection records to: {rejections_path}")

    # Split Leakage Check
    leakage = find_video_leakage(train_samples, val_samples)
    shared_vids = leakage.get("train_val", set())
    if shared_vids:
        print(f"WARNING: Video leakage detected! {len(shared_vids)} shared video(s) between train and val!")
    else:
        print("VERIFIED: 0 shared videos between train and val splits.")

    # Profiles
    train_profile = profile_split(train_samples, "train")
    val_profile = profile_split(val_samples, "val")

    # Sanity References Check
    EXPECTED_SANITY = {
        "train": {"samples": 7218, "videos": 7100},
        "val": {"samples": 1550, "videos": 1519},
        "leakage_shared_videos": 0,
    }

    sanity_comparison = {
        "train_samples": {
            "expected": EXPECTED_SANITY["train"]["samples"],
            "actual": len(train_samples),
            "difference": len(train_samples) - EXPECTED_SANITY["train"]["samples"],
        },
        "train_videos": {
            "expected": EXPECTED_SANITY["train"]["videos"],
            "actual": train_profile["unique_vids"],
            "difference": train_profile["unique_vids"] - EXPECTED_SANITY["train"]["videos"],
        },
        "val_samples": {
            "expected": EXPECTED_SANITY["val"]["samples"],
            "actual": len(val_samples),
            "difference": len(val_samples) - EXPECTED_SANITY["val"]["samples"],
        },
        "val_videos": {
            "expected": EXPECTED_SANITY["val"]["videos"],
            "actual": val_profile["unique_vids"],
            "difference": val_profile["unique_vids"] - EXPECTED_SANITY["val"]["videos"],
        },
        "train_val_shared_videos": {
            "expected": EXPECTED_SANITY["leakage_shared_videos"],
            "actual": len(shared_vids),
            "difference": len(shared_vids),
        },
    }

    full_report: dict[str, Any] = {
        "dataset": "QVHighlights",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "sanity_check": sanity_comparison,
        "splits": {
            "train": train_profile,
            "val": val_profile,
        },
        "leakage": {
            "train_val_shared_videos_count": len(shared_vids),
            "shared_videos": sorted(list(shared_vids)),
        },
        "rejections": {
            "train_rejected_count": len(train_rejected),
            "val_rejected_count": len(val_rejected),
            "total_rejected": len(all_rejected),
        },
    }

    # Save JSON report
    report_json_path = rep_dir / "data_profile.json"
    report_json_path.write_text(json.dumps(full_report, indent=2), encoding="utf-8")
    print(f"Saved data profile JSON: {report_json_path}")

    # Generate Markdown report
    report_md_path = rep_dir / "data_profile.md"
    md_content = generate_markdown_report(full_report)
    report_md_path.write_text(md_content, encoding="utf-8")
    print(f"Saved data profile Markdown: {report_md_path}")

    return full_report


def generate_markdown_report(report: dict[str, Any]) -> str:
    """Generate human-readable Markdown data profile report."""
    train = report["splits"]["train"]
    val = report["splits"]["val"]
    sanity = report["sanity_check"]

    lines = [
        "# QVHighlights Dataset Ingestion & Profile Report",
        "",
        f"**Generated**: {report['timestamp_utc']}",
        "",
        "## 1. Sanity Verification vs. Official Baseline Reference",
        "",
        "| Metric | Expected Reference | Actual Ingested | Difference | Status |",
        "| :--- | :--- | :--- | :--- | :--- |",
        f"| Train Samples | {sanity['train_samples']['expected']:,} | {sanity['train_samples']['actual']:,} | {sanity['train_samples']['difference']} | {'MATCH' if sanity['train_samples']['difference'] == 0 else 'DIFF'} |",
        f"| Train Videos | {sanity['train_videos']['expected']:,} | {sanity['train_videos']['actual']:,} | {sanity['train_videos']['difference']} | {'MATCH' if sanity['train_videos']['difference'] == 0 else 'DIFF'} |",
        f"| Val Samples | {sanity['val_samples']['expected']:,} | {sanity['val_samples']['actual']:,} | {sanity['val_samples']['difference']} | {'MATCH' if sanity['val_samples']['difference'] == 0 else 'DIFF'} |",
        f"| Val Videos | {sanity['val_videos']['expected']:,} | {sanity['val_videos']['actual']:,} | {sanity['val_videos']['difference']} | {'MATCH' if sanity['val_videos']['difference'] == 0 else 'DIFF'} |",
        f"| Train/Val Video Overlap | 0 | {sanity['train_val_shared_videos']['actual']} | 0 | {'ZERO_LEAKAGE' if sanity['train_val_shared_videos']['actual'] == 0 else 'LEAKAGE_DETECTED'} |",
        "",
        "## 2. Split Summaries",
        "",
        "| Metric | Train Split | Validation Split |",
        "| :--- | :--- | :--- |",
        f"| Total Samples | {train['sample_count']:,} | {val['sample_count']:,} |",
        f"| Unique QIDs | {train['unique_qids']:,} | {val['unique_qids']:,} |",
        f"| Unique Videos (vids) | {train['unique_vids']:,} | {val['unique_vids']:,} |",
        f"| Unique Queries | {train['unique_queries']:,} | {val['unique_queries']:,} |",
        f"| Duplicate QIDs | {train['duplicate_qids_count']} | {val['duplicate_qids_count']} |",
        f"| Duplicate (vid, query) | {train['duplicate_vid_query_pairs_count']} | {val['duplicate_vid_query_pairs_count']} |",
        "",
        "## 3. Temporal Distributions",
        "",
        "| Metric | Train (min / mean / med / max) | Val (min / mean / med / max) |",
        "| :--- | :--- | :--- |",
        f"| Video Duration (s) | {train['video_duration_seconds']['min']} / {train['video_duration_seconds']['mean']} / {train['video_duration_seconds']['median']} / {train['video_duration_seconds']['max']} | {val['video_duration_seconds']['min']} / {val['video_duration_seconds']['mean']} / {val['video_duration_seconds']['median']} / {val['video_duration_seconds']['max']} |",
        f"| Window Length (s) | {train['window_length_seconds']['min']} / {train['window_length_seconds']['mean']} / {train['window_length_seconds']['median']} / {train['window_length_seconds']['max']} | {val['window_length_seconds']['min']} / {val['window_length_seconds']['mean']} / {val['window_length_seconds']['median']} / {val['window_length_seconds']['max']} |",
        f"| Windows per Query | {train['windows_per_query']['min']} / {train['windows_per_query']['mean']} / {train['windows_per_query']['median']} / {train['windows_per_query']['max']} | {val['windows_per_query']['min']} / {val['windows_per_query']['mean']} / {val['windows_per_query']['median']} / {val['windows_per_query']['max']} |",
        f"| Relevant Clips / Query | {train['relevant_clips_per_query']['min']} / {train['relevant_clips_per_query']['mean']} / {train['relevant_clips_per_query']['median']} / {train['relevant_clips_per_query']['max']} | {val['relevant_clips_per_query']['min']} / {val['relevant_clips_per_query']['mean']} / {val['relevant_clips_per_query']['median']} / {val['relevant_clips_per_query']['max']} |",
        "",
        "## 4. Multi-Annotator Saliency & Disagreement Analysis",
        "",
        "In QVHighlights, each relevant 2-second clip is rated by **3 independent annotators** on a discrete 0–4 scale (`0=Very Bad`, `1=Bad`, `2=Fair`, `3=Good`, `4=Very Good`).",
        "",
        "### Annotator Rating Distribution (Train + Val)",
        f"- **Train Ratings Count**: {train['saliency_statistics']['total_annotator_ratings']:,}",
        f"- **Train Score Histogram**: {train['saliency_statistics']['score_distribution']}",
        f"- **Val Ratings Count**: {val['saliency_statistics']['total_annotator_ratings']:,}",
        f"- **Val Score Histogram**: {val['saliency_statistics']['score_distribution']}",
        "",
        "### Clip-Level Annotator Spread (`max_score - min_score`)",
        "Spread measures annotation variance across the 3 raters for a single relevant clip:",
        f"- **Train Mean Spread**: {train['saliency_statistics']['annotator_disagreement_spread']['mean_spread']:.3f} (median: {train['saliency_statistics']['annotator_disagreement_spread']['median_spread']}, max: {train['saliency_statistics']['annotator_disagreement_spread']['max_spread']})",
        f"- **Train Spread Distribution**: {train['saliency_statistics']['annotator_disagreement_spread']['spread_distribution']}",
        f"- **Val Mean Spread**: {val['saliency_statistics']['annotator_disagreement_spread']['mean_spread']:.3f} (median: {val['saliency_statistics']['annotator_disagreement_spread']['median_spread']}, max: {val['saliency_statistics']['annotator_disagreement_spread']['max_spread']})",
        f"- **Val Spread Distribution**: {val['saliency_statistics']['annotator_disagreement_spread']['spread_distribution']}",
        "",
        "## 5. Temporal Grid Alignment (~2-second clips)",
        f"- **Train Window 2s-Alignment**: {train['temporal_grid']['aligned_pct']}% of windows align with 2s boundaries.",
        f"- **Val Window 2s-Alignment**: {val['temporal_grid']['aligned_pct']}% of windows align with 2s boundaries.",
        "",
        "## 6. Pipeline Rejection & Integrity Summary",
        f"- **Train Rejections**: {report['rejections']['train_rejected_count']}",
        f"- **Val Rejections**: {report['rejections']['val_rejected_count']}",
        f"- **Total Rejections**: {report['rejections']['total_rejected']}",
        "",
        "> Ground truth windows and annotator ratings were preserved exactly without lossy flattening or rounding.",
    ]

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest and profile official QVHighlights annotations.")
    parser.add_argument(
        "--train",
        default="local_data/raw/qvhighlights/highlight_train_release.jsonl",
        help="Path to official raw train JSONL",
    )
    parser.add_argument(
        "--val",
        default="local_data/raw/qvhighlights/highlight_val_release.jsonl",
        help="Path to official raw val JSONL",
    )
    parser.add_argument(
        "--output-root",
        default="local_data/canonical/qvhighlights",
        help="Directory to save canonical JSONL files",
    )
    parser.add_argument(
        "--report-root",
        default="local_data/reports/qvhighlights",
        help="Directory to save profile reports",
    )
    parser.add_argument(
        "--namespace-vid",
        action="store_true",
        help="Prefix video IDs with 'qvh__'",
    )
    args = parser.parse_args()

    try:
        ingest_qvhighlights(
            train_source_path=args.train,
            val_source_path=args.val,
            output_canonical_dir=args.output_root,
            output_reports_dir=args.report_root,
            namespace_vid=args.namespace_vid,
        )
    except Exception as exc:
        print(f"Ingestion failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
