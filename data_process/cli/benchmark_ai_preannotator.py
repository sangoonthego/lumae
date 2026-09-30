"""CLI tool for benchmarking AI temporal pre-annotations against human ground truth.

Stage A.2.5C Protocol:
- Evaluates 5 human-verified samples (or specified subsets)
- Computes Top-1 tIoU, start/end absolute errors, Hit@0.5, Hit@0.7
- Assesses semantic event matching
- Evaluates the planning quality gate (PASS_FOR_AI_ASSISTED_REVIEW vs FAIL_CURRENT_PREANNOTATOR)
- Generates JSON, Markdown, and per-sample CSV reports
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_process.adsqa.ai_benchmark import (
    BENCHMARK_TARGET_SAMPLES,
    evaluate_5sample_benchmark,
    evaluate_semantic_v2_benchmark,
)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    parser = argparse.ArgumentParser(
        description="LUMAE Ads — AI Preannotator 5-Sample Benchmark CLI (Stages A.2.5C & A.2.5D)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--sample-id",
        action="append",
        default=[],
        dest="sample_ids",
        help="Sample ID to include in benchmark (can be specified multiple times).",
    )
    parser.add_argument(
        "--version",
        choices=["v1", "v2"],
        default="v1",
        help="Preannotator version to benchmark: 'v1' (heuristic) or 'v2' (semantic).",
    )
    parser.add_argument(
        "--semantic-v2",
        action="store_true",
        help="Convenience alias to evaluate semantic_v2 pre-annotations (Stage A.2.5D).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Custom output directory for benchmark reports.",
    )
    parser.add_argument(
        "--json-only",
        action="store_true",
        help="Generate only machine-readable JSON report.",
    )
    parser.add_argument(
        "--strict-csv-only",
        action="store_true",
        help="Exclude in-memory diagnostic candidates and evaluate strictly against ai_preannotations.csv.",
    )

    args = parser.parse_args()

    target_samples = args.sample_ids if args.sample_ids else list(BENCHMARK_TARGET_SAMPLES)
    out_dir = Path(args.output_dir) if args.output_dir else None
    is_v2 = (args.version == "v2") or args.semantic_v2

    if not args.json_only:
        stage_title = "Stage A.2.5D (Semantic V2)" if is_v2 else "Stage A.2.5C (Heuristic V1)"
        print("=" * 70)
        print(f"LUMAE Ads - AI Preannotator Benchmark against Human GT ({stage_title})")
        print("=" * 70)
        print(f"Target Samples ({len(target_samples)}): {', '.join(target_samples)}\n")

    try:
        if is_v2:
            report = evaluate_semantic_v2_benchmark(
                sample_ids=target_samples,
                output_dir=out_dir,
                json_only=args.json_only,
            )
        else:
            report = evaluate_5sample_benchmark(
                sample_ids=target_samples,
                output_dir=out_dir,
                json_only=args.json_only,
                include_diagnostic_candidates=(not args.strict_csv_only),
            )
    except Exception as e:
        print(f"[ERROR] Benchmark execution failed: {e}", file=sys.stderr)
        sys.exit(1)

    if args.json_only:
        print(json.dumps(report, indent=2))
        return

    inp = report["input_audit"]
    q_gate = report["quality_gate"]
    agg = report["aggregate_metrics"]
    sem = report["semantic_event_quality"]

    print("INPUT AUDIT:")
    print(f"  Human CSV SHA256: {inp['human_primary_csv_sha256'][:16]}...")
    if is_v2:
        print(f"  V2 CSV SHA256:    {inp['semantic_preannotations_v2_csv_sha256'][:16]}...")
    else:
        print(f"  AI CSV SHA256:    {inp['ai_preannotations_csv_sha256'][:16]}...")
    print(f"  Evaluated:        {inp['evaluated_sample_count']} / {inp['total_benchmark_samples']}")
    if inp["missing_samples"]:
        print(f"  Missing:          {', '.join(inp['missing_samples'])}")

    if is_v2 and "v1_vs_v2_comparison" in report:
        comp = report["v1_vs_v2_comparison"]
        print("\nV1 vs V2 COMPARISON TABLE:")
        print("-" * 78)
        print(f"{'Sample ID':<24} {'Human GT':<14} {'V1 Window':<14} {'V1 tIoU':<8} {'V2 Window':<14} {'V2 tIoU':<8}")
        print("-" * 78)
        for r in comp["sample_comparisons"]:
            h_str = f"[{r['human_gt'][0]:.1f}, {r['human_gt'][1]:.1f}]"
            v1_str = f"[{r['v1_best_window'][0]:.1f}, {r['v1_best_window'][1]:.1f}]" if r["v1_best_window"] else "[]"
            v2_str = f"[{r['v2_best_window'][0]:.1f}, {r['v2_best_window'][1]:.1f}]" if r["v2_best_window"] else "[]"
            print(f"{r['sample_id']:<24} {h_str:<14} {v1_str:<14} {r['v1_tiou']:<8.4f} {v2_str:<14} {r['v2_tiou']:<8.4f}")
        print("-" * 78)
        print(f"Mean Top-1 tIoU: V1 = {comp['v1_mean_tiou']:.4f} -> V2 = {comp['v2_mean_tiou']:.4f} (Delta: {comp['tiou_absolute_improvement']:+.4f})")
        print(f"R1@0.5 (Hit@0.5): V1 = {comp['v1_r1_0_5']:.1f}% -> V2 = {comp['v2_r1_0_5']:.1f}%")
        print(f"R1@0.7 (Hit@0.7): V1 = {comp['v1_r1_0_7']:.1f}% -> V2 = {comp['v2_r1_0_7']:.1f}%")
        print(f"Semantic Correct: V1 = {comp['v1_semantic_correct']} / 5 -> V2 = {comp['v2_semantic_correct']} / 5")
    else:
        print("\nPER-SAMPLE EVALUATION:")
        print("-" * 70)
        print(f"{'Sample ID':<24} {'Human GT':<14} {'AI Window':<14} {'tIoU':<8} {'Semantic Match':<14}")
        print("-" * 70)
        for s in report["sample_level_results"]:
            h_str = f"[{s['human_gt'][0]:.1f}, {s['human_gt'][1]:.1f}]" if s.get("human_gt") else "[]"
            b_ai = s.get("best_ai_window", [])
            ai_str = f"[{b_ai[0]:.1f}, {b_ai[1]:.1f}]" if b_ai else "[]"
            print(f"{s['sample_id']:<24} {h_str:<14} {ai_str:<14} {s['tIoU']:<8.4f} {s['semantic_event_match']:<14}")
        print("-" * 70)

    print("\nAGGREGATE PERFORMANCE METRICS:")
    print(f"  Mean Top-1 tIoU:       {agg['mean_tIoU']:.4f} (Threshold >= 0.6000)")
    print(f"  Median tIoU:           {agg['median_tIoU']:.4f}")
    print(f"  Min / Max tIoU:        {agg['min_tIoU']:.4f} / {agg['max_tIoU']:.4f}")
    print(f"  R1@0.5 (Hit@0.5):      {agg['r1_at_0_5']:.1f}%")
    print(f"  R1@0.7 (Hit@0.7):      {agg['r1_at_0_7']:.1f}%")
    print(f"  Mean Start Abs Error:  {agg['mean_start_abs_error_seconds']:.2f}s")
    print(f"  Mean End Abs Error:    {agg['mean_end_abs_error_seconds']:.2f}s")
    print(f"  Mean Boundary MAE:     {agg['mean_boundary_mae_seconds']:.2f}s")

    print("\nSEMANTIC EVENT QUALITY:")
    print(f"  CORRECT_EVENT:         {sem['correct_event_count']} (Threshold >= 4)")
    print(f"  PARTIAL_EVENT:         {sem['partial_event_count']}")
    print(f"  WRONG_EVENT:           {sem['wrong_event_count']} (Threshold <= 1)")
    print(f"  UNRESOLVED:            {sem['unresolved_count']}")

    print("\n" + "=" * 70)
    print(f"PLANNING QUALITY GATE: {q_gate['gate_result']}")
    print(f"STAGE STATUS:          {q_gate['final_status']}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
