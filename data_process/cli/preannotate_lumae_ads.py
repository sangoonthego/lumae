"""CLI tool for AI-assisted temporal pre-annotation of LUMAE Ads pilot.

Stage A.2.5B Protocol:
- Identifies pending unreviewed samples
- Distinguishes blind human holdout samples
- Generates AI candidate annotations into local_data/annotations/lumae_ads/ai_preannotations.csv
- Supports --dry-run, --sample-id, --limit, --force, --force-all, --benchmark-reviewed
- ABSOLUTE GUARD: NEVER modifies human_primary.csv or ground truth
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add repository root to path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_process.adsqa.ai_preannotator import preannotate_sample
from data_process.adsqa.annotation_qc import compute_temporal_iou
from data_process.annotation.models import AIConfidence, AIPreannotationRecord, ReviewStatus
from data_process.annotation.storage import (
    append_or_update_ai_preannotation,
    get_ai_preannotations_path,
    get_videos_dir,
    is_blind_holdout,
    load_ai_preannotations,
    load_blind_holdout,
    load_primary_records,
    save_ai_preannotations,
)


def run_preannotation(
    sample_id: str | None = None,
    limit: int | None = None,
    dry_run: bool = False,
    force: bool = False,
    force_all: bool = False,
    benchmark_reviewed: bool = False,
) -> int:
    """Execute pre-annotation CLI pipeline."""
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    print("=" * 60)
    print("LUMAE Ads - AI-Assisted Temporal Pre-Annotation (Stage A.2.5B)")
    print("=" * 60)
    print("SCIENTIFIC PRINCIPLE: AI candidates are advisory assistance only, NOT human GT.")
    print("ABSOLUTE GUARD: human_primary.csv will NEVER be modified by this tool.\n")

    primary_records = load_primary_records()
    total_samples = len(primary_records)
    reviewed_records = [r for r in primary_records if r.is_human_reviewed()]
    pending_records = [r for r in primary_records if not r.is_human_reviewed()]

    holdout_info = load_blind_holdout()
    holdout_ids = set(holdout_info.get("sample_ids", []))

    print(f"Total pilot samples:      {total_samples}")
    print(f"Already human-reviewed:   {len(reviewed_records)} ({[r.sample_id for r in reviewed_records]})")
    print(f"Pending review:           {len(pending_records)}")
    print(f"Blind human holdout pool: {len(holdout_ids)} samples (seed={holdout_info.get('seed', 42)})")

    existing_preannotations = load_ai_preannotations()
    output_path = get_ai_preannotations_path()
    videos_dir = get_videos_dir()

    # If benchmark mode is requested, run on reviewed samples for sanity comparison
    if benchmark_reviewed:
        print("\n[BENCHMARK] Evaluating AI candidate generator on known human-reviewed samples:")
        for r in reviewed_records:
            cand = preannotate_sample(r, videos_dir=videos_dir)
            human_w = r.get_effective_windows()
            ai_w = cand.get_windows()
            tiou = compute_temporal_iou(human_w, ai_w)
            print(f"  Sample {r.sample_id}:")
            print(f"    Human GT:     {human_w}")
            print(f"    AI Candidate: {ai_w} (Confidence: {cand.ai_confidence})")
            print(f"    Top-1 tIoU:   {tiou:.4f}")
        return 0

    # Determine worklist
    if sample_id:
        targets = [r for r in primary_records if r.sample_id == sample_id]
        if not targets:
            print(f"[ERROR] Sample '{sample_id}' not found in primary records.")
            return 1
    else:
        targets = pending_records

    # Filter targets based on resumability and force flags
    worklist = []
    skipped_count = 0
    for r in targets:
        is_already_done = r.sample_id in existing_preannotations
        if is_already_done and not force_all and not (force and sample_id == r.sample_id):
            skipped_count += 1
        else:
            worklist.append(r)

    if limit is not None and limit > 0:
        worklist = worklist[:limit]

    print(f"\nSamples to process:       {len(worklist)} (Skipped already generated: {skipped_count})")
    print(f"Target output file:       {output_path}\n")

    if dry_run:
        print("[DRY RUN] Execution plan (no annotations will be generated):")
        print("-" * 60)
        for idx, r in enumerate(worklist, 1):
            is_holdout = r.sample_id in holdout_ids
            holdout_badge = "[BLIND HOLDOUT]" if is_holdout else "[NON-HOLDOUT]"
            v_path = videos_dir / r.video_filename
            exists = "EXISTS" if v_path.is_file() else "MISSING"
            dur = r.duration_seconds
            sampling = 3.0 if dur > 60.0 else 2.0
            est_frames = int(dur / sampling)
            print(
                f"  {idx:02d}. {r.sample_id} {holdout_badge:<16} | {r.video_filename} ({exists}) | "
                f"Dur: {dur:5.1f}s | Intended sampling: {sampling}s (~{est_frames} frames) | Query: '{r.query[:35]}...'"
            )
        print("-" * 60)
        print("Dry run completed successfully. Zero files modified.")
        return 0

    if not worklist:
        print("[INFO] No pending samples require pre-annotation. Use --force or --force-all to re-run.")
        return 0

    print("[RUN] Starting AI pre-annotation batch generation...")
    success_count = 0
    fail_count = 0

    for idx, r in enumerate(worklist, 1):
        is_holdout = r.sample_id in holdout_ids
        holdout_tag = " [BLIND HOLDOUT]" if is_holdout else ""
        print(f"[{idx:02d}/{len(worklist):02d}] Processing {r.sample_id}{holdout_tag} ({r.video_filename})...", end="", flush=True)

        try:
            cand = preannotate_sample(r, videos_dir=videos_dir)
            append_or_update_ai_preannotation(cand)
            success_count += 1
            w_str = f"[{cand.ai_start_seconds}s-{cand.ai_end_seconds}s]" if cand.ai_start_seconds is not None else "[]"
            print(f" DONE -> {cand.ai_query_status} | {w_str} | {cand.ai_confidence}")
        except Exception as e:
            fail_count += 1
            print(f" FAILED -> {e}")

    print("\n" + "=" * 60)
    print("PRE-ANNOTATION SUMMARY")
    print("=" * 60)
    print(f"Successfully generated: {success_count}")
    print(f"Failed:                 {fail_count}")
    print(f"Saved to:               {output_path}")

    # Reload all preannotations to print distribution
    all_pre = load_ai_preannotations()
    high_c = sum(1 for c in all_pre.values() if c.ai_confidence == AIConfidence.HIGH.value)
    med_c = sum(1 for c in all_pre.values() if c.ai_confidence == AIConfidence.MEDIUM.value)
    low_c = sum(1 for c in all_pre.values() if c.ai_confidence == AIConfidence.LOW.value)

    print(f"\nTotal Candidates in Catalog: {len(all_pre)}")
    print(f"  HIGH Confidence:   {high_c}")
    print(f"  MEDIUM Confidence: {med_c}")
    print(f"  LOW Confidence:    {low_c}")
    print("=" * 60)
    return 0 if fail_count == 0 else 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LUMAE Ads AI-Assisted Temporal Pre-Annotation Generator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dry-run", action="store_true", help="Display execution plan without generating candidates.")
    parser.add_argument("--sample-id", type=str, default=None, help="Process a single sample ID.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples to process.")
    parser.add_argument("--force", action="store_true", help="Force regeneration for specified sample ID.")
    parser.add_argument("--force-all", action="store_true", help="Force regeneration across all target samples.")
    parser.add_argument("--benchmark-reviewed", action="store_true", help="Evaluate candidates against known reviewed human GT.")
    parser.add_argument("--headless", action="store_true", help="Acknowledge headless environment.")

    args = parser.parse_args()

    sys.exit(
        run_preannotation(
            sample_id=args.sample_id,
            limit=args.limit,
            dry_run=args.dry_run,
            force=args.force,
            force_all=args.force_all,
            benchmark_reviewed=args.benchmark_reviewed,
        )
    )


if __name__ == "__main__":
    main()
