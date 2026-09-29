"""CLI utility to enforce quality gates, calculate human agreement, and freeze Lumae Ads Pilot GT."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

from ..adapters.lumae_ads import LumaeAdsAdapter
from ..adsqa.annotation_qc import compute_temporal_iou
from ..annotation.models import ReviewStatus
from ..annotation.storage import (
    get_annotations_dir,
    get_repo_root,
    get_reports_dir,
    get_videos_dir,
    load_adjudicated_records,
    load_multi_windows,
    load_primary_records,
    load_secondary_records,
)
from ..annotation.validation import validate_multi_windows, validate_query, validate_temporal_boundaries
from ..io.jsonl import write_jsonl
from ..schemas.temporal_sample import CanonicalTemporalSample


def compute_sha256(path: Path) -> str:
    """Compute SHA256 hexadecimal digest of a local file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def freeze_lumae_ads_pilot(
    force: bool = False,
    min_secondary_fraction: float = 0.20,
    agreement_threshold: float = 0.70,
) -> dict[str, Any]:
    """Execute quality gates, agreement calculation, and generate frozen dataset v1.0."""
    ann_dir = get_annotations_dir()
    rep_dir = get_reports_dir()
    v_dir = get_videos_dir()

    primary_records = load_primary_records()
    secondary_records = load_secondary_records()
    adjudicated_records = {r.sample_id: r for r in load_adjudicated_records()}
    multi_windows_map = load_multi_windows()

    print(f"Loaded {len(primary_records)} primary records.")

    # Gate 1: Check all records are reviewed or excluded
    unreviewed = [r for r in primary_records if not r.is_human_reviewed() and not r.is_excluded()]
    if unreviewed and not force:
        missing_ids = [r.sample_id for r in unreviewed[:5]]
        raise ValueError(
            f"Gate 1 Failed: {len(unreviewed)} samples are unreviewed (DRAFT). "
            f"All samples must be REVIEWED or EXCLUDED before freeze. (e.g. {missing_ids})"
        )

    # Gate 2: Validate reviewed samples
    reviewed = [r for r in primary_records if r.is_human_reviewed()]
    excluded = [r for r in primary_records if r.is_excluded()]

    if not reviewed and not force:
        raise ValueError("Gate 2 Failed: No human-reviewed samples found.")

    for r in reviewed:
        q_ok, q_err = validate_query(r.query, r.query_status)
        if not q_ok and not force:
            raise ValueError(f"Gate 2 Failed for {r.sample_id}: {q_err}")

        # If sample has adjudicated window, prefer it
        effective_windows = r.get_effective_windows()
        if r.sample_id in adjudicated_records:
            adj = adjudicated_records[r.sample_id]
            effective_windows = adj.get_effective_windows()

        if r.sample_id in multi_windows_map:
            effective_windows = multi_windows_map[r.sample_id]

        w_ok, w_err = validate_multi_windows(effective_windows, r.duration_seconds)
        if not w_ok and not force:
            raise ValueError(f"Gate 2 Failed for {r.sample_id}: {w_err}")

    # Gate 3: Check secondary annotation coverage (>=20% of usable samples)
    sec_by_id = {r.sample_id: r for r in secondary_records if r.is_human_reviewed()}
    usable_count = len(reviewed)
    required_sec_count = max(1, int(round(usable_count * min_secondary_fraction)))

    if len(sec_by_id) < required_sec_count and not force:
        raise ValueError(
            f"Gate 3 Failed: Insufficient secondary human annotation coverage. "
            f"Found {len(sec_by_id)} secondary reviews, required at least {required_sec_count} "
            f"({min_secondary_fraction*100:.1f}% of {usable_count} usable samples)."
        )

    # Gate 4: Compute Agreement and verify Adjudication
    dual_comparisons: list[dict[str, Any]] = []
    unresolved_adjudications: list[str] = []
    total_iou = 0.0

    for sid, sec_r in sec_by_id.items():
        prim_r = next((r for r in reviewed if r.sample_id == sid), None)
        if not prim_r:
            continue

        w_prim = prim_r.get_effective_windows()
        w_sec = sec_r.get_effective_windows()
        iou = compute_temporal_iou(w_prim, w_sec)
        total_iou += iou

        is_agreed = iou >= agreement_threshold
        has_adjudication = sid in adjudicated_records

        if not is_agreed and not has_adjudication:
            unresolved_adjudications.append(sid)

        status_str = "AGREED" if is_agreed else ("ADJUDICATED" if has_adjudication else "ADJUDICATION_REQUIRED")

        dual_comparisons.append({
            "sample_id": sid,
            "vid": prim_r.vid,
            "query": prim_r.query,
            "windows_primary": w_prim,
            "windows_secondary": w_sec,
            "tIoU": iou,
            "agreement_status": status_str,
            "has_adjudication": has_adjudication,
        })

    if unresolved_adjudications and not force:
        raise ValueError(
            f"Gate 4 Failed: {len(unresolved_adjudications)} dual-annotation samples have tIoU < {agreement_threshold} "
            f"and lack human adjudication: {unresolved_adjudications}"
        )

    num_dual = len(dual_comparisons)
    mean_iou = round(total_iou / num_dual, 4) if num_dual > 0 else 0.0
    pass_count = sum(1 for c in dual_comparisons if c["agreement_status"] == "AGREED")
    adj_count = sum(1 for c in dual_comparisons if c["has_adjudication"])

    agreement_report = {
        "report_name": "Lumae Ads Human Agreement Report",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dual_sample_count": num_dual,
        "coverage_percentage": round(num_dual / usable_count * 100, 1) if usable_count else 0.0,
        "agreement_threshold": agreement_threshold,
        "mean_tIoU": mean_iou,
        "pass_count": pass_count,
        "adjudicated_count": adj_count,
        "unresolved_count": len(unresolved_adjudications),
        "comparisons": dual_comparisons,
    }

    rep_dir.mkdir(parents=True, exist_ok=True)
    agr_path = rep_dir / "annotation_agreement_human.json"
    agr_path.write_text(json.dumps(agreement_report, indent=2), encoding="utf-8")
    print(f"Saved human agreement report: {agr_path}")

    # Output 1: frozen_gt.csv
    frozen_csv_path = ann_dir / "frozen_gt.csv"
    fieldnames = [
        "sample_id",
        "video_filename",
        "vid",
        "source_dataset",
        "source_video_id",
        "source_split",
        "source_url",
        "product_category",
        "query",
        "duration_seconds",
        "gt_start_seconds",
        "gt_end_seconds",
        "annotator_id",
        "annotation_notes",
        "review_status",
    ]

    frozen_rows: list[dict[str, str]] = []
    canonical_samples: list[CanonicalTemporalSample] = []

    for r in reviewed:
        # Determine final window (prefer adjudicated if present)
        eff_windows = r.get_effective_windows()
        ann_id = r.annotator_id
        notes = r.annotation_notes

        if r.sample_id in adjudicated_records:
            adj = adjudicated_records[r.sample_id]
            eff_windows = adj.get_effective_windows()
            ann_id = "human_adjudicator"
            notes = f"[ADJUDICATED] {adj.annotation_notes}".strip()

        if r.sample_id in multi_windows_map:
            eff_windows = multi_windows_map[r.sample_id]

        row_dict = r.to_csv_dict()
        row_dict["gt_start_seconds"] = f"{eff_windows[0][0]:.1f}"
        row_dict["gt_end_seconds"] = f"{eff_windows[0][1]:.1f}"
        row_dict["annotator_id"] = ann_id
        row_dict["annotation_notes"] = notes
        row_dict["review_status"] = "FROZEN_V1"
        frozen_rows.append(row_dict)

        # Build canonical sample
        meta: dict[str, Any] = {
            "source_dataset": r.source_dataset,
            "source_video_id": r.source_video_id,
            "source_split": r.source_split,
            "source_url": r.source_url,
            "video_filename": r.video_filename,
            "product_category": r.product_category,
            "annotator_id": ann_id,
            "human_annotation_status": "FROZEN_V1",
            "annotation_version": "v1.0",
        }
        if notes:
            meta["annotation_notes"] = notes

        sample = CanonicalTemporalSample(
            qid=r.sample_id,
            vid=r.vid,
            query=r.query,
            duration=r.duration_seconds,
            relevant_windows=eff_windows,
            relevant_clip_ids=None,
            saliency_scores=None,  # STRICT NULL
            source="lumae_product_ads",
            intent="product_demo",
            original_id=r.sample_id,
            metadata=meta,
        )
        canonical_samples.append(sample)

    with frozen_csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in frozen_rows:
            writer.writerow(row)
    print(f"Generated frozen GT CSV: {frozen_csv_path} ({len(frozen_rows)} samples).")

    root = get_repo_root()

    def _rel(p: Path) -> str:
        try:
            return str(p.relative_to(root))
        except ValueError:
            return str(p)

    # Output 2: pilot_v1_human_verified.jsonl
    can_dir = root / "local_data" / "canonical" / "lumae_ads"
    can_dir.mkdir(parents=True, exist_ok=True)
    canonical_jsonl_path = can_dir / "pilot_v1_human_verified.jsonl"
    write_jsonl(canonical_jsonl_path, canonical_samples)
    print(f"Generated canonical frozen JSONL: {canonical_jsonl_path} ({len(canonical_samples)} samples).")

    # Output 3: lumae_ads_pilot_v1.json manifest
    manifest_dir = root / "local_data" / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / "lumae_ads_pilot_v1.json"

    frozen_csv_sha = compute_sha256(frozen_csv_path)
    canonical_jsonl_sha = compute_sha256(canonical_jsonl_path)

    manifest_data = {
        "dataset": "LumaeAdsPilot",
        "version": "v1.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "total_samples": len(primary_records),
        "usable_sample_count": usable_count,
        "excluded_count": len(excluded),
        "annotation_metrics": {
            "primary_human_coverage": round(usable_count / len(primary_records), 4) if primary_records else 0.0,
            "secondary_human_coverage": round(num_dual / usable_count, 4) if usable_count else 0.0,
            "agreement_threshold_tiou": agreement_threshold,
            "mean_human_human_tiou": mean_iou,
            "adjudicated_count": adj_count,
        },
        "artifacts": {
            "frozen_gt_csv": {
                "path": _rel(frozen_csv_path),
                "sha256": frozen_csv_sha,
                "samples_count": len(frozen_rows),
            },
            "canonical_jsonl": {
                "path": _rel(canonical_jsonl_path),
                "sha256": canonical_jsonl_sha,
                "samples_count": len(canonical_samples),
            },
        },
        "provenance": {
            "raw_video_dir": "local_data/raw/adsqa/videos",
            "audit_log": "local_data/annotations/lumae_ads/human_review_log.jsonl",
            "agreement_report": "local_data/reports/lumae_ads/annotation_agreement_human.json",
        },
        "ground_truth": {
            "status": "FROZEN",
            "policy": "Immutable ground truth. Model predictions may never alter GT.",
        },
    }

    manifest_path.write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")
    print(f"Generated frozen manifest: {manifest_path}")

    return manifest_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Enforce quality gates and freeze Lumae Ads Pilot human GT.")
    parser.add_argument("--force", action="store_true", help="Bypass completion gates for testing preview")
    parser.add_argument("--min-secondary", type=float, default=0.20, help="Minimum secondary annotation ratio (default: 0.20)")
    parser.add_argument("--threshold", type=float, default=0.70, help="tIoU agreement threshold (default: 0.70)")
    args = parser.parse_args()

    try:
        freeze_lumae_ads_pilot(
            force=args.force,
            min_secondary_fraction=args.min_secondary,
            agreement_threshold=args.threshold,
        )
    except Exception as exc:
        print(f"Freeze failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
