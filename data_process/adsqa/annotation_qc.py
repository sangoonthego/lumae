"""Quality control and multi-annotator agreement calculations for Lumae Ads."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def interval_length(intervals: list[list[float]]) -> float:
    """Compute total non-overlapping duration of interval list."""
    if not intervals:
        return 0.0
    sorted_int = sorted(intervals, key=lambda x: x[0])
    merged: list[list[float]] = []
    for s, e in sorted_int:
        if not merged:
            merged.append([s, e])
        else:
            prev = merged[-1]
            if s <= prev[1]:
                prev[1] = max(prev[1], e)
            else:
                merged.append([s, e])
    return sum(e - s for s, e in merged)


def compute_temporal_iou(
    windows_a: list[list[float]],
    windows_b: list[list[float]],
) -> float:
    """Compute pairwise temporal IoU between two sets of intervals.

    Supports single or multiple discontinuous windows.
    tIoU = |A ∩ B| / |A ∪ B|.
    """
    if not windows_a and not windows_b:
        return 1.0
    if not windows_a or not windows_b:
        return 0.0

    # Calculate intersection
    intersection_total = 0.0
    for s1, e1 in windows_a:
        for s2, e2 in windows_b:
            inter_s = max(s1, s2)
            inter_e = min(e1, e2)
            if inter_e > inter_s:
                intersection_total += inter_e - inter_s

    len_a = interval_length(windows_a)
    len_b = interval_length(windows_b)
    union_total = len_a + len_b - intersection_total

    if union_total <= 0:
        return 0.0

    iou = intersection_total / union_total
    return round(iou, 4)


def evaluate_dual_annotations(
    primary_annotations: list[dict[str, Any]],
    secondary_annotations: list[dict[str, Any]],
    acceptance_threshold: float = 0.70,
    output_report_path: Path | str | None = None,
) -> dict[str, Any]:
    """Compare primary and secondary annotations and compute agreement statistics."""
    sec_by_sample = {s["sample_id"]: s for s in secondary_annotations}

    compared: list[dict[str, Any]] = []
    high_agreement = 0
    adjudication_needed = 0
    total_iou = 0.0

    for prim in primary_annotations:
        sid = prim["sample_id"]
        sec = sec_by_sample.get(sid)
        if not sec:
            continue

        w_a = prim.get("relevant_windows") or [[prim["gt_start_seconds"], prim["gt_end_seconds"]]]
        w_b = sec.get("relevant_windows") or [[sec["gt_start_seconds"], sec["gt_end_seconds"]]]

        # Ensure floats
        w_a_clean = [[float(w[0]), float(w[1])] for w in w_a]
        w_b_clean = [[float(w[0]), float(w[1])] for w in w_b]

        iou = compute_temporal_iou(w_a_clean, w_b_clean)
        total_iou += iou

        if iou >= acceptance_threshold:
            status = "AGREED"
            high_agreement += 1
        else:
            status = "ADJUDICATION_REQUIRED"
            adjudication_needed += 1

        compared.append({
            "sample_id": sid,
            "vid": prim.get("vid", sid),
            "annotator_primary": prim.get("annotator_id", "primary"),
            "annotator_secondary": sec.get("annotator_id", "secondary"),
            "windows_primary": w_a_clean,
            "windows_secondary": w_b_clean,
            "tIoU": iou,
            "agreement_status": status,
            "requires_adjudication": (iou < acceptance_threshold),
        })

    num_compared = len(compared)
    mean_iou = round(total_iou / num_compared, 4) if num_compared > 0 else 0.0
    double_pct = (
        round(num_compared / len(primary_annotations) * 100, 1)
        if primary_annotations
        else 0.0
    )

    report = {
        "total_primary_samples": len(primary_annotations),
        "double_annotated_count": num_compared,
        "double_annotated_percentage": double_pct,
        "acceptance_threshold": acceptance_threshold,
        "mean_tIoU": mean_iou,
        "high_agreement_count": high_agreement,
        "adjudication_required_count": adjudication_needed,
        "qc_policy": "tIoU >= 0.70 required for consensus; no blind averaging",
        "dual_annotation_records": compared,
    }

    if output_report_path:
        out_p = Path(output_report_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Saved annotation agreement report to: {out_p}")

    return report
