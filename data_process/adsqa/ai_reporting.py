"""Reporting utilities for AI Pre-annotation vs Human Ground Truth.

Stage A.2.5B Protocol:
1. AI vs Blind Human Agreement (Top-1 tIoU, boundary absolute error, query agreement).
   CRITICAL DISTINCTION: This is NOT human-human agreement.
2. AI Assistance Acceptance Rate Report (AI query accepted/edited/rejected %, AI window accepted/edited/rejected %).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from data_process.adsqa.annotation_qc import compute_temporal_iou
from data_process.annotation.models import QueryAction, ReviewStatus, WindowAction
from data_process.annotation.storage import (
    get_ai_preannotations_path,
    get_annotations_dir,
    get_blind_holdout_path,
    get_reports_dir,
    load_ai_preannotations,
    load_blind_holdout,
    load_primary_records,
)


def generate_ai_vs_blind_human_report(
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Calculate agreement metrics between AI candidates and blind human ground truth.

    Only evaluates samples belonging to blind_holdout.json that have been reviewed by a human.
    """
    holdout_info = load_blind_holdout()
    holdout_sample_ids = set(holdout_info.get("sample_ids", []))

    primary_records = {r.sample_id: r for r in load_primary_records()}
    ai_candidates = load_ai_preannotations()

    if output_path is None:
        output_path = get_reports_dir() / "ai_vs_blind_human.json"

    evaluated_samples: list[dict[str, Any]] = []
    tious: list[float] = []
    start_errors: list[float] = []
    end_errors: list[float] = []
    query_matches = 0

    for sid in sorted(holdout_sample_ids):
        prim = primary_records.get(sid)
        cand = ai_candidates.get(sid)

        if not prim or not cand:
            continue

        # Check if human has actually reviewed this sample
        if not prim.is_human_reviewed():
            continue

        human_windows = prim.get_effective_windows()
        ai_windows = cand.get_windows()

        if not human_windows or not ai_windows:
            continue

        # Compute temporal IoU
        tiou = compute_temporal_iou(human_windows, ai_windows)
        tious.append(tiou)

        # Boundary absolute error on primary window
        h_s, h_e = human_windows[0][0], human_windows[0][1]
        a_s, a_e = ai_windows[0][0], ai_windows[0][1]

        s_err = abs(h_s - a_s)
        e_err = abs(h_e - a_e)
        start_errors.append(s_err)
        end_errors.append(e_err)

        # Query agreement (semantic match or exact normalized match)
        q_agree = prim.query.strip().lower() == (cand.ai_proposed_query or cand.original_query).strip().lower()
        if q_agree:
            query_matches += 1

        evaluated_samples.append({
            "sample_id": sid,
            "video_filename": prim.video_filename,
            "human_query": prim.query,
            "ai_query": cand.ai_proposed_query or cand.original_query,
            "query_agreement": q_agree,
            "human_windows": human_windows,
            "ai_windows": ai_windows,
            "tiou": round(tiou, 4),
            "start_error_seconds": round(s_err, 2),
            "end_error_seconds": round(e_err, 2),
            "ai_confidence": cand.ai_confidence,
        })

    n = len(evaluated_samples)
    report = {
        "report_type": "AI vs Blind Human Agreement",
        "description": "Scientific comparison of AI pre-annotation candidate vs independent blind human ground truth.",
        "note_on_provenance": "CRITICAL: This is an AI-human agreement metric, NEVER to be confused with human-human agreement.",
        "sample_count": n,
        "total_holdout_count": len(holdout_sample_ids),
        "reviewed_holdout_count": n,
        "mean_tiou": round(float(np.mean(tious)), 4) if tious else 0.0,
        "median_tiou": round(float(np.median(tious)), 4) if tious else 0.0,
        "min_tiou": round(float(np.min(tious)), 4) if tious else 0.0,
        "max_tiou": round(float(np.max(tious)), 4) if tious else 0.0,
        "mean_start_error_seconds": round(float(np.mean(start_errors)), 2) if start_errors else 0.0,
        "mean_end_error_seconds": round(float(np.mean(end_errors)), 2) if end_errors else 0.0,
        "query_agreement_rate": round(query_matches / n, 4) if n > 0 else 0.0,
        "evaluated_samples": evaluated_samples,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report


def generate_ai_assistance_report(
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Calculate operator acceptance rate for AI-assisted human reviews."""
    log_path = get_annotations_dir() / "human_review_log.jsonl"
    if output_path is None:
        output_path = get_reports_dir() / "ai_assistance_report.json"

    entries: list[dict[str, Any]] = []
    if log_path.is_file():
        with log_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except Exception:
                    continue

    # Filter for AI-assisted human reviews
    assisted = [
        e for e in entries
        if e.get("annotation_mode") in ("AI_ASSISTED_PRIMARY", "AI_ASSISTED_HUMAN_VERIFIED")
        or e.get("query_action") is not None
        or e.get("window_action") is not None
    ]

    total_assisted = len(assisted)
    query_accepted = sum(1 for e in assisted if e.get("query_action") == QueryAction.ACCEPTED_AI.value)
    query_edited = sum(1 for e in assisted if e.get("query_action") == QueryAction.EDITED_AI.value)
    query_rejected = sum(1 for e in assisted if e.get("query_action") == QueryAction.REJECTED_AI.value)

    window_accepted = sum(1 for e in assisted if e.get("window_action") == WindowAction.ACCEPTED_AI.value)
    window_edited = sum(1 for e in assisted if e.get("window_action") == WindowAction.EDITED_AI.value)
    window_rejected = sum(1 for e in assisted if e.get("window_action") == WindowAction.REJECTED_AI.value)

    def pct(count: int, total: int) -> float:
        return round((count / total) * 100.0, 2) if total > 0 else 0.0

    report = {
        "report_type": "AI Assistance Operator Acceptance Report",
        "description": "Quantifies human reviewer acceptance, modification, and rejection of AI candidate proposals.",
        "total_assisted_reviews": total_assisted,
        "query_actions": {
            "accepted_unchanged_count": query_accepted,
            "accepted_unchanged_pct": pct(query_accepted, total_assisted),
            "edited_count": query_edited,
            "edited_pct": pct(query_edited, total_assisted),
            "rejected_count": query_rejected,
            "rejected_pct": pct(query_rejected, total_assisted),
        },
        "window_actions": {
            "accepted_unchanged_count": window_accepted,
            "accepted_unchanged_pct": pct(window_accepted, total_assisted),
            "edited_count": window_edited,
            "edited_pct": pct(window_edited, total_assisted),
            "rejected_count": window_rejected,
            "rejected_pct": pct(window_rejected, total_assisted),
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report
