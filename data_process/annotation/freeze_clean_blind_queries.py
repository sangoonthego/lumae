"""Script to validate and freeze human-verified clean-blind queries.

Stage A.2.5I Protocol:
1. Load selected clean-blind 5 from semantic_v3_clean_query_blind5_selection.json.
2. Load query reviews from local_data/annotations/lumae_ads/query_reviews.csv.
3. Verify all 5 samples satisfy:
   - human_final_query is non-empty
   - query_localizable == True
   - query_review_status in {"VALID_AS_IS", "HUMAN_EDITED"}
   - reviewer_id.startswith("human_")
4. If incomplete: abort cleanly and return status "WAITING_FOR_HUMAN_QUERY_REVIEW".
5. If complete:
   - Create local_data/annotations/lumae_ads/blind_eval/semantic_v3_clean_query_blind5_queries_frozen.csv
   - Calculate SHA256 of frozen queries and query_reviews.csv
   - Create local_data/manifests/semantic_v3_clean_query_blind5_query_freeze.json
   - Return status "CLEAN_BLIND_QUERIES_FROZEN_BEFORE_V3_INFERENCE"
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Dict

from data_process.annotation.query_sanitation import (
    get_annotations_dir,
    get_manifests_dir,
    get_query_reviews_path,
    load_query_reviews,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def get_selection_manifest_path() -> Path:
    return get_manifests_dir() / "semantic_v3_clean_query_blind5_selection.json"


def check_and_freeze_clean_blind_queries() -> Dict[str, Any]:
    """Check human query reviews and freeze if complete, otherwise return WAITING_FOR_HUMAN_QUERY_REVIEW."""
    sel_path = get_selection_manifest_path()
    assert sel_path.is_file(), f"Missing selection manifest: {sel_path}"
    with open(sel_path, "r", encoding="utf-8") as f:
        sel_data = json.load(f)

    target_sample_ids = sel_data["selected_sample_ids"]
    assert len(target_sample_ids) == 5, f"Expected 5 samples, found {len(target_sample_ids)}"

    reviews = load_query_reviews()

    # Check review completion for all 5 target samples
    missing_or_incomplete = []
    for sid in target_sample_ids:
        if sid not in reviews:
            missing_or_incomplete.append((sid, "NOT_IN_QUERY_REVIEWS"))
            continue
        rec = reviews[sid]
        if not rec.human_final_query or len(rec.human_final_query.strip()) < 5:
            missing_or_incomplete.append((sid, "EMPTY_OR_SHORT_FINAL_QUERY"))
            continue
        if not rec.query_localizable:
            missing_or_incomplete.append((sid, "QUERY_NOT_LOCALIZABLE"))
            continue
        if rec.query_review_status not in ("VALID_AS_IS", "HUMAN_EDITED"):
            missing_or_incomplete.append((sid, f"INVALID_STATUS_{rec.query_review_status}"))
            continue
        if not rec.reviewer_id.startswith("human_"):
            missing_or_incomplete.append((sid, f"INVALID_REVIEWER_{rec.reviewer_id}"))
            continue

    if missing_or_incomplete:
        return {
            "status": "WAITING_FOR_HUMAN_QUERY_REVIEW",
            "message": "Human query review is not complete for all 5 selected clean-blind samples.",
            "target_samples": target_sample_ids,
            "incomplete_details": missing_or_incomplete,
        }

    # All 5 are genuinely verified by a human! Proceed to freeze.
    qr_path = get_query_reviews_path()
    qr_sha256 = hashlib.sha256(qr_path.read_bytes()).hexdigest()

    frozen_dir = get_annotations_dir() / "blind_eval"
    frozen_dir.mkdir(parents=True, exist_ok=True)
    frozen_csv_path = frozen_dir / "semantic_v3_clean_query_blind5_queries_frozen.csv"

    frozen_fields = [
        "sample_id",
        "video_filename",
        "source_dataset",
        "source_video_id",
        "original_query",
        "human_final_query",
        "original_query_quality",
        "query_review_status",
        "query_change_type",
        "query_localizable",
        "query_observable",
        "reviewer_id",
        "review_notes",
        "query_version",
    ]

    rows = []
    for sid in sorted(target_sample_ids):
        r = reviews[sid]
        rows.append({
            "sample_id": r.sample_id,
            "video_filename": r.video_filename,
            "source_dataset": r.source_dataset,
            "source_video_id": r.source_video_id,
            "original_query": r.original_query,
            "human_final_query": r.human_final_query,
            "original_query_quality": r.original_query_quality,
            "query_review_status": r.query_review_status,
            "query_change_type": r.query_change_type,
            "query_localizable": "TRUE" if r.query_localizable is True or str(r.query_localizable).upper() == "TRUE" else "FALSE",
            "query_observable": "TRUE" if r.query_observable is True or str(r.query_observable).upper() == "TRUE" else "FALSE",
            "reviewer_id": r.reviewer_id,
            "review_notes": r.review_notes,
            "query_version": r.query_version,
        })

    with open(frozen_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=frozen_fields)
        writer.writeheader()
        writer.writerows(rows)

    frozen_bytes = frozen_csv_path.read_bytes()
    query_sha256 = hashlib.sha256(frozen_bytes).hexdigest()
    sel_sha256 = hashlib.sha256(sel_path.read_bytes()).hexdigest()
    reviewers = sorted(list({r["reviewer_id"] for r in rows if r["reviewer_id"]}))

    freeze_manifest = {
        "experiment": "semantic_v3_clean_query_blind5",
        "sample_ids": sorted(target_sample_ids),
        "query_file": "local_data/annotations/lumae_ads/blind_eval/semantic_v3_clean_query_blind5_queries_frozen.csv",
        "query_file_sha256": query_sha256,
        "source_query_reviews_sha256": qr_sha256,
        "selection_manifest_sha256": sel_sha256,
        "freeze_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "reviewer_count": len(reviewers),
        "status": "CLEAN_BLIND_QUERIES_FROZEN_BEFORE_V3_INFERENCE",
    }

    freeze_manifest_path = get_manifests_dir() / "semantic_v3_clean_query_blind5_query_freeze.json"
    with open(freeze_manifest_path, "w", encoding="utf-8") as f:
        json.dump(freeze_manifest, f, indent=2)

    return {
        "status": "CLEAN_BLIND_QUERIES_FROZEN_BEFORE_V3_INFERENCE",
        "manifest": freeze_manifest,
    }


if __name__ == "__main__":
    result = check_and_freeze_clean_blind_queries()
    print("STATUS:", result["status"])
    if result["status"] == "WAITING_FOR_HUMAN_QUERY_REVIEW":
        print("Incomplete details:", result.get("incomplete_details"))
        sys.exit(0)
    else:
        print("Frozen manifest written:", result.get("manifest"))
