"""Stage A.2.5I: Query Sanitation, Temporal Exposure Audit, and Clean-Blind Sample Selection.

Modules:
1. Exposure Registry Generator: Audits historical temporal exposure across QVHighlights / AdsQA pilot.
2. Deterministic Clean-Blind 5 Selector: Selects 5 uncontaminated pending samples using seed 20260930.
3. Sanitation Worklist Generator: Builds local_data/annotations/lumae_ads/query_sanitation_worklist.csv.
4. Query Reviews Storage: Atomic reading and writing of query_reviews.csv.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile
from typing import Any, Dict, List, Set

import pandas as pd

from data_process.annotation.models import (
    OriginalQueryQuality,
    QueryReviewRecord,
    QueryReviewStatus,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Known historical exposure groups
DEVELOPMENT_SAMPLES: Set[str] = {
    "lumae_ads_pilot_0004",
    "lumae_ads_pilot_0006",
    "lumae_ads_pilot_0007",
    "lumae_ads_pilot_0008",
    "lumae_ads_pilot_0010",
    "lumae_ads_pilot_0011",
    "lumae_ads_pilot_0012",
    "lumae_ads_pilot_0013",
}

CLEAN3_BLIND_SAMPLES: Set[str] = {
    "lumae_ads_pilot_0008",
    "lumae_ads_pilot_0010",
    "lumae_ads_pilot_0011",
}

FRESH5_A25H_SAMPLES: Set[str] = {
    "lumae_ads_pilot_0017",
    "lumae_ads_pilot_0018",
    "lumae_ads_pilot_0020",
    "lumae_ads_pilot_0037",
    "lumae_ads_pilot_0043",
}

LEGACY_SEED42_HOLDOUT_SAMPLES: Set[str] = {
    "lumae_ads_pilot_0002",
    "lumae_ads_pilot_0008",
    "lumae_ads_pilot_0009",
    "lumae_ads_pilot_0010",
    "lumae_ads_pilot_0011",
    "lumae_ads_pilot_0017",
    "lumae_ads_pilot_0018",
    "lumae_ads_pilot_0020",
    "lumae_ads_pilot_0037",
    "lumae_ads_pilot_0043",
}

SELECTION_SEED = 20260930


def get_annotations_dir() -> Path:
    return REPO_ROOT / "local_data" / "annotations" / "lumae_ads"


def get_manifests_dir() -> Path:
    return REPO_ROOT / "local_data" / "manifests"


def get_query_reviews_path() -> Path:
    return get_annotations_dir() / "query_reviews.csv"


def get_query_worklist_path() -> Path:
    return get_annotations_dir() / "query_sanitation_worklist.csv"


def generate_exposure_registry() -> Dict[str, Any]:
    """Audit all 48 pilot samples and generate temporal_exposure_registry.json."""
    hp_path = get_annotations_dir() / "human_primary.csv"
    assert hp_path.is_file(), f"Missing {hp_path}"
    df_hp = pd.read_csv(hp_path)

    reviewed_ids = set(df_hp[df_hp["review_status"] == "REVIEWED"]["sample_id"])

    # Categorize all 48 samples
    per_sample_exposure: Dict[str, List[str]] = {}
    excluded_for_clean_blind: Dict[str, str] = {}
    eligible_for_clean_blind: List[str] = []

    for sid in sorted(df_hp["sample_id"].unique()):
        flags = []
        if sid in DEVELOPMENT_SAMPLES:
            flags.append("DEVELOPMENT_EXPOSED")
        if sid in CLEAN3_BLIND_SAMPLES or sid in FRESH5_A25H_SAMPLES:
            flags.append("BLIND_EVALUATED")
        if sid in LEGACY_SEED42_HOLDOUT_SAMPLES:
            flags.append("LEGACY_HOLDOUT")
        if sid in reviewed_ids:
            flags.append("TEMPORAL_GT_REVIEWED")
        if sid in DEVELOPMENT_SAMPLES or sid in FRESH5_A25H_SAMPLES:
            flags.append("REPORT_TIMESTAMP_EXPOSED")

        per_sample_exposure[sid] = flags

        # Exclusion reason
        if sid in DEVELOPMENT_SAMPLES:
            excluded_for_clean_blind[sid] = "Development set sample used for semantic_v3 training/tuning"
        elif sid in FRESH5_A25H_SAMPLES:
            excluded_for_clean_blind[sid] = "Fresh5 sample evaluated in Stage A.2.5H"
        elif sid in LEGACY_SEED42_HOLDOUT_SAMPLES:
            excluded_for_clean_blind[sid] = "Legacy seed-42 holdout (contamination risk: 0002/0009)"
        elif sid in reviewed_ids:
            excluded_for_clean_blind[sid] = "Human temporal ground truth already reviewed"
        else:
            eligible_for_clean_blind.append(sid)

    registry = {
        "registry_name": "lumae_ads_temporal_exposure_registry",
        "total_pilot_samples": len(df_hp),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "exposure_summary": {
            "DEVELOPMENT_EXPOSED": sorted(list(DEVELOPMENT_SAMPLES)),
            "BLIND_EVALUATED": sorted(list(CLEAN3_BLIND_SAMPLES | FRESH5_A25H_SAMPLES)),
            "LEGACY_HOLDOUT": sorted(list(LEGACY_SEED42_HOLDOUT_SAMPLES)),
            "TEMPORAL_GT_REVIEWED": sorted(list(reviewed_ids)),
            "REPORT_TIMESTAMP_EXPOSED": sorted(list(DEVELOPMENT_SAMPLES | FRESH5_A25H_SAMPLES)),
        },
        "clean_blind_eligibility": {
            "eligible_count": len(eligible_for_clean_blind),
            "excluded_count": len(excluded_for_clean_blind),
            "eligible_sample_ids": sorted(eligible_for_clean_blind),
            "excluded_sample_ids": sorted(list(excluded_for_clean_blind.keys())),
            "exclusion_reasons": excluded_for_clean_blind,
        },
        "per_sample_exposure": per_sample_exposure,
    }

    manifests_dir = get_manifests_dir()
    manifests_dir.mkdir(parents=True, exist_ok=True)
    out_file = manifests_dir / "temporal_exposure_registry.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)

    return registry


def select_clean_blind5(seed: int = SELECTION_SEED) -> Dict[str, Any]:
    """Deterministically select 5 clean-blind samples from eligible pending population."""
    registry = generate_exposure_registry()
    eligible = sorted(registry["clean_blind_eligibility"]["eligible_sample_ids"])
    assert len(eligible) >= 5, f"Not enough eligible samples: {len(eligible)}"

    rng = random.Random(seed)
    selected_5 = sorted(rng.sample(eligible, 5))

    selection_manifest = {
        "experiment": "semantic_v3_clean_query_blind5",
        "selection_seed": seed,
        "eligible_sample_count": len(eligible),
        "excluded_sample_ids": sorted(list(registry["clean_blind_eligibility"]["excluded_sample_ids"])),
        "exclusion_reasons": registry["clean_blind_eligibility"]["exclusion_reasons"],
        "selected_sample_ids": selected_5,
        "selection_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "CLEAN_BLIND5_SELECTED_BEFORE_QUERY_REVIEW",
    }

    manifest_path = get_manifests_dir() / "semantic_v3_clean_query_blind5_selection.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(selection_manifest, f, indent=2)

    return selection_manifest


def generate_query_sanitation_worklist() -> List[Dict[str, Any]]:
    """Build query_sanitation_worklist.csv covering all 35 pending pilot samples."""
    hp_path = get_annotations_dir() / "human_primary.csv"
    df_hp = pd.read_csv(hp_path)

    # Load clean-blind selection
    sel_manifest_path = get_manifests_dir() / "semantic_v3_clean_query_blind5_selection.json"
    if not sel_manifest_path.is_file():
        select_clean_blind5()
    with open(sel_manifest_path, "r", encoding="utf-8") as f:
        sel_data = json.load(f)
    reserved_blind5 = set(sel_data["selected_sample_ids"])

    pending_df = df_hp[df_hp["review_status"] == "DRAFT"].copy()
    assert len(pending_df) == 35, f"Expected 35 pending samples, found {len(pending_df)}"

    worklist_rows = []
    for _, r in pending_df.iterrows():
        sid = str(r["sample_id"])
        is_reserved = sid in reserved_blind5
        worklist_rows.append({
            "sample_id": sid,
            "video_filename": str(r["video_filename"]),
            "original_query": str(r["query"]),
            "source_video_id": str(r["source_video_id"]),
            "duration_seconds": f"{float(r['duration_seconds']):.2f}",
            "query_review_status": QueryReviewStatus.PENDING_QUERY_REVIEW.value,
            "clean_blind_reserved": "TRUE" if is_reserved else "FALSE",
            "temporal_annotation_locked": "TRUE",
        })

    worklist_csv = get_query_worklist_path()
    fieldnames = [
        "sample_id",
        "video_filename",
        "original_query",
        "source_video_id",
        "duration_seconds",
        "query_review_status",
        "clean_blind_reserved",
        "temporal_annotation_locked",
    ]

    with open(worklist_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(worklist_rows)

    return worklist_rows


def load_query_reviews() -> Dict[str, QueryReviewRecord]:
    """Load query_reviews.csv into a dictionary keyed by sample_id."""
    path = get_query_reviews_path()
    if not path.is_file():
        return {}

    records = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rec = QueryReviewRecord.from_csv_dict(row)
            records[rec.sample_id] = rec
    return records


def save_query_review(record: QueryReviewRecord) -> None:
    """Save or update a query review record atomically in query_reviews.csv."""
    reviews = load_query_reviews()
    reviews[record.sample_id] = record

    fieldnames = [
        "sample_id",
        "video_filename",
        "original_query",
        "human_final_query",
        "original_query_quality",
        "query_review_status",
        "query_localizable",
        "reviewer_id",
        "review_notes",
        "reviewed_at_utc",
        "query_version",
    ]

    out_path = get_query_reviews_path()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", delete=False, dir=out_path.parent) as tf:
        writer = csv.DictWriter(tf, fieldnames=fieldnames)
        writer.writeheader()
        for sid in sorted(reviews.keys()):
            r = reviews[sid]
            writer.writerow({
                "sample_id": r.sample_id,
                "video_filename": r.video_filename,
                "original_query": r.original_query,
                "human_final_query": r.human_final_query,
                "original_query_quality": r.original_query_quality,
                "query_review_status": r.query_review_status,
                "query_localizable": str(r.query_localizable),
                "reviewer_id": r.reviewer_id,
                "review_notes": r.review_notes,
                "reviewed_at_utc": r.reviewed_at_utc or datetime.now(timezone.utc).isoformat(),
                "query_version": r.query_version,
            })
        temp_name = tf.name

    os.replace(temp_name, out_path)
