"""Stage A.2.5I: Query Sanitation, Temporal Exposure Audit, and Clean-Blind Sample Selection.

Modules:
1. Exposure Registry Generator: Audits historical temporal exposure across QVHighlights / AdsQA pilot.
2. Deterministic Clean-Blind 5 Selector: Selects 5 uncontaminated pending samples using seed 20260930.
3. Sanitation Worklist Generator: Builds local_data/annotations/lumae_ads/query_sanitation_worklist.csv.
4. Query Reviews Storage: Atomic reading and writing of query_reviews.csv.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys
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
    if os.environ.get("LUMAE_ANNOTATIONS_DIR"):
        return Path(os.environ["LUMAE_ANNOTATIONS_DIR"])
    return REPO_ROOT / "local_data" / "annotations" / "lumae_ads"


def get_manifests_dir() -> Path:
    if os.environ.get("LUMAE_MANIFESTS_DIR"):
        return Path(os.environ["LUMAE_MANIFESTS_DIR"])
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
    if (get_manifests_dir() / "semantic_v3_clean_query_blind5_query_freeze.json").exists():
        raise RuntimeError("Clean-blind selection is frozen; refusing to regenerate its manifest")
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
    """Build a query worklist from the pilot's current DRAFT rows."""
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


# Canonical Query Review Schema (17 fields)
CANONICAL_QUERY_REVIEW_FIELDS: List[str] = [
    "sample_id",
    "video_filename",
    "source_dataset",
    "source_video_id",
    "original_query",
    "ai_suggested_query",
    "human_final_query",
    "query_review_status",
    "original_query_quality",
    "query_change_type",
    "query_localizable",
    "query_observable",
    "reviewer_id",
    "review_notes",
    "reviewed_at_utc",
    "query_version",
    "temporal_annotation_locked",
]

FORBIDDEN_TEMPORAL_FIELDS: Set[str] = {
    "gt_start_seconds",
    "gt_end_seconds",
}

LEGACY_PILOT_FIELDS: Set[str] = {
    "gt_start_seconds",
    "gt_end_seconds",
    "vid",
    "source_split",
    "source_url",
    "product_category",
    "query",
    "duration_seconds",
    "annotator_id",
    "annotation_notes",
    "review_status",
}

CONFIRMED_HUMAN_REVIEW_0028: Dict[str, Any] = {
    "sample_id": "lumae_ads_pilot_0028",
    "original_query_quality": "VAGUE_BUT_RELEVANT",
    "human_final_query": "A hand turns the washing machine temperature dial to the cold setting.",
    "query_review_status": "HUMAN_EDITED",
    "query_change_type": "NARROWED_INTENT",
    "query_localizable": True,
    "query_observable": True,
    "reviewer_id": "human_01",
    "review_notes": (
        "Human query-only blind review: original query was relevant but too generic; "
        "rewritten to a concrete visible washing-machine temperature adjustment event."
    ),
    "query_version": "1",
    "temporal_annotation_locked": True,
}


class QueryReviewSchemaMismatchError(ValueError):
    """Raised when query_reviews.csv does not conform to the canonical schema."""

    pass


def validate_query_review_schema(target: Path | str | List[str]) -> None:
    """Strictly validate that a query review file or header list conforms to the canonical schema.

    Raises:
        QueryReviewSchemaMismatchError: If required columns are missing, forbidden temporal
        columns are present, or unexpected legacy columns are present.
    """
    if isinstance(target, (Path, str)):
        p = Path(target)
        if not p.is_file():
            raise FileNotFoundError(f"Query review file not found: {p}")
        with p.open("r", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            headers = next(reader, [])
    else:
        headers = list(target)

    actual_columns = [h.strip().lstrip("\ufeff") for h in headers if h.strip()]
    missing_required = [f for f in CANONICAL_QUERY_REVIEW_FIELDS if f not in actual_columns]
    forbidden_present = [f for f in actual_columns if f in FORBIDDEN_TEMPORAL_FIELDS]
    unexpected_legacy = [
        f for f in actual_columns if f in LEGACY_PILOT_FIELDS and f not in CANONICAL_QUERY_REVIEW_FIELDS
    ]
    all_unexpected = sorted(list(set(forbidden_present) | set(unexpected_legacy)))

    if missing_required or forbidden_present or unexpected_legacy:
        msg = (
            "QUERY_REVIEW_SCHEMA_MISMATCH:\n"
            f"  actual columns: {actual_columns}\n"
            f"  missing required columns: {missing_required}\n"
            f"  unexpected legacy columns: {all_unexpected}\n"
            "  recommended repair command: python -m data_process.annotation.query_sanitation init"
        )
        raise QueryReviewSchemaMismatchError(msg)


def atomic_write_query_reviews(path: Path, records: List[QueryReviewRecord]) -> None:
    """Atomically write QueryReviewRecord instances to path using tempfile + fsync + os.replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", delete=False, dir=path.parent) as tf:
        temp_path = Path(tf.name)
        writer = csv.DictWriter(tf, fieldnames=CANONICAL_QUERY_REVIEW_FIELDS)
        writer.writeheader()
        for rec in sorted(records, key=lambda r: r.sample_id):
            writer.writerow(rec.to_csv_dict())
        tf.flush()
        try:
            os.fsync(tf.fileno())
        except OSError:
            pass

    os.replace(temp_path, path)


def load_query_reviews(path: Path | None = None) -> Dict[str, QueryReviewRecord]:
    """Load query_reviews.csv into a dictionary keyed by sample_id with strict schema validation."""
    if path is None:
        path = get_query_reviews_path()
    if not path.is_file():
        return {}

    validate_query_review_schema(path)

    records = {}
    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Clean any possible BOM remnants from keys
            clean_row = {k.strip().lstrip("\ufeff"): v for k, v in row.items() if k is not None}
            rec = QueryReviewRecord.from_csv_dict(clean_row)
            records[rec.sample_id] = rec
    return records


def save_query_review(record: QueryReviewRecord, path: Path | None = None) -> None:
    """Save or update a query review record atomically in query_reviews.csv."""
    if record.query_review_status in {QueryReviewStatus.VALID_AS_IS.value, QueryReviewStatus.HUMAN_EDITED.value}:
        if not (record.reviewer_id.startswith("human_") and record.human_final_query.strip()
                and record.query_localizable is True and record.query_observable is True):
            raise ValueError("Approved queries require a human reviewer, final query, and both observable/localizable checks")
    if path is None:
        path = get_query_reviews_path()

    reviews: Dict[str, QueryReviewRecord] = {}
    if path.is_file():
        reviews = load_query_reviews(path)

    reviews[record.sample_id] = record
    atomic_write_query_reviews(path, list(reviews.values()))


def init_or_migrate_query_reviews(
    dest_path: Path | None = None,
    worklist_path: Path | None = None,
    backup_path: Path | None = None,
    restore_0028: bool = True,
) -> Dict[str, Any]:
    """Rebuild query_reviews.csv safely from canonical sanitation worklist.

    1. Audits existing destination file:
       - If exists and has legacy/bad schema:
         - Backs up to backup_path (default: query_reviews.bad_schema.backup.csv).
         - Verifies backup SHA256 matches original SHA256.
         - Refuses to overwrite existing backup if hash differs.
    2. Loads canonical metadata from worklist_path (query_sanitation_worklist.csv).
    3. Initializes canonical query-only records.
    4. Restores confirmed human review for lumae_ads_pilot_0028 if restore_0028 is True.
    5. Preserves existing valid human reviews if dest was already canonical.
    6. Writes atomically via atomic_write_query_reviews.
    7. Validates final output with validate_query_review_schema.
    """
    if dest_path is None:
        dest_path = get_query_reviews_path()
    if worklist_path is None:
        worklist_path = get_query_worklist_path()
    if backup_path is None:
        backup_path = dest_path.parent / "query_reviews.bad_schema.backup.csv"

    # Step 1: Audit existing destination
    original_sha256 = None
    backup_sha256 = None
    backup_created = False
    is_legacy = False

    if dest_path.is_file():
        original_bytes = dest_path.read_bytes()
        original_sha256 = hashlib.sha256(original_bytes).hexdigest()
        try:
            validate_query_review_schema(dest_path)
            is_legacy = False
        except QueryReviewSchemaMismatchError:
            is_legacy = True

        if is_legacy:
            if backup_path.exists():
                existing_backup_bytes = backup_path.read_bytes()
                existing_backup_sha256 = hashlib.sha256(existing_backup_bytes).hexdigest()
                if existing_backup_sha256 != original_sha256:
                    raise RuntimeError(
                        f"Refusing to overwrite existing backup with differing hash!\n"
                        f"Existing backup SHA256: {existing_backup_sha256}\n"
                        f"Current file SHA256:    {original_sha256}"
                    )
                backup_sha256 = existing_backup_sha256
            else:
                shutil.copy2(dest_path, backup_path)
                backup_sha256 = hashlib.sha256(backup_path.read_bytes()).hexdigest()
                assert backup_sha256 == original_sha256, "Backup hash mismatch after copy!"
                backup_created = True

    # Step 2: Ensure worklist exists
    if not worklist_path.is_file():
        generate_query_sanitation_worklist()
    assert worklist_path.is_file(), f"Worklist not found: {worklist_path}"

    # Load source dataset lookup from human_primary.csv if present
    hp_path = get_annotations_dir() / "human_primary.csv"
    hp_datasets: Dict[str, str] = {}
    if hp_path.is_file():
        df_hp = pd.read_csv(hp_path)
        for _, row in df_hp.iterrows():
            hp_datasets[str(row["sample_id"])] = str(row.get("source_dataset", "AdsQA"))

    # Read worklist rows
    worklist_records: List[Dict[str, str]] = []
    with worklist_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            worklist_records.append(r)

    # Load existing canonical reviews to preserve if non-legacy
    existing_canonical_reviews: Dict[str, QueryReviewRecord] = {}
    if dest_path.is_file() and not is_legacy:
        try:
            existing_canonical_reviews = load_query_reviews(dest_path)
        except Exception:
            existing_canonical_reviews = {}

    # Step 3: Rebuild records
    rebuilt_map: Dict[str, QueryReviewRecord] = {}
    migration_timestamp = datetime.now(timezone.utc).isoformat()

    for r in worklist_records:
        sid = r["sample_id"].strip()
        orig_q = r.get("original_query", "").strip() or r.get("query", "").strip()
        v_file = r.get("video_filename", "").strip()
        s_vid = r.get("source_video_id", "").strip()
        s_ds = hp_datasets.get(sid, "AdsQA")

        rec = QueryReviewRecord(
            sample_id=sid,
            video_filename=v_file,
            source_dataset=s_ds,
            source_video_id=s_vid,
            original_query=orig_q,
            ai_suggested_query="",
            human_final_query="",
            query_review_status=QueryReviewStatus.PENDING_QUERY_REVIEW.value,
            original_query_quality="",
            query_change_type="",
            query_localizable="",
            query_observable="",
            reviewer_id="",
            review_notes="",
            reviewed_at_utc="",
            query_version="1",
            temporal_annotation_locked=True,
        )

        # Step 4: Restore 0028 human review
        if restore_0028 and sid == "lumae_ads_pilot_0028":
            rec.original_query_quality = CONFIRMED_HUMAN_REVIEW_0028["original_query_quality"]
            rec.human_final_query = CONFIRMED_HUMAN_REVIEW_0028["human_final_query"]
            rec.query_review_status = CONFIRMED_HUMAN_REVIEW_0028["query_review_status"]
            rec.query_change_type = CONFIRMED_HUMAN_REVIEW_0028["query_change_type"]
            rec.query_localizable = CONFIRMED_HUMAN_REVIEW_0028["query_localizable"]
            rec.query_observable = CONFIRMED_HUMAN_REVIEW_0028["query_observable"]
            rec.reviewer_id = CONFIRMED_HUMAN_REVIEW_0028["reviewer_id"]
            rec.review_notes = CONFIRMED_HUMAN_REVIEW_0028["review_notes"]
            rec.query_version = CONFIRMED_HUMAN_REVIEW_0028["query_version"]
            rec.temporal_annotation_locked = CONFIRMED_HUMAN_REVIEW_0028["temporal_annotation_locked"]
            rec.reviewed_at_utc = migration_timestamp
        elif sid in existing_canonical_reviews:
            prev = existing_canonical_reviews[sid]
            if prev.query_review_status != QueryReviewStatus.PENDING_QUERY_REVIEW.value:
                rec = prev

        rebuilt_map[sid] = rec

    # Step 5: Atomic write
    records_list = [rebuilt_map[k] for k in sorted(rebuilt_map.keys())]
    atomic_write_query_reviews(dest_path, records_list)

    # Step 6: Validate output
    validate_query_review_schema(dest_path)
    final_sha256 = hashlib.sha256(dest_path.read_bytes()).hexdigest()

    return {
        "status": "MIGRATION_SUCCESS",
        "destination_path": str(dest_path),
        "rows_migrated": len(records_list),
        "rows_failed": 0,
        "temporal_gt_columns_present": False,
        "original_sha256": original_sha256,
        "backup_path": str(backup_path) if is_legacy else None,
        "backup_sha256": backup_sha256,
        "final_sha256": final_sha256,
        "sample_0028_restored": restore_0028 and "lumae_ads_pilot_0028" in rebuilt_map,
        "sample_0028_status": rebuilt_map["lumae_ads_pilot_0028"].query_review_status if "lumae_ads_pilot_0028" in rebuilt_map else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Query Sanitation CLI & Migration Tool")
    parser.add_argument("command", choices=["init", "migrate", "validate", "audit", "worklist"], help="Command to execute")
    parser.add_argument("--force", action="store_true", help="Force execution")
    args = parser.parse_args()

    if args.command in ("init", "migrate"):
        res = init_or_migrate_query_reviews()
        print("Migration Result:", json.dumps(res, indent=2))
    elif args.command == "validate":
        p = get_query_reviews_path()
        validate_query_review_schema(p)
        print(f"Schema VALID for: {p}")
        with open(p, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            print("Columns:", header)
    elif args.command == "audit":
        registry = generate_exposure_registry()
        print("Exposure Registry Summary:")
        print(json.dumps(registry["exposure_summary"], indent=2))
    elif args.command == "worklist":
        rows = generate_query_sanitation_worklist()
        print(f"Generated worklist with {len(rows)} pending samples.")


if __name__ == "__main__":
    main()
