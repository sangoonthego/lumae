"""Persistent storage, atomic updates, backups, and audit logging for human annotation."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import shutil
from typing import Any

from .models import (
    AIConfidence,
    AIPreannotationRecord,
    AIPreannotationStatus,
    AIQueryStatus,
    AnnotationMode,
    AnnotationRecord,
    QueryAction,
    QueryStatus,
    ReviewLogEntry,
    ReviewStatus,
    TemporalWindow,
    WindowAction,
)


def get_repo_root() -> Path:
    """Resolve the repository root directory safely."""
    curr = Path(__file__).resolve().parent
    for parent in [curr] + list(curr.parents):
        if (parent / "AGENTS.md").is_file() or (parent / ".git").exists():
            return parent
    return Path.cwd()


def get_annotations_dir() -> Path:
    """Return the base directory for Lumae Ads annotations."""
    if os.environ.get("LUMAE_ANNOTATIONS_DIR"):
        return Path(os.environ["LUMAE_ANNOTATIONS_DIR"])
    return get_repo_root() / "local_data" / "annotations" / "lumae_ads"


def get_reports_dir() -> Path:
    """Return the reports directory for Lumae Ads."""
    if os.environ.get("LUMAE_REPORTS_DIR"):
        return Path(os.environ["LUMAE_REPORTS_DIR"])
    return get_repo_root() / "local_data" / "reports" / "lumae_ads"


def get_videos_dir() -> Path:
    """Return directory containing acquired AdsQA videos."""
    if os.environ.get("LUMAE_VIDEOS_DIR"):
        return Path(os.environ["LUMAE_VIDEOS_DIR"])
    return get_repo_root() / "local_data" / "raw" / "adsqa" / "videos"


def atomic_write_text(dest_path: Path, content: str, make_backup: bool = True) -> None:
    """Write text content atomically with optional backup."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    if make_backup and dest_path.is_file():
        backup_dir = dest_path.parent / "backups"
        backup_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        backup_file = backup_dir / f"{dest_path.stem}_{timestamp}{dest_path.suffix}"
        try:
            shutil.copy2(dest_path, backup_file)
        except Exception:
            pass

    tmp_path = dest_path.with_suffix(f"{dest_path.suffix}.tmp_{os.getpid()}")
    try:
        with tmp_path.open("w", encoding="utf-8", newline="") as f:
            f.write(content)
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                pass
        tmp_path.replace(dest_path)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


def append_review_audit_log(entry: ReviewLogEntry) -> None:
    """Append a human review decision to the immutable audit log."""
    log_path = get_annotations_dir() / "human_review_log.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")


def load_multi_windows() -> dict[str, list[list[float]]]:
    """Load multi-window annotations map from JSONL."""
    mw_path = get_annotations_dir() / "human_multi_windows.jsonl"
    if not mw_path.is_file():
        return {}

    mapping: dict[str, list[list[float]]] = {}
    with mw_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                sid = item.get("sample_id")
                windows = item.get("windows", [])
                if sid and isinstance(windows, list):
                    mapping[sid] = [[float(w[0]), float(w[1])] for w in windows]
            except Exception:
                continue
    return mapping


def save_multi_window(sample_id: str, windows: list[list[float]]) -> None:
    """Save or update multi-window record in human_multi_windows.jsonl."""
    current = load_multi_windows()
    current[sample_id] = [[round(float(w[0]), 1), round(float(w[1]), 1)] for w in windows]

    mw_path = get_annotations_dir() / "human_multi_windows.jsonl"
    lines = [json.dumps({"sample_id": sid, "windows": w_list}) for sid, w_list in sorted(current.items())]
    content = "\n".join(lines) + ("\n" if lines else "")
    atomic_write_text(mw_path, content, make_backup=False)


def load_primary_records() -> list[AnnotationRecord]:
    """Load primary human annotation records from human_primary.csv.

    STRICT BLINDNESS GUARANTEE:
    Only samples with review_status == 'REVIEWED' and annotator_id starting with 'human_'
    have populated timestamps. All other rows are strictly blank.
    Draft timestamps are NEVER loaded or used for prefilling.
    """
    csv_path = get_annotations_dir() / "human_primary.csv"
    if not csv_path.is_file():
        # Fallback to unannotated worklist if primary has not been initialized
        csv_path = get_annotations_dir() / "pilot_worklist_unannotated.csv"

    if not csv_path.is_file():
        raise FileNotFoundError(f"Annotation file not found: {csv_path}")

    multi_windows_map = load_multi_windows()
    records: list[AnnotationRecord] = []

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sample_id = row.get("sample_id", "").strip()
            ann_id = row.get("annotator_id", "").strip()
            rev_status = row.get("review_status", "DRAFT").strip().upper()

            # Preserve human reviewed status
            is_human_reviewed = rev_status == ReviewStatus.REVIEWED.value and ann_id.startswith("human_")
            is_excluded = rev_status == ReviewStatus.EXCLUDED.value

            start_val: float | None = None
            end_val: float | None = None

            if is_human_reviewed and not is_excluded:
                try:
                    s_str = row.get("gt_start_seconds", "").strip()
                    e_str = row.get("gt_end_seconds", "").strip()
                    if s_str and e_str:
                        start_val = round(float(s_str), 1)
                        end_val = round(float(e_str), 1)
                except (ValueError, TypeError):
                    start_val, end_val = None, None

            # Multi-window check
            mw_list: list[TemporalWindow] = []
            if sample_id in multi_windows_map and is_human_reviewed:
                mw_list = [TemporalWindow.from_list(w) for w in multi_windows_map[sample_id]]

            try:
                dur = float(row.get("duration_seconds", 0.0))
            except (ValueError, TypeError):
                dur = 0.0

            rec = AnnotationRecord(
                sample_id=sample_id,
                video_filename=row.get("video_filename", "").strip(),
                vid=row.get("vid", "").strip(),
                source_dataset=row.get("source_dataset", "AdsQA").strip(),
                source_video_id=row.get("source_video_id", "").strip(),
                source_split=row.get("source_split", "unassigned").strip(),
                source_url=row.get("source_url", "").strip(),
                product_category=row.get("product_category", "Product Demonstration").strip(),
                query=row.get("query", "").strip(),
                duration_seconds=dur,
                gt_start_seconds=start_val,
                gt_end_seconds=end_val,
                annotator_id=ann_id if is_human_reviewed else "human_01",
                annotation_notes=row.get("annotation_notes", "").strip(),
                review_status=rev_status,
                original_query=row.get("query", "").strip(),
                multi_windows=mw_list,
            )
            records.append(rec)

    return records


def save_primary_records(records: list[AnnotationRecord]) -> None:
    """Atomically save updated primary records to human_primary.csv.

    STRICT PROVENANCE & CONTAMINATION GUARDS:
    1. Automated pre-annotations can NEVER be saved directly to human_primary.csv as REVIEWED.
    2. Only verified human annotators ('human_01', 'human_adjudicator', etc.) may have REVIEWED status.
    """
    for rec in records:
        if rec.review_status == ReviewStatus.REVIEWED.value and not str(rec.annotator_id).startswith("human_"):
            raise ValueError(
                f"Contamination guard failed: sample {rec.sample_id} cannot be marked REVIEWED "
                f"with non-human annotator_id '{rec.annotator_id}'."
            )

    csv_path = get_annotations_dir() / "human_primary.csv"
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

    import io
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    for rec in records:
        writer.writerow(rec.to_csv_dict())

    atomic_write_text(csv_path, output.getvalue(), make_backup=True)


def generate_secondary_worklist(
    primary_records: list[AnnotationRecord],
    fraction: float = 0.20,
    seed: int = 42,
) -> list[dict[str, str]]:
    """Generate independent secondary worklist for >=20% of usable samples.

    STRICT BLINDNESS GUARANTEE:
    Contains verified final query and video metadata, but STRICTLY BLANK GT.
    Does NOT contain primary timestamps, agent draft timestamps, or boundary notes.
    """
    usable = [r for r in primary_records if r.review_status == ReviewStatus.REVIEWED.value]
    if not usable:
        return []

    target_count = max(1, int(round(len(usable) * fraction)))
    rng = random.Random(seed)
    selected_samples = sorted(rng.sample(usable, target_count), key=lambda r: r.sample_id)

    rows: list[dict[str, str]] = []
    for r in selected_samples:
        rows.append({
            "sample_id": r.sample_id,
            "video_filename": r.video_filename,
            "vid": r.vid,
            "source_dataset": r.source_dataset,
            "source_video_id": r.source_video_id,
            "source_split": r.source_split,
            "source_url": r.source_url,
            "product_category": r.product_category,
            "query": r.query,  # Final verified query
            "duration_seconds": f"{r.duration_seconds:.2f}",
            "gt_start_seconds": "",  # Strictly BLANK
            "gt_end_seconds": "",    # Strictly BLANK
            "annotator_id": "human_02",
            "annotation_notes": "",  # Purged of boundary notes
            "review_status": "DRAFT",
        })

    worklist_path = get_annotations_dir() / "secondary_worklist.csv"
    import io
    output = io.StringIO()
    fieldnames = list(rows[0].keys())
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)

    atomic_write_text(worklist_path, output.getvalue(), make_backup=True)
    return rows


def load_secondary_records() -> list[AnnotationRecord]:
    """Load secondary annotation records from human_secondary.csv or secondary_worklist.csv."""
    sec_csv = get_annotations_dir() / "human_secondary.csv"
    if not sec_csv.is_file():
        sec_csv = get_annotations_dir() / "secondary_worklist.csv"

    if not sec_csv.is_file():
        return []

    records: list[AnnotationRecord] = []
    with sec_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            s_str = row.get("gt_start_seconds", "").strip()
            e_str = row.get("gt_end_seconds", "").strip()
            start_val = float(s_str) if s_str else None
            end_val = float(e_str) if e_str else None
            dur = float(row.get("duration_seconds", 0.0))

            records.append(
                AnnotationRecord(
                    sample_id=row.get("sample_id", "").strip(),
                    video_filename=row.get("video_filename", "").strip(),
                    vid=row.get("vid", "").strip(),
                    source_dataset=row.get("source_dataset", "AdsQA").strip(),
                    source_video_id=row.get("source_video_id", "").strip(),
                    source_split=row.get("source_split", "unassigned").strip(),
                    source_url=row.get("source_url", "").strip(),
                    product_category=row.get("product_category", "Product Demonstration").strip(),
                    query=row.get("query", "").strip(),
                    duration_seconds=dur,
                    gt_start_seconds=start_val,
                    gt_end_seconds=end_val,
                    annotator_id=row.get("annotator_id", "human_02").strip(),
                    annotation_notes=row.get("annotation_notes", "").strip(),
                    review_status=row.get("review_status", "DRAFT").strip().upper(),
                    original_query=row.get("query", "").strip(),
                )
            )
    return records


def save_secondary_records(records: list[AnnotationRecord]) -> None:
    """Atomically save secondary records to human_secondary.csv."""
    sec_csv = get_annotations_dir() / "human_secondary.csv"
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

    import io
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    for rec in records:
        writer.writerow(rec.to_csv_dict())

    atomic_write_text(sec_csv, output.getvalue(), make_backup=True)


def load_adjudicated_records() -> list[AnnotationRecord]:
    """Load adjudicated records from human_adjudicated.csv."""
    adj_csv = get_annotations_dir() / "human_adjudicated.csv"
    if not adj_csv.is_file():
        return []

    records: list[AnnotationRecord] = []
    with adj_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            s_str = row.get("gt_start_seconds", "").strip()
            e_str = row.get("gt_end_seconds", "").strip()
            records.append(
                AnnotationRecord(
                    sample_id=row.get("sample_id", "").strip(),
                    video_filename=row.get("video_filename", "").strip(),
                    vid=row.get("vid", "").strip(),
                    source_dataset=row.get("source_dataset", "AdsQA").strip(),
                    source_video_id=row.get("source_video_id", "").strip(),
                    source_split=row.get("source_split", "unassigned").strip(),
                    source_url=row.get("source_url", "").strip(),
                    product_category=row.get("product_category", "Product Demonstration").strip(),
                    query=row.get("query", "").strip(),
                    duration_seconds=float(row.get("duration_seconds", 0.0)),
                    gt_start_seconds=float(s_str) if s_str else None,
                    gt_end_seconds=float(e_str) if e_str else None,
                    annotator_id=row.get("annotator_id", "human_adjudicator").strip(),
                    annotation_notes=row.get("annotation_notes", "").strip(),
                    review_status=row.get("review_status", ReviewStatus.ADJUDICATED.value).strip().upper(),
                    original_query=row.get("query", "").strip(),
                )
            )
    return records


def save_adjudicated_record(rec: AnnotationRecord) -> None:
    """Save or append single adjudicated record to human_adjudicated.csv."""
    existing = load_adjudicated_records()
    by_id = {r.sample_id: r for r in existing}
    by_id[rec.sample_id] = rec
    updated = list(by_id.values())

    adj_csv = get_annotations_dir() / "human_adjudicated.csv"
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

    import io
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    for r in updated:
        writer.writerow(r.to_csv_dict())

    atomic_write_text(adj_csv, output.getvalue(), make_backup=True)


def get_blind_holdout_path() -> Path:
    """Return path to blind human holdout JSON file."""
    return get_annotations_dir() / "blind_holdout.json"


def load_blind_holdout() -> dict[str, Any]:
    """Load blind human holdout metadata and sample IDs."""
    path = get_blind_holdout_path()
    if not path.is_file():
        return {"seed": 42, "sample_ids": [], "sample_count": 0}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def is_blind_holdout(sample_id: str) -> bool:
    """Check if a sample belongs to the blind human holdout."""
    holdout = load_blind_holdout()
    return sample_id in holdout.get("sample_ids", [])


def create_blind_holdout(
    pending_sample_ids: list[str],
    count: int = 10,
    fraction: float = 0.20,
    seed: int = 42,
) -> dict[str, Any]:
    """Deterministically select blind human holdout samples."""
    pool = sorted(pending_sample_ids)
    target = max(count, int(round(len(pool) * fraction)))
    target = min(target, len(pool))
    rng = random.Random(seed)
    chosen = sorted(rng.sample(pool, target))

    data = {
        "seed": seed,
        "sample_count": len(chosen),
        "pending_pool_size": len(pool),
        "fraction": round(len(chosen) / len(pool), 4) if pool else 0.0,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "sample_ids": chosen,
        "policy": "Blind holdout samples must strictly hide AI candidates from human review until primary human annotation is saved.",
    }
    path = get_blind_holdout_path()
    atomic_write_text(path, json.dumps(data, indent=2), make_backup=True)
    return data


def get_ai_preannotations_path() -> Path:
    """Return path to AI pre-annotations CSV file."""
    return get_annotations_dir() / "ai_preannotations.csv"


def get_ai_tmp_dir() -> Path:
    """Return directory for temporary AI frame extractions and contact sheets."""
    tmp_dir = get_repo_root() / "local_data" / "tmp" / "ai_annotation"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    return tmp_dir


def load_ai_preannotations() -> dict[str, AIPreannotationRecord]:
    """Load AI candidate pre-annotations mapping sample_id to AIPreannotationRecord."""
    csv_path = get_ai_preannotations_path()
    if not csv_path.is_file():
        return {}

    mapping: dict[str, AIPreannotationRecord] = {}
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = row.get("sample_id", "").strip()
            if sid:
                mapping[sid] = AIPreannotationRecord.from_csv_dict(row)
    return mapping


def save_ai_preannotations(records: list[AIPreannotationRecord]) -> None:
    """Atomically save AI pre-annotations to ai_preannotations.csv.

    ABSOLUTE SAFETY RULE:
    This function writes ONLY to ai_preannotations.csv.
    It NEVER touches human_primary.csv or any ground-truth file.
    """
    csv_path = get_ai_preannotations_path()
    fieldnames = [
        "sample_id",
        "video_filename",
        "original_query",
        "ai_query_status",
        "ai_proposed_query",
        "ai_start_seconds",
        "ai_end_seconds",
        "ai_windows_json",
        "ai_confidence",
        "ai_reason",
        "analysis_method",
        "frame_sampling_interval",
        "boundary_refinement_interval",
        "generated_at_utc",
        "status",
    ]

    import io
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    # Sort for deterministic output
    for rec in sorted(records, key=lambda x: x.sample_id):
        writer.writerow(rec.to_csv_dict())

    atomic_write_text(csv_path, output.getvalue(), make_backup=False)


def append_or_update_ai_preannotation(rec: AIPreannotationRecord) -> None:
    """Add or update an individual AI candidate in ai_preannotations.csv."""
    existing = load_ai_preannotations()
    existing[rec.sample_id] = rec
    save_ai_preannotations(list(existing.values()))
