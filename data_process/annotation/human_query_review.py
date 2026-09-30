"""Human-click-only A.2.6R query decisions and the completion-gated freeze."""

from __future__ import annotations

import csv
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import re
import tempfile

from data_process.annotation.models import QueryReviewRecord, QueryReviewStatus
from data_process.annotation.query_sanitation import (
    CANONICAL_QUERY_REVIEW_FIELDS,
    load_query_reviews,
    validate_query_review_schema,
)
from data_process.annotation.validation import validate_query


DECISIONS = ("ACCEPT_SUGGESTION", "EDIT_QUERY", "KEEP_ORIGINAL", "EXCLUDE")
APPROVED = {QueryReviewStatus.HUMAN_EDITED.value, QueryReviewStatus.VALID_AS_IS.value}
MARKER = "A26R_DECISION="
TIMESTAMP = re.compile(r"\b(?:\d{1,2}:\d{2}(?::\d{2})?|\d+(?:\.\d+)?\s*(?:seconds?|secs?|minutes?|mins?|hours?|hrs?)|at\s+\d+(?:\.\d+)?s)\b", re.I)
INVISIBLE_CLAIM = re.compile(r"\b(?:clinically proven|patented|guaranteed|award.winning|most effective|long.lasting|kills \d+%|contains \d+%|made from|hypoallergenic)\b", re.I)
NATURAL_LANGUAGE = re.compile(r"[A-Za-z]+(?:[ '\-][A-Za-z]+){3,}")


def _paths(root: Path) -> tuple[Path, Path, Path, Path]:
    base = root / "local_data"
    return (base / "annotations" / "lumae_ads" / "human_primary.csv",
            base / "annotations" / "lumae_ads" / "ai_query_suggestions.csv",
            base / "annotations" / "lumae_ads" / "query_reviews.csv",
            base / "manifests" / "a26r_human_verified_queries_manifest.json")


def _rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return {r["sample_id"]: r for r in csv.DictReader(f)}


def worklist(root: Path) -> list[dict[str, str]]:
    primary_path, suggestion_path, _, _ = _paths(root)
    primary = _rows(primary_path)
    suggestions = _rows(suggestion_path)
    # The suggestion file is the fixed fast-lane cohort. Primary rows leave DRAFT
    # after temporal review, so recomputing the cohort from current statuses would
    # lose completed samples and break resume/progress.
    target_ids = set(suggestions)
    if not target_ids.issubset(primary):
        raise ValueError("AI suggestions include samples outside the primary pilot")
    result = []
    for sid in sorted(target_ids):
        p, s = primary[sid], suggestions[sid]
        if p["video_filename"] != s["video_filename"] or s.get("status") != "AI_SUGGESTED":
            raise ValueError(f"Invalid suggestion provenance for {sid}")
        result.append({**p, **{f"suggestion_{k}": v for k, v in s.items()}})
    return result


def decision_of(rec: QueryReviewRecord | None) -> str | None:
    if rec is None:
        return None
    match = re.search(r"(?:^|\n)A26R_DECISION=(ACCEPT_SUGGESTION|EDIT_QUERY|KEEP_ORIGINAL|EXCLUDE)(?:\n|$)", rec.review_notes)
    if not match or not rec.reviewer_id.startswith("human_") or not rec.reviewed_at_utc:
        return None
    decision = match.group(1)
    if decision == "EXCLUDE":
        return decision if rec.query_review_status == QueryReviewStatus.EXCLUDE.value else None
    return decision if rec.query_review_status in APPROVED and rec.human_final_query else None


def progress(root: Path) -> dict[str, object]:
    items = worklist(root)
    reviews = load_query_reviews(_paths(root)[2])
    decisions = {r["sample_id"]: decision_of(reviews.get(r["sample_id"])) for r in items}
    order = sorted(items, key=lambda r: (decisions[r["sample_id"]] is not None, r["sample_id"]))
    counts = {d: sum(v == d for v in decisions.values()) for d in DECISIONS}
    return {"items": order, "decisions": decisions, "counts": counts,
            "reviewed": sum(v is not None for v in decisions.values()),
            "remaining": sum(v is None for v in decisions.values())}


def _save_one_preserving_other_rows(path: Path, rec: QueryReviewRecord) -> None:
    """Replace a single CSV row while retaining every other row's original bytes."""
    raw = path.read_bytes()
    newline = "\r\n" if b"\r\n" in raw else "\n"
    lines = raw.decode("utf-8-sig").splitlines(keepends=True)
    reader = csv.reader(lines)
    header = next(reader)
    if header != CANONICAL_QUERY_REVIEW_FIELDS:
        raise ValueError("Canonical query review header/order changed")
    prior_end = reader.line_num
    replacement = None
    chunks = ["".join(lines[:prior_end])]
    for values in reader:
        end = reader.line_num
        if values[0] == rec.sample_id:
            if replacement is not None:
                raise ValueError("Duplicate query review sample ID")
            buffer = io.StringIO(newline="")
            csv.writer(buffer, quoting=csv.QUOTE_ALL, lineterminator=newline).writerow(
                [rec.to_csv_dict()[field] for field in CANONICAL_QUERY_REVIEW_FIELDS]
            )
            chunks.append(buffer.getvalue())
            replacement = True
        else:
            chunks.append("".join(lines[prior_end:end]))
        prior_end = end
    if replacement is None:
        raise ValueError(f"Missing canonical row for {rec.sample_id}")
    payload = (b"\xef\xbb\xbf" if raw.startswith(b"\xef\xbb\xbf") else b"") + "".join(chunks).encode("utf-8")
    temp = None
    try:
        with tempfile.NamedTemporaryFile("wb", delete=False, dir=path.parent) as f:
            temp = Path(f.name)
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        validate_query_review_schema(temp)
        os.replace(temp, path)
    finally:
        if temp and temp.exists():
            temp.unlink()


def submit_decision(root: Path, *, sample_id: str, decision: str, reviewer_id: str,
                    edited_query: str = "", note: str = "", seen_video: bool = False,
                    observable: bool = False, localizable: bool = False,
                    confirm_exclude: bool = False, allow_revision: bool = False,
                    freeze_batch: bool = True) -> QueryReviewRecord:
    """Called only from a Streamlit decision button handler after human input."""
    if decision not in DECISIONS:
        raise ValueError("Choose an explicit decision")
    reviewer = reviewer_id.strip()
    if not re.fullmatch(r"human_[A-Za-z0-9_-]+", reviewer):
        raise ValueError("Enter your actual human_ reviewer ID")
    if not seen_video:
        raise ValueError("Confirm that you watched this video")
    target = {r["sample_id"]: r for r in worklist(root)}.get(sample_id)
    if target is None:
        raise ValueError("Sample is outside the 30 DRAFT query review worklist")
    if target["review_status"] != "DRAFT":
        raise ValueError("Query cannot change after temporal review")
    reviews_path = _paths(root)[2]
    reviews = load_query_reviews(reviews_path)
    old = reviews.get(sample_id)
    if old is None:
        raise ValueError("Missing canonical query review row")
    if decision_of(old) and not allow_revision:
        raise ValueError("This sample already has a human decision")
    if allow_revision and not decision_of(old):
        raise ValueError("There is no prior human query decision to revise")
    suggestion = target["suggestion_ai_suggested_query"].strip()
    original = old.original_query
    if original != target["query"] or old.video_filename != target["video_filename"]:
        raise ValueError("Canonical row and primary metadata differ")
    if decision == "EXCLUDE":
        if not confirm_exclude or not note.strip():
            raise ValueError("Exclusion requires confirmation and a human note")
        final, status, change = "", QueryReviewStatus.EXCLUDE.value, "EXCLUDED"
        observable = localizable = False
    else:
        if decision == "ACCEPT_SUGGESTION":
            final, status, change = suggestion, QueryReviewStatus.HUMAN_EDITED.value, target["suggestion_query_change_type"]
        elif decision == "EDIT_QUERY":
            final, status, change = edited_query.strip(), QueryReviewStatus.HUMAN_EDITED.value, "HUMAN_REWRITE"
            if final == suggestion:
                raise ValueError("Edited query must differ from the AI suggestion; choose Accept instead")
        else:
            final, status, change = original, QueryReviewStatus.VALID_AS_IS.value, "UNCHANGED"
        ok, error = validate_query(final, status)
        if not ok:
            raise ValueError(error)
        if not NATURAL_LANGUAGE.search(final):
            raise ValueError("Final query must be a natural-language action description")
        if TIMESTAMP.search(final):
            raise ValueError("Remove timestamp information from the final query")
        if INVISIBLE_CLAIM.search(final):
            raise ValueError("Final query contains a product claim that may not be visible")
        if not observable or not localizable:
            raise ValueError("Confirm the action is visible and temporally localizable")
    try:
        version = str(int(old.query_version) + 1)
    except ValueError as exc:
        raise ValueError("Invalid current query version") from exc
    notes = f"{MARKER}{decision}\n{note.strip()}".rstrip()
    rec = replace(old, ai_suggested_query=suggestion, human_final_query=final,
                  query_review_status=status,
                  original_query_quality=target["suggestion_original_query_quality"],
                  query_change_type=change, query_localizable=bool(localizable),
                  query_observable=bool(observable), reviewer_id=reviewer,
                  review_notes=notes, reviewed_at_utc=datetime.now(timezone.utc).isoformat(),
                  query_version=version, temporal_annotation_locked=(decision == "EXCLUDE"))
    _save_one_preserving_other_rows(reviews_path, rec)
    if freeze_batch:
        freeze_if_complete(root)
    return rec


def freeze_if_complete(root: Path) -> tuple[Path, str] | None:
    state = progress(root)
    if state["remaining"]:
        return None
    _, _, reviews_path, manifest_path = _paths(root)
    reviews = load_query_reviews(reviews_path)
    approved = []
    for item in sorted(state["items"], key=lambda r: r["sample_id"]):
        rec = reviews[item["sample_id"]]
        if decision_of(rec) == "EXCLUDE":
            continue
        if (rec.query_review_status not in APPROVED or not rec.human_final_query
                or not rec.reviewer_id.startswith("human_") or not rec.reviewed_at_utc
                or rec.query_localizable is not True or rec.query_observable is not True):
            raise ValueError(f"Cannot freeze unapproved query: {rec.sample_id}")
        approved.append({k: getattr(rec, k) for k in (
            "sample_id", "video_filename", "human_final_query", "query_version",
            "reviewer_id", "reviewed_at_utc")})
    payload = {"stage": "A.2.6R", "total_reviewed": state["reviewed"],
               "decision_counts": state["counts"], "approved_queries": approved}
    encoded = (json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", delete=False, dir=manifest_path.parent) as f:
        temp = Path(f.name)
        f.write(encoded)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, manifest_path)
    return manifest_path, hashlib.sha256(encoded).hexdigest()
