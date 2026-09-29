"""Adapter for Lumae Product-Ads dataset annotations (CSV and JSONL)."""

from __future__ import annotations

from collections import defaultdict
import csv
import math
from pathlib import Path
from typing import Any

from ..schemas.temporal_sample import (
    CanonicalTemporalSample,
    ReasonCode,
    RejectedSampleRecord,
)
from ..validation.temporal import validate_duration, validate_window
from .base import BaseDatasetAdapter


class LumaeAdsAdapter(BaseDatasetAdapter):
    """Adapter for ingesting Lumae Product-Ads annotations from CSV or JSONL.

    Enforces the one-query-per-video policy, supports multiple discontinuous windows,
    and sets saliency_scores strictly to None.
    """

    def __init__(self, check_video_exists: bool = False, video_root: Path | str | None = None):
        super().__init__(name="lumae_product_ads")
        self.check_video_exists = check_video_exists
        self.video_root = Path(video_root) if video_root else None

    def convert_record(
        self,
        raw_record: dict[str, Any],
        index: int,
    ) -> tuple[CanonicalTemporalSample | None, RejectedSampleRecord | None]:
        qid = raw_record.get("sample_id") or raw_record.get("qid")
        if not qid or not str(qid).strip():
            qid = f"lumae_ads_{index+1:06d}"

        vid = raw_record.get("vid") or raw_record.get("video_filename")
        if not vid or not str(vid).strip():
            return None, RejectedSampleRecord(
                source=self.name,
                original_id=qid,
                reason_code=ReasonCode.MISSING_VIDEO_ID,
                reason="Lumae Ads record missing 'vid' or 'video_filename'",
                raw_reference=str(raw_record)[:200],
            )

        query = raw_record.get("query")
        if not query or not str(query).strip():
            return None, RejectedSampleRecord(
                source=self.name,
                original_id=qid,
                reason_code=ReasonCode.MISSING_QUERY,
                reason="Lumae Ads record missing or empty 'query'",
                raw_reference=str(raw_record)[:200],
            )

        dur_raw = raw_record.get("duration_seconds") or raw_record.get("duration")
        dur_ok, dur_err = validate_duration(dur_raw)  # type: ignore[arg-type]
        if not dur_ok:
            return None, RejectedSampleRecord(
                source=self.name,
                original_id=qid,
                reason_code=ReasonCode.INVALID_DURATION,
                reason=f"Invalid duration: {dur_err}",
                raw_reference=str(raw_record)[:200],
            )
        duration_val = float(dur_raw)

        # Parse windows
        windows: list[list[float]] = []
        if "relevant_windows" in raw_record and isinstance(raw_record["relevant_windows"], list):
            for w in raw_record["relevant_windows"]:
                if isinstance(w, (list, tuple)) and len(w) == 2:
                    windows.append([float(w[0]), float(w[1])])
        elif "gt_start_seconds" in raw_record and "gt_end_seconds" in raw_record:
            try:
                s = float(raw_record["gt_start_seconds"])
                e = float(raw_record["gt_end_seconds"])
                windows.append([s, e])
            except (ValueError, TypeError) as exc:
                return None, RejectedSampleRecord(
                    source=self.name,
                    original_id=qid,
                    reason_code=ReasonCode.INVALID_WINDOW,
                    reason=f"Failed parsing gt_start/gt_end: {exc}",
                    raw_reference=str(raw_record)[:200],
                )

        if not windows:
            return None, RejectedSampleRecord(
                source=self.name,
                original_id=qid,
                reason_code=ReasonCode.INVALID_WINDOW,
                reason="No temporal window intervals found in record",
                raw_reference=str(raw_record)[:200],
            )

        for w in windows:
            win_ok, win_err = validate_window(w[0], w[1], duration_val)
            if not win_ok:
                code = ReasonCode.WINDOW_OUT_OF_RANGE if "exceeds" in (win_err or "") else ReasonCode.INVALID_WINDOW
                return None, RejectedSampleRecord(
                    source=self.name,
                    original_id=qid,
                    reason_code=code,
                    reason=f"Window invalid: {win_err}",
                    raw_reference=str(raw_record)[:200],
                )

        # Optional check for physical video file presence
        video_filename = raw_record.get("video_filename")
        if self.check_video_exists and self.video_root and video_filename:
            v_path = self.video_root / str(video_filename).strip()
            if not v_path.is_file():
                return None, RejectedSampleRecord(
                    source=self.name,
                    original_id=qid,
                    reason_code=ReasonCode.MISSING_VIDEO_ID,
                    reason=f"Physical video file not found at {v_path}",
                    raw_reference=str(raw_record)[:200],
                )

        metadata: dict[str, Any] = {
            "annotator_id": raw_record.get("annotator_id"),
            "product_category": raw_record.get("product_category"),
            "review_status": raw_record.get("review_status", "DRAFT"),
            "video_filename": video_filename,
        }
        if raw_record.get("source_video_id"):
            metadata["source_video_id"] = raw_record["source_video_id"]
        if raw_record.get("source_split"):
            metadata["source_split"] = raw_record["source_split"]
        if raw_record.get("source_url"):
            metadata["source_url"] = raw_record["source_url"]
        if raw_record.get("source_dataset"):
            metadata["source_dataset"] = raw_record["source_dataset"]
        if raw_record.get("annotation_notes"):
            metadata["annotation_notes"] = raw_record["annotation_notes"]

        sample = CanonicalTemporalSample(
            qid=str(qid).strip(),
            vid=str(vid).strip(),
            query=str(query).strip(),
            duration=duration_val,
            relevant_windows=windows,
            relevant_clip_ids=None,
            saliency_scores=None,  # Strictly None for Lumae Ads v1
            source=self.name,
            intent=raw_record.get("intent", "product_demo"),
            original_id=str(qid).strip(),
            metadata=metadata,
        )
        return sample, None

    def process_csv(
        self,
        csv_path: Path | str,
    ) -> tuple[list[CanonicalTemporalSample], list[RejectedSampleRecord]]:
        """Process an annotation CSV file, merging multiple windows for same sample_id."""
        path = Path(csv_path)
        if not path.is_file():
            raise FileNotFoundError(f"Annotation CSV file not found: {path}")

        raw_rows: list[dict[str, str]] = []
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                raw_rows.append(dict(row))

        # Group rows by sample_id to support multi-window annotations
        grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
        for idx, row in enumerate(raw_rows):
            key = row.get("sample_id") or row.get("qid") or f"row_{idx}"
            grouped[key].append(row)

        valid_samples: list[CanonicalTemporalSample] = []
        rejected_records: list[RejectedSampleRecord] = []

        for sample_key, group in grouped.items():
            first = group[0]
            # If multiple rows, combine their windows
            if len(group) > 1:
                combined_windows: list[list[float]] = []
                has_error = False
                for r in group:
                    try:
                        s = float(r["gt_start_seconds"])
                        e = float(r["gt_end_seconds"])
                        combined_windows.append([s, e])
                    except Exception as exc:
                        rejected_records.append(
                            RejectedSampleRecord(
                                source=self.name,
                                original_id=sample_key,
                                reason_code=ReasonCode.INVALID_WINDOW,
                                reason=f"Failed parsing window in group: {exc}",
                                raw_reference=str(r)[:200],
                            )
                        )
                        has_error = True
                        break
                if has_error:
                    continue

                merged_record = dict(first)
                merged_record["relevant_windows"] = combined_windows
                sample, rej = self.convert_record(merged_record, 0)
            else:
                sample, rej = self.convert_record(first, 0)

            if sample is not None:
                valid_samples.append(sample)
            elif rej is not None:
                rejected_records.append(rej)

        return valid_samples, rejected_records
