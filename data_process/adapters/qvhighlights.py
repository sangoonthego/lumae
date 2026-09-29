"""Adapter for QVHighlights dataset annotations."""

from __future__ import annotations

import math
from typing import Any

from ..schemas.temporal_sample import (
    CanonicalTemporalSample,
    ReasonCode,
    RejectedSampleRecord,
)
from ..validation.temporal import validate_duration, validate_windows
from .base import BaseDatasetAdapter


class QVHighlightsAdapter(BaseDatasetAdapter):
    """Adapter for official QVHighlights JSONL annotation format.

    Validates and preserves ground-truth windows and multi-annotator saliency scores
    while normalizing into the canonical temporal grounding schema.
    """

    def __init__(self, namespace_vid: bool = False, strict_annotator_count: bool = True):
        super().__init__(name="qvhighlights")
        self.namespace_vid = namespace_vid
        self.strict_annotator_count = strict_annotator_count

    def convert_record(
        self,
        raw_record: dict[str, Any],
        index: int,
    ) -> tuple[CanonicalTemporalSample | None, RejectedSampleRecord | None]:
        qid = raw_record.get("qid")
        if qid is None:
            return None, RejectedSampleRecord(
                source=self.name,
                original_id=None,
                reason_code=ReasonCode.DUPLICATE_QID if "qid" in raw_record else ReasonCode.MALFORMED_RECORD,
                reason="Record missing 'qid'",
                raw_reference=str(raw_record)[:200],
            )

        vid = raw_record.get("vid")
        if not vid or not isinstance(vid, str) or not vid.strip():
            return None, RejectedSampleRecord(
                source=self.name,
                original_id=qid,
                reason_code=ReasonCode.MISSING_VIDEO_ID,
                reason="Record missing or has empty 'vid'",
                raw_reference=str(raw_record)[:200],
            )

        query = raw_record.get("query")
        if not query or not isinstance(query, str) or not query.strip():
            return None, RejectedSampleRecord(
                source=self.name,
                original_id=qid,
                reason_code=ReasonCode.MISSING_QUERY,
                reason="Record missing or has empty 'query'",
                raw_reference=str(raw_record)[:200],
            )

        duration = raw_record.get("duration")
        dur_ok, dur_err = validate_duration(duration)  # type: ignore[arg-type]
        if not dur_ok:
            return None, RejectedSampleRecord(
                source=self.name,
                original_id=qid,
                reason_code=ReasonCode.INVALID_DURATION,
                reason=f"Invalid duration: {dur_err}",
                raw_reference=str(raw_record)[:200],
            )

        duration_val = float(duration)
        relevant_windows = raw_record.get("relevant_windows")
        if not isinstance(relevant_windows, list) or not relevant_windows:
            return None, RejectedSampleRecord(
                source=self.name,
                original_id=qid,
                reason_code=ReasonCode.INVALID_WINDOW,
                reason="Missing or empty 'relevant_windows'",
                raw_reference=str(raw_record)[:200],
            )

        # In QVHighlights, windows can occasionally exceed duration by sub-second rounding; allow 1.5s tolerance
        win_ok, win_err = validate_windows(relevant_windows, duration_val, tolerance=1.5)
        if not win_ok:
            code = ReasonCode.WINDOW_OUT_OF_RANGE if "exceeds" in (win_err or "") else ReasonCode.INVALID_WINDOW
            return None, RejectedSampleRecord(
                source=self.name,
                original_id=qid,
                reason_code=code,
                reason=f"Invalid relevant_windows: {win_err}",
                raw_reference=str(raw_record)[:200],
            )

        # Saliency and Clip IDs validation for QVHighlights
        raw_clip_ids = raw_record.get("relevant_clip_ids")
        raw_saliency = raw_record.get("saliency_scores")

        canonical_clip_ids: list[int] | None = None
        canonical_saliency: list[list[float]] | None = None

        if raw_clip_ids is not None or raw_saliency is not None:
            if raw_clip_ids is None or not isinstance(raw_clip_ids, list):
                return None, RejectedSampleRecord(
                    source=self.name,
                    original_id=qid,
                    reason_code=ReasonCode.MISMATCHED_CLIP_SALIENCY,
                    reason="Record has saliency_scores but missing or invalid relevant_clip_ids",
                    raw_reference=str(raw_record)[:200],
                )
            if raw_saliency is None or not isinstance(raw_saliency, list):
                return None, RejectedSampleRecord(
                    source=self.name,
                    original_id=qid,
                    reason_code=ReasonCode.MISMATCHED_CLIP_SALIENCY,
                    reason="Record has relevant_clip_ids but missing or invalid saliency_scores",
                    raw_reference=str(raw_record)[:200],
                )

            if len(raw_clip_ids) != len(raw_saliency):
                return None, RejectedSampleRecord(
                    source=self.name,
                    original_id=qid,
                    reason_code=ReasonCode.MISMATCHED_CLIP_SALIENCY,
                    reason=f"Length mismatch: {len(raw_clip_ids)} clip IDs vs {len(raw_saliency)} saliency rows",
                    raw_reference=str(raw_record)[:200],
                )

            # Validate clip IDs
            for cid in raw_clip_ids:
                if not isinstance(cid, int) or cid < 0:
                    return None, RejectedSampleRecord(
                        source=self.name,
                        original_id=qid,
                        reason_code=ReasonCode.INVALID_SALIENCY,
                        reason=f"Invalid clip ID: {cid}",
                        raw_reference=str(raw_record)[:200],
                    )
            canonical_clip_ids = [int(c) for c in raw_clip_ids]

            # Validate multi-annotator saliency
            canonical_saliency = []
            for c_idx, row in enumerate(raw_saliency):
                if not isinstance(row, (list, tuple)) or len(row) == 0:
                    return None, RejectedSampleRecord(
                        source=self.name,
                        original_id=qid,
                        reason_code=ReasonCode.INVALID_SALIENCY,
                        reason=f"Saliency row at clip {c_idx} is empty or not a list",
                        raw_reference=str(raw_record)[:200],
                    )
                if self.strict_annotator_count and len(row) != 3:
                    return None, RejectedSampleRecord(
                        source=self.name,
                        original_id=qid,
                        reason_code=ReasonCode.INVALID_SALIENCY,
                        reason=f"Expected 3 annotators for QVHighlights, got {len(row)} at clip {c_idx}",
                        raw_reference=str(raw_record)[:200],
                    )
                clean_row: list[float] = []
                for a_idx, s in enumerate(row):
                    if not isinstance(s, (int, float)) or not math.isfinite(s):
                        return None, RejectedSampleRecord(
                            source=self.name,
                            original_id=qid,
                            reason_code=ReasonCode.INVALID_SALIENCY,
                            reason=f"Non-finite saliency score at clip {c_idx}, annotator {a_idx}: {s}",
                            raw_reference=str(raw_record)[:200],
                        )
                    s_val = float(s)
                    if s_val < 0.0 or s_val > 4.0:
                        return None, RejectedSampleRecord(
                            source=self.name,
                            original_id=qid,
                            reason_code=ReasonCode.INVALID_SALIENCY,
                            reason=f"Saliency score out of range [0, 4] at clip {c_idx}, annotator {a_idx}: {s_val}",
                            raw_reference=str(raw_record)[:200],
                        )
                    clean_row.append(s_val)
                canonical_saliency.append(clean_row)

        canonical_vid = f"qvh__{vid.strip()}" if self.namespace_vid else vid.strip()

        sample = CanonicalTemporalSample(
            qid=qid,
            vid=canonical_vid,
            query=query.strip(),
            duration=duration_val,
            relevant_windows=[[float(w[0]), float(w[1])] for w in relevant_windows],
            relevant_clip_ids=canonical_clip_ids,
            saliency_scores=canonical_saliency,
            source="qvhighlights",
            original_id=qid,
            metadata={"original_vid": vid.strip()},
        )
        return sample, None
