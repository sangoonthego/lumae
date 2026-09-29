"""Adapter skeleton and contract for Ego4D Natural Language Queries (NLQ).

Ego4D NLQ addresses query-conditioned temporal localization in long-form egocentric video.
Expected official Ego4D NLQ source annotation format:
- Root dictionary with `videos` list (or split files `nlq_train.json`, `nlq_val.json`).
- Each video has `video_uid`, `clips` list.
- Each clip has `clip_uid`, `video_start_sec`, `video_end_sec`, `annotations` list.
- Each annotation has `language_queries` list.
- Each query record contains:
    - `query`: natural language query string
    - `clip_start_sec`: start of answer window relative to clip
    - `clip_end_sec`: end of answer window relative to clip
    - `annotation_id`: unique annotation identifier

Note: Saliency scores are NOT present in Ego4D NLQ and are NOT fabricated.
"""

from __future__ import annotations

from typing import Any

from ..schemas.temporal_sample import (
    CanonicalTemporalSample,
    ReasonCode,
    RejectedSampleRecord,
)
from ..validation.temporal import validate_duration, validate_window
from .base import BaseDatasetAdapter


class Ego4DNLQAdapter(BaseDatasetAdapter):
    """Adapter for Ego4D NLQ annotations into canonical temporal grounding format.

    Supports both pre-flattened JSONL records and individual clip-query objects.
    """

    def __init__(self, qid_prefix: str = "ego4d_nlq_"):
        super().__init__(name="ego4d_nlq")
        self.qid_prefix = qid_prefix

    def convert_record(
        self,
        raw_record: dict[str, Any],
        index: int,
    ) -> tuple[CanonicalTemporalSample | None, RejectedSampleRecord | None]:
        """Convert a flattened Ego4D NLQ query record into CanonicalTemporalSample.

        Expected minimal flat record:
        {
            "query": "Where did I put the keys?",
            "clip_uid": "abc-123",
            "clip_duration": 48.5, # or "duration": 48.5
            "clip_start_sec": 12.0,
            "clip_end_sec": 18.5,
            "annotation_id": "annot_99" # optional
        }
        """
        # 1. Video / Clip ID
        vid = raw_record.get("clip_uid") or raw_record.get("video_uid") or raw_record.get("vid")
        if not vid or not isinstance(vid, str) or not vid.strip():
            return None, RejectedSampleRecord(
                source=self.name,
                original_id=raw_record.get("annotation_id") or str(index),
                reason_code=ReasonCode.MISSING_VIDEO_ID,
                reason="Ego4D record missing 'clip_uid' or 'video_uid'",
                raw_reference=str(raw_record)[:200],
            )

        # 2. Query
        query = raw_record.get("query")
        if not query or not isinstance(query, str) or not query.strip():
            return None, RejectedSampleRecord(
                source=self.name,
                original_id=raw_record.get("annotation_id") or str(index),
                reason_code=ReasonCode.MISSING_QUERY,
                reason="Ego4D record missing or has empty 'query'",
                raw_reference=str(raw_record)[:200],
            )

        # 3. Duration
        duration = raw_record.get("clip_duration") or raw_record.get("duration")
        dur_ok, dur_err = validate_duration(duration)  # type: ignore[arg-type]
        if not dur_ok:
            return None, RejectedSampleRecord(
                source=self.name,
                original_id=raw_record.get("annotation_id") or str(index),
                reason_code=ReasonCode.INVALID_DURATION,
                reason=f"Invalid clip duration: {dur_err}",
                raw_reference=str(raw_record)[:200],
            )
        duration_val = float(duration)

        # 4. Temporal window (either [clip_start_sec, clip_end_sec] or relevant_windows)
        relevant_windows: list[list[float]] = []
        if "relevant_windows" in raw_record and isinstance(raw_record["relevant_windows"], list):
            for w in raw_record["relevant_windows"]:
                if isinstance(w, (list, tuple)) and len(w) == 2:
                    relevant_windows.append([float(w[0]), float(w[1])])
        elif "clip_start_sec" in raw_record and "clip_end_sec" in raw_record:
            start_sec = raw_record.get("clip_start_sec")
            end_sec = raw_record.get("clip_end_sec")
            if isinstance(start_sec, (int, float)) and isinstance(end_sec, (int, float)):
                relevant_windows.append([float(start_sec), float(end_sec)])

        if not relevant_windows:
            return None, RejectedSampleRecord(
                source=self.name,
                original_id=raw_record.get("annotation_id") or str(index),
                reason_code=ReasonCode.INVALID_WINDOW,
                reason="Ego4D record has no valid temporal window boundaries",
                raw_reference=str(raw_record)[:200],
            )

        # Validate windows
        for w in relevant_windows:
            win_ok, win_err = validate_window(w[0], w[1], duration_val, tolerance=0.5)
            if not win_ok:
                code = ReasonCode.WINDOW_OUT_OF_RANGE if "exceeds" in (win_err or "") else ReasonCode.INVALID_WINDOW
                return None, RejectedSampleRecord(
                    source=self.name,
                    original_id=raw_record.get("annotation_id") or str(index),
                    reason_code=code,
                    reason=f"Temporal window invalid: {win_err}",
                    raw_reference=str(raw_record)[:200],
                )

        # 5. Stable QID
        annot_id = raw_record.get("annotation_id")
        qid = raw_record.get("qid")
        if qid is None:
            qid = f"{self.qid_prefix}{annot_id}" if annot_id else f"{self.qid_prefix}{index}"

        canonical_vid = f"ego4d__{vid.strip()}" if not vid.startswith("ego4d__") else vid.strip()

        sample = CanonicalTemporalSample(
            qid=qid,
            vid=canonical_vid,
            query=query.strip(),
            duration=duration_val,
            relevant_windows=relevant_windows,
            saliency_scores=None,  # Saliency is NEVER fabricated for Ego4D
            source="ego4d_nlq",
            original_id=annot_id or qid,
            metadata={
                "clip_uid": vid.strip(),
                "video_uid": raw_record.get("video_uid"),
                "slot_id": raw_record.get("slot_id"),
            },
        )
        return sample, None
