"""Canonical temporal grounding sample schema and rejection records."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import math
from typing import Any


class ReasonCode(str, Enum):
    """Standardized rejection reason codes."""
    MISSING_QUERY = "MISSING_QUERY"
    INVALID_DURATION = "INVALID_DURATION"
    INVALID_WINDOW = "INVALID_WINDOW"
    WINDOW_OUT_OF_RANGE = "WINDOW_OUT_OF_RANGE"
    MISSING_VIDEO_ID = "MISSING_VIDEO_ID"
    DUPLICATE_QID = "DUPLICATE_QID"
    UNSUPPORTED_SOURCE_RECORD = "UNSUPPORTED_SOURCE_RECORD"
    MALFORMED_RECORD = "MALFORMED_RECORD"
    INVALID_SALIENCY = "INVALID_SALIENCY"
    MISMATCHED_CLIP_SALIENCY = "MISMATCHED_CLIP_SALIENCY"


@dataclass
class CanonicalTemporalSample:
    """Canonical research-domain sample for temporal grounding (offline training/eval).

    Independent of frontend serving types (AnalysisResponse, LumaeInterpretation).

    Saliency Contract (nested annotator scores):
    - `saliency_scores` is a list of lists of floats (or None).
    - Outer list index: corresponds 1:1 with `relevant_clip_ids`.
    - Inner list: all annotator scores for that specific clip (e.g. 3 scores in QVHighlights).
    - Preserves raw annotator scores without averaging, flattening, or normalization.
    """
    qid: int | str
    vid: str
    query: str
    duration: float
    relevant_windows: list[list[float]]
    relevant_clip_ids: list[int] | None = None
    saliency_scores: list[list[float]] | None = None
    source: str | None = None
    intent: str | None = None
    original_id: str | int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Normalize and validate types where possible
        if isinstance(self.qid, str):
            self.qid = self.qid.strip()
            # Try to convert to int if purely numeric string
            if self.qid.isdigit():
                self.qid = int(self.qid)
        if isinstance(self.vid, str):
            self.vid = self.vid.strip()
        if isinstance(self.query, str):
            self.query = self.query.strip()
        if not isinstance(self.duration, (int, float)) or not math.isfinite(self.duration):
            raise ValueError(f"duration must be a finite positive number, got {self.duration}")
        self.duration = float(self.duration)
        if not isinstance(self.relevant_windows, list):
            raise TypeError(f"relevant_windows must be a list of [start, end] pairs, got {type(self.relevant_windows)}")
        normalized_windows: list[list[float]] = []
        for w in self.relevant_windows:
            if not isinstance(w, (list, tuple)) or len(w) != 2:
                raise ValueError(f"Each window must have [start, end], got {w}")
            s, e = float(w[0]), float(w[1])
            normalized_windows.append([s, e])
        self.relevant_windows = normalized_windows

        if self.saliency_scores is not None:
            if not isinstance(self.saliency_scores, list):
                raise TypeError(f"saliency_scores must be a list of lists of annotator scores, got {type(self.saliency_scores)}")
            normalized_saliency: list[list[float]] = []
            for idx, clip_scores in enumerate(self.saliency_scores):
                if not isinstance(clip_scores, (list, tuple)):
                    raise TypeError(f"Each saliency entry must be a list of annotator scores, got {type(clip_scores)} at index {idx}")
                if len(clip_scores) == 0:
                    raise ValueError(f"Saliency entry at index {idx} must not be empty")
                row: list[float] = []
                for s in clip_scores:
                    if not isinstance(s, (int, float)) or not math.isfinite(s):
                        raise ValueError(f"Annotator saliency score must be a finite number, got {s} at index {idx}")
                    row.append(float(s))
                normalized_saliency.append(row)
            self.saliency_scores = normalized_saliency

    def to_dict(self) -> dict[str, Any]:
        """Convert sample to JSON-serializable dictionary."""
        d: dict[str, Any] = {
            "qid": self.qid,
            "vid": self.vid,
            "query": self.query,
            "duration": round(self.duration, 4),
            "relevant_windows": [[round(w[0], 4), round(w[1], 4)] for w in self.relevant_windows],
        }
        if self.relevant_clip_ids is not None:
            d["relevant_clip_ids"] = list(self.relevant_clip_ids)
        if self.saliency_scores is not None:
            d["saliency_scores"] = [[round(float(s), 4) for s in row] for row in self.saliency_scores]
        if self.source is not None:
            d["source"] = self.source
        if self.intent is not None:
            d["intent"] = self.intent
        if self.original_id is not None:
            d["original_id"] = self.original_id
        if self.metadata:
            d["metadata"] = self.metadata
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CanonicalTemporalSample:
        """Construct a CanonicalTemporalSample from a dictionary."""
        saliency_raw = data.get("saliency_scores")
        saliency_scores: list[list[float]] | None = None
        if saliency_raw is not None and isinstance(saliency_raw, list):
            saliency_scores = [
                [float(s) for s in row] if isinstance(row, (list, tuple)) else [float(row)]
                for row in saliency_raw
            ]

        return cls(
            qid=data["qid"],
            vid=data["vid"],
            query=data["query"],
            duration=float(data["duration"]),
            relevant_windows=[[float(w[0]), float(w[1])] for w in data["relevant_windows"]],
            relevant_clip_ids=data.get("relevant_clip_ids"),
            saliency_scores=saliency_scores,
            source=data.get("source"),
            intent=data.get("intent"),
            original_id=data.get("original_id"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class RejectedSampleRecord:
    """Detailed log record for data rejected during adapter parsing or validation."""
    source: str
    original_id: str | int | None
    reason_code: ReasonCode | str
    reason: str
    raw_reference: str | dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert rejection record to JSON-serializable dictionary."""
        code = self.reason_code.value if isinstance(self.reason_code, ReasonCode) else str(self.reason_code)
        return {
            "source": self.source,
            "original_id": self.original_id,
            "reason_code": code,
            "reason": self.reason,
            "raw_reference": self.raw_reference,
        }
