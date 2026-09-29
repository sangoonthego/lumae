"""Data models and enumerations for human temporal annotation workflow."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class QueryStatus(str, Enum):
    """Classification of query validity for the associated video."""

    VALID = "VALID"
    NEEDS_EDIT = "NEEDS_EDIT"
    INVALID_VIDEO = "INVALID_VIDEO"


class ReviewStatus(str, Enum):
    """Review progress status of a candidate record."""

    DRAFT = "DRAFT"
    REVIEWED = "REVIEWED"
    EXCLUDED = "EXCLUDED"
    ADJUDICATED = "ADJUDICATED"


class AnnotationMode(str, Enum):
    """Operating mode of the review application."""

    PRIMARY = "PRIMARY"
    AI_ASSISTED_PRIMARY = "AI_ASSISTED_PRIMARY"
    BLIND_PRIMARY = "BLIND_PRIMARY"
    SECONDARY = "SECONDARY"
    ADJUDICATION = "ADJUDICATION"


class AIQueryStatus(str, Enum):
    """Status of AI assessment of query suitability."""

    VALID = "VALID"
    NEEDS_EDIT = "NEEDS_EDIT"
    INVALID_VIDEO = "INVALID_VIDEO"


class AIPreannotationStatus(str, Enum):
    """Execution status of AI pre-annotation generator."""

    AI_PROPOSED = "AI_PROPOSED"
    AI_UNCERTAIN = "AI_UNCERTAIN"
    AI_FAILED = "AI_FAILED"


class AIConfidence(str, Enum):
    """Internal AI confidence classification."""

    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class QueryAction(str, Enum):
    """Human decision action relative to AI proposed query."""

    ACCEPTED_AI = "ACCEPTED_AI"
    EDITED_AI = "EDITED_AI"
    REJECTED_AI = "REJECTED_AI"


class WindowAction(str, Enum):
    """Human decision action relative to AI candidate window."""

    ACCEPTED_AI = "ACCEPTED_AI"
    EDITED_AI = "EDITED_AI"
    REJECTED_AI = "REJECTED_AI"


@dataclass
class TemporalWindow:
    """Continuous temporal window defined by start and end timestamps in seconds."""

    start: float
    end: float

    def to_list(self) -> list[float]:
        return [round(self.start, 1), round(self.end, 1)]

    @classmethod
    def from_list(cls, w: list[float] | tuple[float, float]) -> TemporalWindow:
        return cls(start=float(w[0]), end=float(w[1]))


@dataclass
class AnnotationRecord:
    """Individual sample annotation record corresponding to CSV row with multi-window support."""

    sample_id: str
    video_filename: str
    vid: str
    source_dataset: str
    source_video_id: str
    source_split: str
    source_url: str
    product_category: str
    query: str
    duration_seconds: float
    gt_start_seconds: float | None = None
    gt_end_seconds: float | None = None
    annotator_id: str = "human_01"
    annotation_notes: str = ""
    review_status: str = ReviewStatus.DRAFT.value
    query_status: str = QueryStatus.VALID.value
    original_query: str = ""
    multi_windows: list[TemporalWindow] = field(default_factory=list)

    def is_human_reviewed(self) -> bool:
        """Check if row has genuine human review provenance."""
        return (
            str(self.review_status).upper() == ReviewStatus.REVIEWED.value
            and str(self.annotator_id).startswith("human_")
        )

    def is_excluded(self) -> bool:
        """Check if sample is excluded from dataset."""
        return str(self.review_status).upper() == ReviewStatus.EXCLUDED.value

    def get_effective_windows(self) -> list[list[float]]:
        """Return all valid windows (from multi_windows or primary start/end)."""
        if self.multi_windows:
            return [w.to_list() for w in self.multi_windows]
        if self.gt_start_seconds is not None and self.gt_end_seconds is not None:
            return [[round(self.gt_start_seconds, 1), round(self.gt_end_seconds, 1)]]
        return []

    def to_csv_dict(self) -> dict[str, str]:
        """Convert record to backward-compatible dictionary for CSV export."""
        start_str = f"{self.gt_start_seconds:.1f}" if self.gt_start_seconds is not None else ""
        end_str = f"{self.gt_end_seconds:.1f}" if self.gt_end_seconds is not None else ""
        return {
            "sample_id": self.sample_id,
            "video_filename": self.video_filename,
            "vid": self.vid,
            "source_dataset": self.source_dataset,
            "source_video_id": self.source_video_id,
            "source_split": self.source_split,
            "source_url": self.source_url,
            "product_category": self.product_category,
            "query": self.query,
            "duration_seconds": f"{self.duration_seconds:.2f}",
            "gt_start_seconds": start_str,
            "gt_end_seconds": end_str,
            "annotator_id": self.annotator_id,
            "annotation_notes": self.annotation_notes,
            "review_status": self.review_status,
        }


@dataclass
class AIPreannotationRecord:
    """Automated AI candidate temporal annotation record.
    
    CRITICAL PRINCIPLE:
    AI candidates are NOT ground truth. They serve strictly as advisory pre-annotations
    for accelerating subsequent human verification.
    """

    sample_id: str
    video_filename: str
    original_query: str
    ai_query_status: str  # VALID, NEEDS_EDIT, INVALID_VIDEO
    ai_proposed_query: str
    ai_start_seconds: float | None = None
    ai_end_seconds: float | None = None
    ai_windows_json: str = "[]"
    ai_confidence: str = AIConfidence.MEDIUM.value  # HIGH, MEDIUM, LOW
    ai_reason: str = ""
    analysis_method: str = "ffmpeg_coarse_dense_visual_flow"
    frame_sampling_interval: float = 2.0
    boundary_refinement_interval: float = 0.5
    generated_at_utc: str = ""
    status: str = AIPreannotationStatus.AI_PROPOSED.value

    def get_windows(self) -> list[list[float]]:
        """Return candidate windows parsed from JSON or start/end."""
        import json
        if self.ai_windows_json:
            try:
                parsed = json.loads(self.ai_windows_json)
                if isinstance(parsed, list) and len(parsed) > 0:
                    return [[round(float(w[0]), 1), round(float(w[1]), 1)] for w in parsed]
            except Exception:
                pass
        if self.ai_start_seconds is not None and self.ai_end_seconds is not None:
            return [[round(self.ai_start_seconds, 1), round(self.ai_end_seconds, 1)]]
        return []

    def to_csv_dict(self) -> dict[str, str]:
        """Convert record to dictionary for ai_preannotations.csv export."""
        s_str = f"{self.ai_start_seconds:.1f}" if self.ai_start_seconds is not None else ""
        e_str = f"{self.ai_end_seconds:.1f}" if self.ai_end_seconds is not None else ""
        return {
            "sample_id": self.sample_id,
            "video_filename": self.video_filename,
            "original_query": self.original_query,
            "ai_query_status": self.ai_query_status,
            "ai_proposed_query": self.ai_proposed_query,
            "ai_start_seconds": s_str,
            "ai_end_seconds": e_str,
            "ai_windows_json": self.ai_windows_json,
            "ai_confidence": self.ai_confidence,
            "ai_reason": self.ai_reason,
            "analysis_method": self.analysis_method,
            "frame_sampling_interval": f"{self.frame_sampling_interval:.2f}",
            "boundary_refinement_interval": f"{self.boundary_refinement_interval:.2f}",
            "generated_at_utc": self.generated_at_utc,
            "status": self.status,
        }

    @classmethod
    def from_csv_dict(cls, row: dict[str, str]) -> AIPreannotationRecord:
        """Construct record from CSV row."""
        s_raw = row.get("ai_start_seconds", "").strip()
        e_raw = row.get("ai_end_seconds", "").strip()
        s_val = round(float(s_raw), 1) if s_raw else None
        e_val = round(float(e_raw), 1) if e_raw else None

        f_samp = float(row.get("frame_sampling_interval", 2.0))
        b_ref = float(row.get("boundary_refinement_interval", 0.5))

        return cls(
            sample_id=row.get("sample_id", "").strip(),
            video_filename=row.get("video_filename", "").strip(),
            original_query=row.get("original_query", "").strip(),
            ai_query_status=row.get("ai_query_status", AIQueryStatus.VALID.value).strip(),
            ai_proposed_query=row.get("ai_proposed_query", "").strip(),
            ai_start_seconds=s_val,
            ai_end_seconds=e_val,
            ai_windows_json=row.get("ai_windows_json", "[]").strip(),
            ai_confidence=row.get("ai_confidence", AIConfidence.MEDIUM.value).strip(),
            ai_reason=row.get("ai_reason", "").strip(),
            analysis_method=row.get("analysis_method", "ffmpeg_coarse_dense_visual_flow").strip(),
            frame_sampling_interval=f_samp,
            boundary_refinement_interval=b_ref,
            generated_at_utc=row.get("generated_at_utc", "").strip(),
            status=row.get("status", AIPreannotationStatus.AI_PROPOSED.value).strip(),
        )


@dataclass
class ReviewLogEntry:
    """Immutable audit record logging human query revisions, boundary decisions, and AI provenance."""

    sample_id: str
    video_filename: str
    original_query: str
    final_query: str
    query_status: str
    windows: list[list[float]]
    annotator_id: str
    notes: str
    reviewed_at_utc: str
    previous_review_status: str
    annotation_mode: str = AnnotationMode.PRIMARY.value
    query_action: str | None = None
    window_action: str | None = None
    ai_query_status: str | None = None
    ai_proposed_query: str | None = None
    ai_candidate_windows: list[list[float]] | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # Filter None values to keep logs clean
        return {k: v for k, v in d.items() if v is not None}
