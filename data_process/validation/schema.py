"""Schema-level validation for canonical temporal grounding records."""

from __future__ import annotations

import math
from typing import Any

from ..schemas.temporal_sample import CanonicalTemporalSample
from .temporal import validate_duration, validate_windows


class ValidationError(Exception):
    """Raised when canonical sample validation fails."""
    def __init__(self, message: str, errors: list[str] | None = None):
        super().__init__(message)
        self.errors = errors or [message]


def validate_canonical_dict(
    data: dict[str, Any],
    tolerance: float = 0.0,
    allow_empty_windows: bool = False,
    saliency_min: float | None = None,
    saliency_max: float | None = None,
    expected_annotator_count: int | None = None,
) -> tuple[bool, list[str]]:
    """Validate a raw dictionary against canonical temporal grounding rules.

    Returns:
        (is_valid, list_of_error_strings)
    """
    errors: list[str] = []

    # 1. qid
    if "qid" not in data or data["qid"] is None:
        errors.append("Missing required field 'qid'")
    elif isinstance(data["qid"], str) and not data["qid"].strip():
        errors.append("Field 'qid' must be non-empty string or integer")
    elif not isinstance(data["qid"], (int, str)):
        errors.append(f"Field 'qid' must be an integer or string, got {type(data['qid'])}")

    # 2. vid
    if "vid" not in data or not isinstance(data["vid"], str) or not data["vid"].strip():
        errors.append("Field 'vid' must be a non-empty string")

    # 3. query
    if "query" not in data or not isinstance(data["query"], str) or not data["query"].strip():
        errors.append("Field 'query' must be a non-empty string")

    # 4. duration
    duration = data.get("duration")
    dur_ok, dur_err = validate_duration(duration)  # type: ignore[arg-type]
    if not dur_ok:
        errors.append(f"Invalid 'duration': {dur_err}")

    # 5. relevant_windows
    if dur_ok:
        windows = data.get("relevant_windows")
        if windows is None or not isinstance(windows, list):
            errors.append("Field 'relevant_windows' must be a list")
        else:
            win_ok, win_err = validate_windows(windows, float(duration), tolerance=tolerance, allow_empty=allow_empty_windows)
            if not win_ok:
                errors.append(f"Invalid 'relevant_windows': {win_err}")

    # 6. Optional: relevant_clip_ids
    clip_ids = data.get("relevant_clip_ids")
    if clip_ids is not None:
        if not isinstance(clip_ids, list):
            errors.append("Field 'relevant_clip_ids' must be a list of integers")
        else:
            for i, cid in enumerate(clip_ids):
                if not isinstance(cid, int) or cid < 0:
                    errors.append(f"Clip ID at index {i} must be non-negative integer, got {cid}")
                    break

    # 7. Optional: saliency_scores (nested list of annotator scores per clip)
    saliency_scores = data.get("saliency_scores")
    if saliency_scores is not None:
        if not isinstance(saliency_scores, list):
            errors.append("Field 'saliency_scores' must be a list of lists of annotator scores")
        else:
            # Cross-field length check if clip_ids is also provided
            if clip_ids is not None and isinstance(clip_ids, list):
                if len(clip_ids) != len(saliency_scores):
                    errors.append(
                        f"Mismatched lengths: 'relevant_clip_ids' has {len(clip_ids)} items "
                        f"but 'saliency_scores' has {len(saliency_scores)} entries"
                    )

            for i, clip_row in enumerate(saliency_scores):
                if not isinstance(clip_row, (list, tuple)):
                    errors.append(f"Saliency entry at index {i} must be a list of annotator scores, got {type(clip_row)}")
                    break
                if len(clip_row) == 0:
                    errors.append(f"Saliency entry at index {i} is empty; must contain at least one annotator score")
                    break
                if expected_annotator_count is not None and len(clip_row) != expected_annotator_count:
                    errors.append(
                        f"Saliency entry at index {i} has {len(clip_row)} annotator scores; expected {expected_annotator_count}"
                    )
                    break
                for j, score in enumerate(clip_row):
                    if not isinstance(score, (int, float)) or not math.isfinite(score):
                        errors.append(f"Saliency score at clip {i}, annotator {j} must be a finite number, got {score}")
                        break
                    if saliency_min is not None and score < saliency_min:
                        errors.append(f"Saliency score {score} at clip {i}, annotator {j} is below minimum {saliency_min}")
                        break
                    if saliency_max is not None and score > saliency_max:
                        errors.append(f"Saliency score {score} at clip {i}, annotator {j} exceeds maximum {saliency_max}")
                        break

    return len(errors) == 0, errors


def validate_canonical_sample(
    sample: CanonicalTemporalSample,
    tolerance: float = 0.0,
    allow_empty_windows: bool = False,
    saliency_min: float | None = None,
    saliency_max: float | None = None,
    expected_annotator_count: int | None = None,
) -> tuple[bool, list[str]]:
    """Validate a CanonicalTemporalSample instance."""
    return validate_canonical_dict(
        sample.to_dict(),
        tolerance=tolerance,
        allow_empty_windows=allow_empty_windows,
        saliency_min=saliency_min,
        saliency_max=saliency_max,
        expected_annotator_count=expected_annotator_count,
    )
