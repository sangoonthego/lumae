"""Temporal boundary validation for durations and windows."""

from __future__ import annotations

import math
from typing import Sequence


def validate_duration(duration: float | int | str) -> tuple[bool, str | None]:
    """Validate video duration is a finite, positive number."""
    if isinstance(duration, str):
        try:
            duration = float(duration.strip())
        except (ValueError, TypeError):
            return False, f"Duration must be a finite number, got {duration}"
    if not isinstance(duration, (int, float)) or not math.isfinite(duration):
        return False, f"Duration must be a finite number, got {duration}"
    if duration <= 0:
        return False, f"Duration must be strictly positive (>0), got {duration}"
    return True, None


def validate_window(
    start: float,
    end: float,
    duration: float,
    tolerance: float = 0.0,
) -> tuple[bool, str | None]:
    """Validate a single temporal window against video duration.

    Rules:
    - start and end must be finite numbers.
    - 0 <= start.
    - start < end.
    - end <= duration + tolerance.
    """
    if not isinstance(start, (int, float)) or not math.isfinite(start):
        return False, f"Window start must be a finite number, got {start}"
    if not isinstance(end, (int, float)) or not math.isfinite(end):
        return False, f"Window end must be a finite number, got {end}"
    if start < 0:
        return False, f"Window start must be non-negative (>=0), got {start}"
    if start >= end:
        return False, f"Window start ({start}) must be strictly less than end ({end})"
    if end > duration + tolerance:
        return False, f"Window end ({end}) exceeds video duration ({duration})"
    return True, None


def validate_windows(
    windows: Sequence[Sequence[float]],
    duration: float,
    tolerance: float = 0.0,
    allow_empty: bool = False,
) -> tuple[bool, str | None]:
    """Validate a collection of temporal windows for a given video."""
    if not allow_empty and not windows:
        return False, "Relevant windows list must not be empty for supervised samples"

    for idx, w in enumerate(windows):
        if not isinstance(w, (list, tuple)) or len(w) != 2:
            return False, f"Window at index {idx} must be a 2-element sequence [start, end], got {w}"
        valid, err = validate_window(w[0], w[1], duration, tolerance=tolerance)
        if not valid:
            return False, f"Window at index {idx} invalid: {err}"

    return True, None
