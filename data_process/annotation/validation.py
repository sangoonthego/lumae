"""Validation rules for human temporal boundaries and query revisions."""

from __future__ import annotations

import re
from typing import Any

DISALLOWED_SUBJECTIVE_PATTERNS = [
    r"\bbest (?:part|moment|scene)\b",
    r"\bviral (?:hook|moment|scene)\b",
    r"\bmost engaging\b",
    r"\bfunny moment\b",
    r"\bcoolest part\b",
]


def validate_temporal_boundaries(
    start: float | None,
    end: float | None,
    duration: float,
) -> tuple[bool, str | None]:
    """Validate a single temporal boundary interval against duration constraints."""
    if start is None or end is None:
        return False, "Both START and END timestamps are required."

    try:
        s = float(start)
        e = float(end)
    except (ValueError, TypeError):
        return False, "START and END must be valid numeric values."

    if s < 0.0:
        return False, f"START timestamp ({s:.1f}s) cannot be negative."

    if s >= e:
        return False, f"START timestamp ({s:.1f}s) must precede END timestamp ({e:.1f}s)."

    if (e - s) < 0.1:
        return False, f"Temporal window duration ({e - s:.2f}s) must be at least 0.1s."

    if duration > 0 and e > (duration + 0.1):
        return False, f"END timestamp ({e:.1f}s) exceeds video duration ({duration:.1f}s)."

    return True, None


def validate_multi_windows(
    windows: list[list[float]],
    duration: float,
) -> tuple[bool, str | None]:
    """Validate a collection of temporal windows for ordering, boundaries, and overlap."""
    if not windows:
        return False, "At least one temporal window is required."

    sorted_windows = sorted(windows, key=lambda w: w[0])
    for idx, w in enumerate(sorted_windows):
        if len(w) != 2:
            return False, f"Window index {idx} does not have exactly [start, end] format."
        ok, err = validate_temporal_boundaries(w[0], w[1], duration)
        if not ok:
            return False, f"Window {idx + 1} invalid: {err}"

        if idx > 0:
            prev_end = sorted_windows[idx - 1][1]
            curr_start = w[0]
            if curr_start < prev_end:
                return False, f"Window {idx + 1} overlaps with preceding window (starts at {curr_start:.1f}s, previous ends at {prev_end:.1f}s)."

    return True, None


def validate_query(
    query: str,
    query_status: str,
) -> tuple[bool, str | None]:
    """Validate query text for clarity, verifiability, and policy compliance."""
    status = str(query_status).upper()
    if status == "INVALID_VIDEO":
        return True, None

    cleaned = str(query).strip()
    if not cleaned:
        return False, "Query cannot be empty for reviewed product demonstration samples."

    if len(cleaned) < 5:
        return False, "Query must be a descriptive sentence of at least 5 characters."

    lower = cleaned.lower()
    for pat in DISALLOWED_SUBJECTIVE_PATTERNS:
        if re.search(pat, lower):
            match = re.findall(pat, lower)[0]
            return False, f"Subjective or unverifiable phrase '{match}' is disallowed by research protocol."

    return True, None
