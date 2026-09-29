"""Video-level leakage prevention across dataset splits."""

from __future__ import annotations

from typing import Any, Sequence

from ..schemas.temporal_sample import CanonicalTemporalSample


class VideoLeakageError(Exception):
    """Raised when underlying videos are shared across evaluation/training splits."""
    def __init__(self, message: str, leaked_videos: dict[str, set[str]]):
        super().__init__(message)
        self.leaked_videos = leaked_videos


def _extract_vids(samples: Sequence[CanonicalTemporalSample | dict[str, Any]]) -> set[str]:
    vids: set[str] = set()
    for s in samples:
        if isinstance(s, CanonicalTemporalSample):
            vids.add(s.vid)
        elif isinstance(s, dict) and "vid" in s:
            vids.add(str(s["vid"]).strip())
        else:
            raise TypeError(f"Expected CanonicalTemporalSample or dict with 'vid', got {type(s)}")
    return vids


def find_video_leakage(
    train_samples: Sequence[CanonicalTemporalSample | dict[str, Any]],
    val_samples: Sequence[CanonicalTemporalSample | dict[str, Any]],
    test_samples: Sequence[CanonicalTemporalSample | dict[str, Any]] | None = None,
) -> dict[str, set[str]]:
    """Compute video ID intersections across splits.

    Returns:
        Dict mapping split pair names ('train_val', 'train_test', 'val_test')
        to sets of overlapping video IDs.
    """
    train_vids = _extract_vids(train_samples)
    val_vids = _extract_vids(val_samples)
    test_vids = _extract_vids(test_samples) if test_samples is not None else set()

    leakage: dict[str, set[str]] = {
        "train_val": train_vids.intersection(val_vids),
    }
    if test_samples is not None:
        leakage["train_test"] = train_vids.intersection(test_vids)
        leakage["val_test"] = val_vids.intersection(test_vids)

    return leakage


def assert_no_video_leakage(
    train_samples: Sequence[CanonicalTemporalSample | dict[str, Any]],
    val_samples: Sequence[CanonicalTemporalSample | dict[str, Any]],
    test_samples: Sequence[CanonicalTemporalSample | dict[str, Any]] | None = None,
) -> None:
    """Assert that underlying videos do not leak across splits.

    Raises:
        VideoLeakageError: if any split shares one or more video IDs.
    """
    leakage = find_video_leakage(train_samples, val_samples, test_samples)
    violating_pairs = {k: v for k, v in leakage.items() if len(v) > 0}

    if violating_pairs:
        details = ", ".join(
            f"{pair}: {len(vids)} leaked video(s) (e.g. {sorted(list(vids))[:3]})"
            for pair, vids in violating_pairs.items()
        )
        raise VideoLeakageError(
            f"Video-level data leakage detected across splits! Violations: {details}",
            leaked_videos=violating_pairs,
        )
