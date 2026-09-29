"""Deterministic video-level splitting to prevent train/val leakage."""

from __future__ import annotations

from collections import defaultdict
import random
from typing import Any, Sequence

from ..schemas.temporal_sample import CanonicalTemporalSample
from ..validation.leakage import assert_no_video_leakage


def deterministic_video_split(
    samples: Sequence[CanonicalTemporalSample | dict[str, Any]],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[
    list[CanonicalTemporalSample | dict[str, Any]],
    list[CanonicalTemporalSample | dict[str, Any]],
    list[CanonicalTemporalSample | dict[str, Any]],
    dict[str, Any],
]:
    """Deterministically partition samples into train, val, and test splits by video ID (vid).

    All queries associated with the same vid are guaranteed to be placed into the same split.

    Args:
        samples: Sequence of canonical samples or dictionaries.
        train_ratio: Fraction of videos assigned to train split.
        val_ratio: Fraction of videos assigned to val split.
        test_ratio: Fraction of videos assigned to test split.
        seed: Random seed for deterministic shuffling.

    Returns:
        (train_samples, val_samples, test_samples, split_stats)
    """
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-5:
        raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio}")

    # Group samples by vid
    by_vid: dict[str, list[CanonicalTemporalSample | dict[str, Any]]] = defaultdict(list)
    for s in samples:
        vid = s.vid if isinstance(s, CanonicalTemporalSample) else str(s["vid"]).strip()
        by_vid[vid].append(s)

    unique_vids = sorted(list(by_vid.keys()))
    rng = random.Random(seed)
    rng.shuffle(unique_vids)

    n_vids = len(unique_vids)
    n_train = int(round(n_vids * train_ratio))
    n_val = int(round(n_vids * val_ratio))

    # Adjust rounding discrepancies
    if n_train + n_val > n_vids:
        n_val = n_vids - n_train

    train_vids = set(unique_vids[:n_train])
    val_vids = set(unique_vids[n_train : n_train + n_val])
    test_vids = set(unique_vids[n_train + n_val :])

    train_samples: list[CanonicalTemporalSample | dict[str, Any]] = []
    val_samples: list[CanonicalTemporalSample | dict[str, Any]] = []
    test_samples: list[CanonicalTemporalSample | dict[str, Any]] = []

    for vid in unique_vids:
        vid_samples = by_vid[vid]
        if vid in train_vids:
            train_samples.extend(vid_samples)
        elif vid in val_vids:
            val_samples.extend(vid_samples)
        else:
            test_samples.extend(vid_samples)

    # Double-check assertion
    assert_no_video_leakage(train_samples, val_samples, test_samples)

    stats: dict[str, Any] = {
        "seed": seed,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "total_videos": n_vids,
        "total_samples": len(samples),
        "train": {
            "videos": len(train_vids),
            "samples": len(train_samples),
            "video_pct": round(len(train_vids) / max(1, n_vids) * 100, 2),
        },
        "val": {
            "videos": len(val_vids),
            "samples": len(val_samples),
            "video_pct": round(len(val_vids) / max(1, n_vids) * 100, 2),
        },
        "test": {
            "videos": len(test_vids),
            "samples": len(test_samples),
            "video_pct": round(len(test_vids) / max(1, n_vids) * 100, 2),
        },
    }

    return train_samples, val_samples, test_samples, stats
