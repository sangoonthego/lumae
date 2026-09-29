"""Tests for video-level leakage prevention and deterministic splitting."""

import pytest

from data_process.schemas.temporal_sample import CanonicalTemporalSample
from data_process.splitting.video_level import deterministic_video_split
from data_process.validation.leakage import (
    VideoLeakageError,
    assert_no_video_leakage,
    find_video_leakage,
)


def _make_sample(qid: int, vid: str) -> CanonicalTemporalSample:
    return CanonicalTemporalSample(
        qid=qid,
        vid=vid,
        query=f"Query {qid}",
        duration=20.0,
        relevant_windows=[[1.0, 5.0]],
    )


def test_leakage_detection_when_splits_overlap():
    train = [_make_sample(1, "video_A"), _make_sample(2, "video_B")]
    val = [_make_sample(3, "video_B"), _make_sample(4, "video_C")]

    leakage = find_video_leakage(train, val)
    assert "video_B" in leakage["train_val"]

    with pytest.raises(VideoLeakageError) as exc_info:
        assert_no_video_leakage(train, val)
    assert "video_B" in str(exc_info.value)


def test_leakage_detection_when_disjoint():
    train = [_make_sample(1, "video_A"), _make_sample(2, "video_B")]
    val = [_make_sample(3, "video_C")]
    test = [_make_sample(4, "video_D")]

    leakage = find_video_leakage(train, val, test)
    assert len(leakage["train_val"]) == 0
    assert len(leakage["train_test"]) == 0
    assert len(leakage["val_test"]) == 0

    # Should not raise
    assert_no_video_leakage(train, val, test)


def test_deterministic_video_split_guarantees_no_leakage():
    # 10 videos, multiple queries per video
    samples = []
    qid = 1
    for v_idx in range(10):
        vid = f"vid_{v_idx:02d}"
        for _ in range(3):  # 3 queries per video
            samples.append(_make_sample(qid, vid))
            qid += 1

    train, val, test, stats = deterministic_video_split(
        samples, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, seed=123
    )

    # 1. Total count preserved
    assert len(train) + len(val) + len(test) == len(samples)

    # 2. No leakage
    assert_no_video_leakage(train, val, test)

    # 3. Video counts match ratios
    assert stats["train"]["videos"] == 7
    assert stats["val"]["videos"] == 2
    assert stats["test"]["videos"] == 1


def test_deterministic_video_split_reproducibility():
    samples = [_make_sample(i, f"vid_{i % 5}") for i in range(20)]

    run1 = deterministic_video_split(samples, seed=42)
    run2 = deterministic_video_split(samples, seed=42)

    train_qids1 = [s.qid for s in run1[0]]
    train_qids2 = [s.qid for s in run2[0]]
    assert train_qids1 == train_qids2
