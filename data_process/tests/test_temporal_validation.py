"""Tests for temporal and schema validation rules."""

import pytest

from data_process.schemas.temporal_sample import CanonicalTemporalSample
from data_process.validation.schema import validate_canonical_dict, validate_canonical_sample
from data_process.validation.temporal import (
    validate_duration,
    validate_window,
    validate_windows,
)


def test_valid_temporal_window():
    ok, err = validate_window(start=2.0, end=5.5, duration=10.0)
    assert ok is True
    assert err is None


def test_negative_window_start():
    ok, err = validate_window(start=-1.0, end=5.0, duration=10.0)
    assert ok is False
    assert "non-negative" in (err or "")


def test_start_greater_than_or_equal_to_end():
    ok, err = validate_window(start=5.0, end=5.0, duration=10.0)
    assert ok is False
    assert "strictly less than" in (err or "")

    ok2, err2 = validate_window(start=6.0, end=5.0, duration=10.0)
    assert ok2 is False
    assert "strictly less than" in (err2 or "")


def test_window_exceeds_duration():
    ok, err = validate_window(start=5.0, end=15.0, duration=10.0)
    assert ok is False
    assert "exceeds video duration" in (err or "")


def test_duration_validation():
    assert validate_duration(10.5)[0] is True
    assert validate_duration(0.0)[0] is False
    assert validate_duration(-5.0)[0] is False
    assert validate_duration(float("nan"))[0] is False
    assert validate_duration(float("inf"))[0] is False


def test_validate_windows_empty():
    ok, err = validate_windows([], duration=10.0, allow_empty=False)
    assert ok is False
    assert "must not be empty" in (err or "")

    ok_allowed, _ = validate_windows([], duration=10.0, allow_empty=True)
    assert ok_allowed is True


def test_canonical_schema_validator_catches_empty_query():
    bad = {
        "qid": 1,
        "vid": "vid_1",
        "query": "   ",
        "duration": 10.0,
        "relevant_windows": [[1.0, 3.0]],
    }
    ok, errors = validate_canonical_dict(bad)
    assert ok is False
    assert any("query" in e.lower() for e in errors)


def test_canonical_schema_validator_catches_missing_vid():
    bad = {
        "qid": 1,
        "query": "Valid query",
        "duration": 10.0,
        "relevant_windows": [[1.0, 3.0]],
    }
    ok, errors = validate_canonical_dict(bad)
    assert ok is False
    assert any("vid" in e.lower() for e in errors)


def test_canonical_sample_validation_pass():
    sample = CanonicalTemporalSample(
        qid=1,
        vid="vid_1",
        query="Valid action description",
        duration=15.0,
        relevant_windows=[[2.0, 5.0], [7.0, 10.0]],
        relevant_clip_ids=[1, 2],
        saliency_scores=[[3.0, 4.0, 3.0], [2.0, 3.0, 4.0]],
    )
    ok, errors = validate_canonical_sample(
        sample, saliency_min=0.0, saliency_max=4.0, expected_annotator_count=3
    )
    assert ok is True
    assert len(errors) == 0


def test_canonical_schema_validator_catches_mismatched_clip_saliency():
    data = {
        "qid": 1,
        "vid": "vid_1",
        "query": "Valid query",
        "duration": 15.0,
        "relevant_windows": [[2.0, 5.0]],
        "relevant_clip_ids": [1, 2],
        "saliency_scores": [[3.0, 4.0, 3.0]],  # 1 row vs 2 clips
    }
    ok, errors = validate_canonical_dict(data)
    assert ok is False
    assert any("mismatched" in e.lower() for e in errors)


def test_canonical_schema_validator_catches_empty_saliency_row():
    data = {
        "qid": 1,
        "vid": "vid_1",
        "query": "Valid query",
        "duration": 15.0,
        "relevant_windows": [[2.0, 5.0]],
        "saliency_scores": [[]],
    }
    ok, errors = validate_canonical_dict(data)
    assert ok is False
    assert any("empty" in e.lower() for e in errors)


def test_canonical_schema_validator_catches_saliency_bounds():
    data = {
        "qid": 1,
        "vid": "vid_1",
        "query": "Valid query",
        "duration": 15.0,
        "relevant_windows": [[2.0, 5.0]],
        "saliency_scores": [[5.0, 3.0, 3.0]],  # 5.0 > 4.0 max
    }
    ok, errors = validate_canonical_dict(data, saliency_max=4.0)
    assert ok is False
    assert any("exceeds maximum" in e.lower() for e in errors)
