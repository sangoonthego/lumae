"""Tests for QVHighlights and Ego4D NLQ dataset adapters."""

from data_process.adapters.ego4d_nlq import Ego4DNLQAdapter
from data_process.adapters.qvhighlights import QVHighlightsAdapter
from data_process.schemas.temporal_sample import ReasonCode


def test_qvhighlights_adapter_valid_record():
    adapter = QVHighlightsAdapter(namespace_vid=True)
    raw = {
        "qid": 10001,
        "query": "A person makes a salad",
        "duration": 40.0,
        "vid": "xYz_123",
        "relevant_windows": [[5.0, 12.0]],
        "relevant_clip_ids": [2, 3, 4],
        "saliency_scores": [
            [4.0, 3.0, 4.0],
            [3.0, 2.0, 3.0],
            [4.0, 4.0, 3.0],
        ],
    }
    sample, rejected = adapter.convert_record(raw, 0)
    assert rejected is None
    assert sample is not None
    assert sample.qid == 10001
    assert sample.vid == "qvh__xYz_123"
    assert sample.query == "A person makes a salad"
    assert sample.duration == 40.0
    assert sample.relevant_clip_ids == [2, 3, 4]
    assert sample.saliency_scores == [
        [4.0, 3.0, 4.0],
        [3.0, 2.0, 3.0],
        [4.0, 4.0, 3.0],
    ]


def test_qvhighlights_adapter_mismatched_clip_and_saliency_length():
    adapter = QVHighlightsAdapter()
    raw = {
        "qid": 10002,
        "query": "A person runs",
        "duration": 30.0,
        "vid": "v1",
        "relevant_windows": [[1.0, 5.0]],
        "relevant_clip_ids": [1, 2],
        "saliency_scores": [[3.0, 3.0, 3.0]],  # Only 1 row vs 2 clips
    }
    sample, rejected = adapter.convert_record(raw, 0)
    assert sample is None
    assert rejected is not None
    assert rejected.reason_code == ReasonCode.MISMATCHED_CLIP_SALIENCY


def test_qvhighlights_adapter_empty_saliency_row_rejected():
    adapter = QVHighlightsAdapter()
    raw = {
        "qid": 10003,
        "query": "A person runs",
        "duration": 30.0,
        "vid": "v1",
        "relevant_windows": [[1.0, 5.0]],
        "relevant_clip_ids": [1],
        "saliency_scores": [[]],  # Empty row
    }
    sample, rejected = adapter.convert_record(raw, 0)
    assert sample is None
    assert rejected is not None
    assert rejected.reason_code == ReasonCode.INVALID_SALIENCY


def test_qvhighlights_adapter_non_finite_saliency_rejected():
    adapter = QVHighlightsAdapter()
    raw = {
        "qid": 10004,
        "query": "A person runs",
        "duration": 30.0,
        "vid": "v1",
        "relevant_windows": [[1.0, 5.0]],
        "relevant_clip_ids": [1],
        "saliency_scores": [[1.0, float("nan"), 3.0]],
    }
    sample, rejected = adapter.convert_record(raw, 0)
    assert sample is None
    assert rejected is not None
    assert rejected.reason_code == ReasonCode.INVALID_SALIENCY


def test_qvhighlights_adapter_out_of_range_saliency_rejected():
    adapter = QVHighlightsAdapter()
    raw = {
        "qid": 10005,
        "query": "A person runs",
        "duration": 30.0,
        "vid": "v1",
        "relevant_windows": [[1.0, 5.0]],
        "relevant_clip_ids": [1],
        "saliency_scores": [[1.0, 5.0, 3.0]],  # 5.0 > 4.0 max
    }
    sample, rejected = adapter.convert_record(raw, 0)
    assert sample is None
    assert rejected is not None
    assert rejected.reason_code == ReasonCode.INVALID_SALIENCY


def test_qvhighlights_adapter_three_annotator_rule():
    adapter = QVHighlightsAdapter(strict_annotator_count=True)
    raw = {
        "qid": 10006,
        "query": "A person runs",
        "duration": 30.0,
        "vid": "v1",
        "relevant_windows": [[1.0, 5.0]],
        "relevant_clip_ids": [1],
        "saliency_scores": [[1.0, 2.0]],  # Only 2 annotators
    }
    sample, rejected = adapter.convert_record(raw, 0)
    assert sample is None
    assert rejected is not None
    assert rejected.reason_code == ReasonCode.INVALID_SALIENCY
    assert "Expected 3 annotators" in rejected.reason


def test_qvhighlights_adapter_missing_query_rejected():
    adapter = QVHighlightsAdapter()
    raw = {
        "qid": 10007,
        "query": "",
        "duration": 30.0,
        "vid": "v1",
        "relevant_windows": [[1.0, 5.0]],
    }
    sample, rejected = adapter.convert_record(raw, 1)
    assert sample is None
    assert rejected is not None
    assert rejected.reason_code == ReasonCode.MISSING_QUERY
    assert rejected.original_id == 10007


def test_qvhighlights_adapter_invalid_duration_rejected():
    adapter = QVHighlightsAdapter()
    raw = {
        "qid": 10008,
        "query": "Some query",
        "duration": -10.0,
        "vid": "v1",
        "relevant_windows": [[1.0, 5.0]],
    }
    sample, rejected = adapter.convert_record(raw, 2)
    assert sample is None
    assert rejected is not None
    assert rejected.reason_code == ReasonCode.INVALID_DURATION


def test_ego4d_nlq_adapter_valid_record():
    adapter = Ego4DNLQAdapter()
    raw = {
        "query": "Where did I put down the hammer?",
        "clip_uid": "clip_abc_789",
        "clip_duration": 60.0,
        "clip_start_sec": 14.5,
        "clip_end_sec": 22.0,
        "annotation_id": "ann_42",
    }
    sample, rejected = adapter.convert_record(raw, 0)
    assert rejected is None
    assert sample is not None
    assert sample.qid == "ego4d_nlq_ann_42"
    assert sample.vid == "ego4d__clip_abc_789"
    assert sample.relevant_windows == [[14.5, 22.0]]
    assert sample.saliency_scores is None  # Ego4D has no saliency scores


def test_ego4d_nlq_adapter_window_out_of_range_rejected():
    adapter = Ego4DNLQAdapter()
    raw = {
        "query": "Where is the pen?",
        "clip_uid": "clip_xyz",
        "clip_duration": 30.0,
        "clip_start_sec": 10.0,
        "clip_end_sec": 55.0,  # Exceeds 30s
    }
    sample, rejected = adapter.convert_record(raw, 0)
    assert sample is None
    assert rejected is not None
    assert rejected.reason_code == ReasonCode.WINDOW_OUT_OF_RANGE
