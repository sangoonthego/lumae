"""Tests for CanonicalTemporalSample and JSONL serialization."""

from pathlib import Path
import tempfile

import pytest

from data_process.io.jsonl import read_jsonl, write_jsonl, JsonlParseError
from data_process.schemas.temporal_sample import (
    CanonicalTemporalSample,
    ReasonCode,
    RejectedSampleRecord,
)


def test_canonical_sample_creation_and_dict_roundtrip():
    sample = CanonicalTemporalSample(
        qid=1001,
        vid="ego4d__test_01",
        query="Person picks up the keys",
        duration=25.5,
        relevant_windows=[[4.0, 8.2]],
        source="ego4d_nlq",
        original_id="orig_1001",
    )
    assert sample.qid == 1001
    assert sample.vid == "ego4d__test_01"
    assert sample.duration == 25.5
    assert sample.relevant_windows == [[4.0, 8.2]]
    assert sample.saliency_scores is None

    payload = sample.to_dict()
    assert payload["qid"] == 1001
    assert payload["vid"] == "ego4d__test_01"
    assert payload["relevant_windows"] == [[4.0, 8.2]]
    assert "saliency_scores" not in payload

    recovered = CanonicalTemporalSample.from_dict(payload)
    assert recovered.qid == sample.qid
    assert recovered.vid == sample.vid
    assert recovered.duration == sample.duration
    assert recovered.relevant_windows == sample.relevant_windows
    assert recovered.saliency_scores is None


def test_canonical_sample_nested_saliency_preservation():
    raw_saliency = [
        [4.0, 3.0, 4.0],
        [3.0, 2.0, 3.0],
        [4.0, 4.0, 3.0],
    ]
    sample = CanonicalTemporalSample(
        qid=2001,
        vid="qvh__sample_vid",
        query="Cooking pasta in the kitchen",
        duration=40.0,
        relevant_windows=[[10.0, 16.0]],
        relevant_clip_ids=[5, 6, 7],
        saliency_scores=raw_saliency,
        source="qvhighlights",
    )
    assert sample.saliency_scores == raw_saliency
    payload = sample.to_dict()
    assert payload["saliency_scores"] == raw_saliency

    recovered = CanonicalTemporalSample.from_dict(payload)
    assert recovered.saliency_scores == raw_saliency
    # Confirm exact preservation of each annotator score (no averaging)
    assert recovered.saliency_scores[0] == [4.0, 3.0, 4.0]
    assert recovered.saliency_scores[1] == [3.0, 2.0, 3.0]


def test_canonical_sample_invalid_saliency_types():
    # Saliency row cannot be empty
    with pytest.raises(ValueError):
        CanonicalTemporalSample(
            qid=1, vid="v", query="q", duration=10.0, relevant_windows=[[1.0, 2.0]],
            saliency_scores=[[]],
        )

    # Saliency score must be finite
    with pytest.raises(ValueError):
        CanonicalTemporalSample(
            qid=1, vid="v", query="q", duration=10.0, relevant_windows=[[1.0, 2.0]],
            saliency_scores=[[1.0, float("nan")]],
        )


def test_canonical_sample_string_qid_normalization():
    sample = CanonicalTemporalSample(
        qid="54321",
        vid="vid_abc",
        query="Some action",
        duration=10.0,
        relevant_windows=[[1.0, 3.0]],
    )
    assert sample.qid == 54321


def test_rejected_sample_record():
    record = RejectedSampleRecord(
        source="ego4d_nlq",
        original_id="clip_99",
        reason_code=ReasonCode.MISSING_QUERY,
        reason="Query text was empty string",
        raw_reference="{'clip_uid': 'clip_99'}",
    )
    d = record.to_dict()
    assert d["source"] == "ego4d_nlq"
    assert d["reason_code"] == "MISSING_QUERY"
    assert d["original_id"] == "clip_99"


def test_jsonl_io_roundtrip():
    samples = [
        CanonicalTemporalSample(
            qid=1, vid="v1", query="Query one", duration=12.0, relevant_windows=[[1.0, 4.0]]
        ),
        CanonicalTemporalSample(
            qid=2, vid="v2", query="Query two", duration=18.0, relevant_windows=[[5.0, 9.0]]
        ),
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        jsonl_path = Path(tmpdir) / "test_out.jsonl"
        count = write_jsonl(jsonl_path, samples)
        assert count == 2
        assert jsonl_path.is_file()

        read_back = list(read_jsonl(jsonl_path))
        assert len(read_back) == 2
        assert read_back[0]["qid"] == 1
        assert read_back[1]["qid"] == 2


def test_jsonl_malformed_line_raises_clear_error():
    with tempfile.TemporaryDirectory() as tmpdir:
        bad_jsonl = Path(tmpdir) / "bad.jsonl"
        bad_jsonl.write_text('{"qid": 1}\n{not valid json\n{"qid": 2}\n', encoding="utf-8")

        with pytest.raises(JsonlParseError) as exc_info:
            list(read_jsonl(bad_jsonl))
        assert exc_info.value.line_num == 2
        assert "not valid json" in str(exc_info.value)
