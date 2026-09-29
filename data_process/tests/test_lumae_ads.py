"""Tests for Lumae Product-Ads adapter, CSV parsing, and split safety."""

from pathlib import Path
import tempfile

import pytest

from data_process.adapters.lumae_ads import LumaeAdsAdapter
from data_process.schemas.temporal_sample import ReasonCode
from data_process.splitting.video_level import deterministic_video_split
from data_process.validation.leakage import assert_no_video_leakage


def test_lumae_ads_valid_sample():
    adapter = LumaeAdsAdapter()
    raw = {
        "sample_id": "lumae_ads_000001",
        "video_filename": "vacuum_demo.mp4",
        "vid": "vacuum_000001",
        "product_category": "appliances",
        "query": "The creator demonstrates the vacuum suction.",
        "duration_seconds": 22.5,
        "gt_start_seconds": 4.0,
        "gt_end_seconds": 12.0,
        "annotator_id": "ann_01",
    }
    sample, rejected = adapter.convert_record(raw, 0)
    assert rejected is None
    assert sample is not None
    assert sample.qid == "lumae_ads_000001"
    assert sample.vid == "vacuum_000001"
    assert sample.duration == 22.5
    assert sample.relevant_windows == [[4.0, 12.0]]
    assert sample.saliency_scores is None  # Strictly None for Lumae Ads
    assert sample.relevant_clip_ids is None


def test_lumae_ads_empty_query_rejected():
    adapter = LumaeAdsAdapter()
    raw = {
        "sample_id": "lumae_ads_000002",
        "vid": "vacuum_000001",
        "query": "   ",
        "duration_seconds": 20.0,
        "gt_start_seconds": 2.0,
        "gt_end_seconds": 5.0,
    }
    sample, rejected = adapter.convert_record(raw, 0)
    assert sample is None
    assert rejected is not None
    assert rejected.reason_code == ReasonCode.MISSING_QUERY


def test_lumae_ads_invalid_window_rejected():
    adapter = LumaeAdsAdapter()
    raw = {
        "sample_id": "lumae_ads_000003",
        "vid": "vacuum_000001",
        "query": "Valid query",
        "duration_seconds": 10.0,
        "gt_start_seconds": 8.0,
        "gt_end_seconds": 5.0,  # start > end
    }
    sample, rejected = adapter.convert_record(raw, 0)
    assert sample is None
    assert rejected is not None
    assert rejected.reason_code == ReasonCode.INVALID_WINDOW


def test_lumae_ads_window_exceeds_duration_rejected():
    adapter = LumaeAdsAdapter()
    raw = {
        "sample_id": "lumae_ads_000004",
        "vid": "vacuum_000001",
        "query": "Valid query",
        "duration_seconds": 10.0,
        "gt_start_seconds": 2.0,
        "gt_end_seconds": 15.0,  # end > duration
    }
    sample, rejected = adapter.convert_record(raw, 0)
    assert sample is None
    assert rejected is not None
    assert rejected.reason_code == ReasonCode.WINDOW_OUT_OF_RANGE


def test_lumae_ads_csv_parsing_and_multi_window_merging():
    csv_content = """sample_id,video_filename,vid,product_category,query,duration_seconds,gt_start_seconds,gt_end_seconds,annotator_id,annotation_notes,review_status
sample_01,vid_01.mp4,v1,gadget,The creator shows product use.,30.0,2.0,8.0,ann1,part 1,REVIEWED
sample_01,vid_01.mp4,v1,gadget,The creator shows product use.,30.0,14.0,20.0,ann1,part 2,REVIEWED
sample_02,vid_02.mp4,v2,beauty,The user applies product.,15.0,1.0,6.0,ann2,single part,DRAFT
"""
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_file = Path(tmpdir) / "test_annotations.csv"
        csv_file.write_text(csv_content, encoding="utf-8")

        adapter = LumaeAdsAdapter()
        samples, rejections = adapter.process_csv(csv_file)

        assert len(rejections) == 0
        assert len(samples) == 2  # sample_01 was merged into one sample with 2 windows

        sample_01 = next(s for s in samples if s.qid == "sample_01")
        assert sample_01.relevant_windows == [[2.0, 8.0], [14.0, 20.0]]
        assert sample_01.saliency_scores is None

        sample_02 = next(s for s in samples if s.qid == "sample_02")
        assert sample_02.relevant_windows == [[1.0, 6.0]]


def test_lumae_ads_video_level_split_safety():
    adapter = LumaeAdsAdapter()
    samples = []
    for i in range(12):
        s, rej = adapter.convert_record({
            "sample_id": f"lumae_ads_{i+1:06d}",
            "vid": f"video_{i:02d}",
            "query": f"Query for video {i}",
            "duration_seconds": 20.0,
            "gt_start_seconds": 2.0,
            "gt_end_seconds": 8.0,
        }, i)
        assert s is not None
        samples.append(s)

    train, val, test, stats = deterministic_video_split(
        samples, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42
    )

    assert len(train) + len(val) + len(test) == 12
    assert_no_video_leakage(train, val, test)
