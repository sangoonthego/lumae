"""Unit tests for Stage A.2 AdsQA ingestion, validation, and annotation QC."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from data_process.adapters.lumae_ads import LumaeAdsAdapter
from data_process.adsqa.annotation_qc import compute_temporal_iou, evaluate_dual_annotations
from data_process.adsqa.candidate_selection import classify_text
from data_process.adsqa.video_validation import MAX_PILOT_DURATION_SECONDS, probe_video_file
from data_process.cli.download_adsqa_subset import download_single_video
from data_process.cli.fetch_adsqa_metadata import validate_video_urls_structure
from data_process.cli.prepare_lumae_ads_pilot import generate_pilot_worklist
from data_process.schemas.temporal_sample import ReasonCode


def test_adsqa_metadata_parsing_valid():
    raw_data = [
        {"url": "https://video.example.com/1", "target_name": "vid001.mp4"},
        {"url": "https://video.example.com/2", "target_name": "vid002.mp4"},
    ]
    validated = validate_video_urls_structure(raw_data)
    assert len(validated) == 2
    assert validated[0]["target_name"] == "vid001.mp4"
    assert validated[1]["url"] == "https://video.example.com/2"


def test_adsqa_metadata_parsing_malformed():
    # Non-list
    with pytest.raises(ValueError, match="root must be a list"):
        validate_video_urls_structure({"url": "foo"})

    # Empty list
    with pytest.raises(ValueError, match="is empty"):
        validate_video_urls_structure([])

    # Missing target_name
    with pytest.raises(ValueError, match="missing 'target_name'"):
        validate_video_urls_structure([{"url": "https://video.example.com/1"}])

    # Empty url
    with pytest.raises(ValueError, match="invalid or missing 'url'"):
        validate_video_urls_structure([{"url": "  ", "target_name": "vid.mp4"}])


def test_source_split_preservation(tmp_path: Path):
    manifest_file = tmp_path / "source_manifest.json"
    manifest_data = {
        "dataset": "AdsQA",
        "videos": [
            {
                "source_video_id": "v_train_01",
                "target_name": "v_train_01.mp4",
                "source_url": "https://video.example.com/t1",
                "source_split": "train",
            },
            {
                "source_video_id": "v_test_01",
                "target_name": "v_test_01.mp4",
                "source_url": "https://video.example.com/te1",
                "source_split": "test",
            },
        ],
    }
    manifest_file.write_text(json.dumps(manifest_data), encoding="utf-8")

    loaded = json.loads(manifest_file.read_text(encoding="utf-8"))
    splits = {v["source_video_id"]: v["source_split"] for v in loaded["videos"]}
    assert splits["v_train_01"] == "train"
    assert splits["v_test_01"] == "test"


def test_download_status_records(tmp_path: Path):
    out_dir = tmp_path / "videos"
    out_dir.mkdir()

    # Test HTTP 404 maps to UNAVAILABLE
    cand_404 = {
        "source_video_id": "missing_vid",
        "source_url": "https://video.example.com/missing",
        "source_split": "test",
        "category": "Household Supply",
    }

    import urllib.error
    with patch("urllib.request.urlopen", side_effect=urllib.error.HTTPError("url", 404, "Not Found", None, None)):  # type: ignore
        res = download_single_video(cand_404, out_dir, timeout=5, user_agent="test")
        assert res["download_status"] == "UNAVAILABLE"
        assert res["http_status"] == 404
        assert "404" in res["error_reason"]

    # Test generic connection error maps to FAILED
    with patch("urllib.request.urlopen", side_effect=RuntimeError("Connection reset")):
        res = download_single_video(cand_404, out_dir, timeout=5, user_agent="test")
        assert res["download_status"] == "FAILED"
        assert "Connection reset" in res["error_reason"]


def test_ffprobe_metadata_parser(tmp_path: Path):
    fake_video = tmp_path / "fake.mp4"
    fake_video.write_bytes(b"mock video binary content")

    ffprobe_json_output = {
        "format": {
            "duration": "24.500",
            "size": "1048576",
        },
        "streams": [
            {
                "codec_type": "video",
                "codec_name": "h264",
                "width": 1080,
                "height": 1920,
                "r_frame_rate": "30/1",
                "duration": "24.500",
            }
        ],
    }

    mock_proc = MagicMock()
    mock_proc.returncode = 0
    mock_proc.stdout = json.dumps(ffprobe_json_output)

    with patch("shutil.which", return_value="/usr/bin/ffprobe"), \
         patch("subprocess.run", return_value=mock_proc):
        probe = probe_video_file(fake_video)
        assert probe["duration"] == 24.5
        assert probe["width"] == 1080
        assert probe["height"] == 1920
        assert probe["codec"] == "h264"
        assert probe["fps"] == 30.0
        assert probe["has_visual_stream"] is True
        assert probe["is_model_compatible"] is True
        assert probe["incompatibility_reason"] is None


def test_duration_exceeds_150s_incompatibility(tmp_path: Path):
    fake_video = tmp_path / "long_video.mp4"
    fake_video.write_bytes(b"long video binary content")

    ffprobe_json = {
        "format": {"duration": "180.200", "size": "5000000"},
        "streams": [{"codec_type": "video", "codec_name": "h264", "width": 1920, "height": 1080}],
    }

    mock_proc = MagicMock()
    mock_proc.returncode = 0
    mock_proc.stdout = json.dumps(ffprobe_json)

    with patch("shutil.which", return_value="/usr/bin/ffprobe"), \
         patch("subprocess.run", return_value=mock_proc):
        probe = probe_video_file(fake_video)
        assert probe["duration"] == 180.2
        assert probe["is_model_compatible"] is False
        assert "exceeds" in probe["incompatibility_reason"].lower()


def test_candidate_category_classification():
    cat, status, reason = classify_text("Demonstrates the portable vacuum cleaner suction power on car seats.")
    assert "Cleaning" in cat or "Demonstration" in cat
    assert status == "ACCEPT"

    cat2, status2, reason2 = classify_text("Support our charity fundraising campaign for national wildlife preservation.")
    assert status2 == "REJECT"


def test_annotation_worklist_generation(tmp_path: Path):
    v_dir = tmp_path / "videos"
    v_dir.mkdir()
    fake_v = v_dir / "test_vid_123.mp4"
    fake_v.write_bytes(b"data")

    cand_csv = tmp_path / "candidate_review.csv"
    with cand_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "source_video_id",
                "source_split",
                "source_url",
                "category",
                "candidate_status",
                "candidate_reason",
                "duration_seconds",
                "download_status",
                "notes",
            ],
        )
        writer.writeheader()
        writer.writerow({
            "source_video_id": "test_vid_123",
            "source_split": "train",
            "source_url": "https://video.example.com/123",
            "category": "Home & Kitchen Appliance",
            "candidate_status": "ACCEPT",
            "candidate_reason": "Appliance demo",
            "duration_seconds": "18.5",
            "download_status": "DOWNLOADED",
            "notes": "Blender demo",
        })

    out_csv = tmp_path / "pilot_worklist.csv"

    mock_probe = {
        "file_path": str(fake_v),
        "duration": 18.5,
        "width": 1080,
        "height": 1920,
        "is_model_compatible": True,
    }

    with patch("data_process.cli.prepare_lumae_ads_pilot.probe_video_file", return_value=mock_probe):
        worklist = generate_pilot_worklist(v_dir, cand_csv, out_csv, limit=10)
        assert len(worklist) == 1
        entry = worklist[0]
        assert entry["source_video_id"] == "test_vid_123"
        assert entry["source_split"] == "train"
        assert entry["gt_start_seconds"] == ""  # Must be blank initially
        assert entry["gt_end_seconds"] == ""    # Must be blank initially


def test_blank_gt_remains_uningestable(tmp_path: Path):
    csv_file = tmp_path / "blank_gt.csv"
    with csv_file.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_id",
                "video_filename",
                "vid",
                "product_category",
                "query",
                "duration_seconds",
                "gt_start_seconds",
                "gt_end_seconds",
                "annotator_id",
                "annotation_notes",
                "review_status",
            ],
        )
        writer.writeheader()
        writer.writerow({
            "sample_id": "lumae_ads_blank_01",
            "video_filename": "vid01.mp4",
            "vid": "lumae_vid01",
            "product_category": "Electronics",
            "query": "The creator demonstrates the smartphone screen.",
            "duration_seconds": "15.0",
            "gt_start_seconds": "",  # Blank
            "gt_end_seconds": "",    # Blank
            "annotator_id": "ann_lead",
            "annotation_notes": "Needs human annotation",
            "review_status": "DRAFT",
        })

    adapter = LumaeAdsAdapter(check_video_exists=False)
    samples, rejections = adapter.process_csv(csv_file)
    assert len(samples) == 0
    assert len(rejections) == 1
    assert rejections[0].reason_code == ReasonCode.INVALID_WINDOW


def test_valid_annotated_row_ingestion_and_provenance(tmp_path: Path):
    csv_file = tmp_path / "valid_gt.csv"
    with csv_file.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_id",
                "video_filename",
                "vid",
                "source_dataset",
                "source_video_id",
                "source_split",
                "source_url",
                "product_category",
                "query",
                "duration_seconds",
                "gt_start_seconds",
                "gt_end_seconds",
                "annotator_id",
                "annotation_notes",
                "review_status",
            ],
        )
        writer.writeheader()
        writer.writerow({
            "sample_id": "lumae_ads_pilot_0001",
            "video_filename": "vid01.mp4",
            "vid": "lumae_vid01",
            "source_dataset": "AdsQA",
            "source_video_id": "src_hash_999",
            "source_split": "train",
            "source_url": "https://video.example.com/src999",
            "product_category": "Cleaning",
            "query": "The creator shows how the vacuum works.",
            "duration_seconds": "20.0",
            "gt_start_seconds": "4.5",
            "gt_end_seconds": "12.0",
            "annotator_id": "ann_lead",
            "annotation_notes": "Verified demonstration window",
            "review_status": "REVIEWED",
        })

    adapter = LumaeAdsAdapter(check_video_exists=False)
    samples, rejections = adapter.process_csv(csv_file)
    assert len(rejections) == 0
    assert len(samples) == 1

    sample = samples[0]
    assert sample.qid == "lumae_ads_pilot_0001"
    assert sample.vid == "lumae_vid01"
    assert sample.query == "The creator shows how the vacuum works."
    assert sample.duration == 20.0
    assert sample.relevant_windows == [[4.5, 12.0]]
    assert sample.source == "lumae_product_ads"
    assert sample.intent == "product_demo"
    assert sample.saliency_scores is None  # Strict NULL requirement

    # Check source provenance preserved in metadata
    meta = sample.metadata
    assert meta["source_dataset"] == "AdsQA"
    assert meta["source_video_id"] == "src_hash_999"
    assert meta["source_split"] == "train"
    assert meta["source_url"] == "https://video.example.com/src999"
    assert meta["product_category"] == "Cleaning"


def test_dual_annotation_tiou_calculation():
    # Exact match -> tIoU = 1.0
    w1 = [[2.0, 8.0]]
    w2 = [[2.0, 8.0]]
    assert compute_temporal_iou(w1, w2) == 1.0

    # Partial overlap
    # [2.0, 8.0] and [4.0, 10.0]: intersection [4.0, 8.0] = 4.0; union [2.0, 10.0] = 8.0 -> tIoU = 0.5
    w3 = [[4.0, 10.0]]
    assert compute_temporal_iou(w1, w3) == 0.5

    # Completely disjoint -> tIoU = 0.0
    w4 = [[12.0, 16.0]]
    assert compute_temporal_iou(w1, w4) == 0.0

    # Multi-window support
    w_multi_a = [[2.0, 4.0], [8.0, 10.0]]
    w_multi_b = [[2.0, 4.0], [8.0, 10.0]]
    assert compute_temporal_iou(w_multi_a, w_multi_b) == 1.0

    # Dual annotation agreement evaluation
    prim = [
        {"sample_id": "s1", "vid": "v1", "gt_start_seconds": 2.0, "gt_end_seconds": 8.0, "annotator_id": "a1"},
        {"sample_id": "s2", "vid": "v2", "gt_start_seconds": 2.0, "gt_end_seconds": 8.0, "annotator_id": "a1"},
    ]
    sec = [
        {"sample_id": "s1", "vid": "v1", "gt_start_seconds": 2.2, "gt_end_seconds": 8.0, "annotator_id": "a2"},  # high agreement
        {"sample_id": "s2", "vid": "v2", "gt_start_seconds": 6.0, "gt_end_seconds": 12.0, "annotator_id": "a2"}, # low agreement
    ]

    report = evaluate_dual_annotations(prim, sec, acceptance_threshold=0.70)
    assert report["double_annotated_count"] == 2
    assert report["high_agreement_count"] == 1
    assert report["adjudication_required_count"] == 1
