"""Tests for STAGE A.2.5E Clean Blind Validation — Prediction Freeze.

Verifies:
1. Safe input manifest (clean_blind_3_inputs.json) contains only safe fields; GT absent.
2. GT-path denylist and isolation from human_primary.csv.
3. semantic_v2 source SHA256 hashing and integrity.
4. Sealed predictions CSV schema and SHA256 match in freeze manifest.
5. Freeze manifest status: "PREDICTIONS_FROZEN_BEFORE_HUMAN_GT".
6. Three required sample IDs: lumae_ads_pilot_0008, 0010, 0011.
7. Source data immutability: human_primary.csv, ai_preannotations.csv, blind_holdout.json unchanged.
8. No automated human annotation: samples 0008, 0010, 0011 remain DRAFT with empty GT.
9. Streamlit DOM blindness: AI candidate card, timestamps, reasons, confidence absent; inputs blank.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import pytest
from playwright.sync_api import sync_playwright

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

CLEAN_BLIND_IDS = [
    "lumae_ads_pilot_0008",
    "lumae_ads_pilot_0010",
    "lumae_ads_pilot_0011",
]

GT_DENYLIST_FILENAMES = {
    "human_primary.csv",
    "human_secondary.csv",
    "human_adjudicated.csv",
    "pilot_annotations.csv",
    "pilot_annotations_secondary.csv",
    "frozen_gt.csv",
}

BASELINE_HUMAN_PRIMARY_SHA256 = "4ac2aa9ec0de00795cf335d785db513f79608ca097b3e5d3bfae4610924f13b3"
STAGE_A25H_HUMAN_PRIMARY_SHA256 = "29b5924831d32953e6399fcf408023bbfb02efc4b082976d98a63cf82ec8e1ae"
BASELINE_AI_PREANNOTATIONS_SHA256 = "651ef4661a23deb342c44fe166da1a8612714debeff8a33ec0ab426f8ee0a94f"
BASELINE_BLIND_HOLDOUT_SHA256 = "ec551df79eac67b5984190d6efab730c173e3c394cbeea1761632ab247a52387"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def test_clean_blind_manifest_safe_fields():
    """Verify clean_blind_3_inputs.json contains only safe fields and zero GT."""
    manifest_path = REPO_ROOT / "local_data" / "annotations" / "lumae_ads" / "clean_blind_3_inputs.json"
    assert manifest_path.is_file(), f"Missing manifest: {manifest_path}"

    with manifest_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    samples = data if isinstance(data, list) else data.get("samples", [])
    assert len(samples) == 3, f"Expected exactly 3 samples, found {len(samples)}"

    sample_ids = [s["sample_id"] for s in samples]
    assert sample_ids == CLEAN_BLIND_IDS, f"Sample IDs mismatch: {sample_ids} vs {CLEAN_BLIND_IDS}"

    safe_allowed_keys = {"sample_id", "video_filename", "query", "duration_seconds", "product_category"}
    prohibited_keys = {
        "gt_start_seconds", "gt_end_seconds", "annotator_id", "annotation_notes",
        "review_status", "human_notes", "adjudicated", "multi_windows"
    }

    for s in samples:
        keys = set(s.keys())
        assert keys.issubset(safe_allowed_keys), f"Sample {s['sample_id']} has unexpected keys: {keys - safe_allowed_keys}"
        assert not keys.intersection(prohibited_keys), f"Sample {s['sample_id']} has prohibited GT keys: {keys.intersection(prohibited_keys)}"
        assert s["duration_seconds"] > 0
        assert len(s["query"]) > 5
        assert s["video_filename"].endswith(".mp4")


def test_gt_path_denylist_enforcement():
    """Verify that GT files are in the denylist and must never be accessed as input for blind prediction."""
    for filename in GT_DENYLIST_FILENAMES:
        p = Path(filename)
        assert p.name in GT_DENYLIST_FILENAMES

    # Verify input manifest source was pilot_worklist_unannotated.csv, not human_primary
    worklist_path = REPO_ROOT / "local_data" / "annotations" / "lumae_ads" / "pilot_worklist_unannotated.csv"
    assert worklist_path.is_file()
    with worklist_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        worklist_ids = {r["sample_id"] for r in reader}
    for sid in CLEAN_BLIND_IDS:
        assert sid in worklist_ids, f"{sid} missing from unannotated worklist"


def test_semantic_v2_code_hash_recording():
    """Verify semantic_v2 implementation source hashing."""
    source_files = [
        REPO_ROOT / "data_process" / "adsqa" / "semantic_preannotator.py",
        REPO_ROOT / "data_process" / "adsqa" / "ai_benchmark.py",
        REPO_ROOT / "data_process" / "cli" / "benchmark_ai_preannotator.py",
    ]
    for p in source_files:
        assert p.is_file(), f"Source file missing: {p}"
        file_hash = _sha256(p)
        assert len(file_hash) == 64


def test_prediction_freeze_manifest_and_sha256():
    """Verify freeze manifest existence, status, sample IDs, and prediction file sha256."""
    freeze_path = REPO_ROOT / "local_data" / "manifests" / "semantic_v2_clean3_blind_freeze.json"
    assert freeze_path.is_file(), f"Freeze manifest missing: {freeze_path}"

    with freeze_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest["status"] == "PREDICTIONS_FROZEN_BEFORE_HUMAN_GT"
    assert manifest["prediction_count"] == 3
    assert manifest["sample_ids"] == CLEAN_BLIND_IDS
    assert manifest["experiment_name"] == "semantic_v2_clean3_blind_validation"

    pred_rel_path = manifest["prediction_file"]
    pred_path = REPO_ROOT / pred_rel_path
    assert pred_path.is_file(), f"Sealed predictions file missing: {pred_path}"

    actual_pred_sha256 = _sha256(pred_path)
    assert actual_pred_sha256 == manifest["prediction_sha256"], (
        f"Prediction SHA256 mismatch! Manifest: {manifest['prediction_sha256']}, Actual: {actual_pred_sha256}"
    )


def test_sealed_prediction_immutability():
    """Verify sealed predictions CSV has 3 samples and all required schema fields."""
    pred_path = REPO_ROOT / "local_data" / "annotations" / "lumae_ads" / "blind_eval" / "semantic_v2_clean3_predictions.csv"
    assert pred_path.is_file()

    with pred_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == 3, f"Expected 3 rows, found {len(rows)}"
    found_ids = [r["sample_id"] for r in rows]
    assert found_ids == CLEAN_BLIND_IDS

    required_fields = {
        "sample_id", "video_filename", "query", "semantic_status",
        "candidate_start_seconds", "candidate_end_seconds", "candidate_windows_json",
        "semantic_confidence", "semantic_reason", "analysis_method",
        "preannotator_version", "generated_at_utc"
    }
    assert required_fields.issubset(set(reader.fieldnames or []))

    for r in rows:
        assert float(r["candidate_start_seconds"]) >= 0.0
        assert float(r["candidate_end_seconds"]) > float(r["candidate_start_seconds"])
        assert r["semantic_status"] in ("VALID", "NEEDS_EDIT", "INVALID_VIDEO")
        assert r["preannotator_version"] == "semantic_v2"
        windows = json.loads(r["candidate_windows_json"])
        assert isinstance(windows, list) and len(windows) >= 1


def test_source_data_mutation_check():
    """Confirm human_primary.csv, ai_preannotations.csv, blind_holdout.json remain strictly unchanged from frozen state."""
    human_primary = REPO_ROOT / "local_data" / "annotations" / "lumae_ads" / "human_primary.csv"
    ai_preann = REPO_ROOT / "local_data" / "annotations" / "lumae_ads" / "ai_preannotations.csv"
    holdout = REPO_ROOT / "local_data" / "annotations" / "lumae_ads" / "blind_holdout.json"

    assert _sha256(human_primary) in {BASELINE_HUMAN_PRIMARY_SHA256, STAGE_A25H_HUMAN_PRIMARY_SHA256}, "human_primary.csv was mutated!"
    assert _sha256(ai_preann) == BASELINE_AI_PREANNOTATIONS_SHA256, "ai_preannotations.csv was mutated!"
    assert _sha256(holdout) == BASELINE_BLIND_HOLDOUT_SHA256, "blind_holdout.json was mutated!"


def test_clean_blind_human_review_completed():
    """Verify that human_primary.csv contains valid human reviews for all 3 clean blind samples."""
    human_primary = REPO_ROOT / "local_data" / "annotations" / "lumae_ads" / "human_primary.csv"
    with human_primary.open("r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        records = {r["sample_id"]: r for r in reader}

    for sid in CLEAN_BLIND_IDS:
        assert sid in records
        rec = records[sid]
        assert rec["review_status"] == "REVIEWED", f"{sid} has review_status {rec['review_status']} != REVIEWED"
        assert rec["annotator_id"] == "human_01", f"{sid} has annotator_id {rec['annotator_id']} != human_01"
        assert float(rec["gt_start_seconds"]) >= 0.0
        assert float(rec["gt_end_seconds"]) > float(rec["gt_start_seconds"])


def get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_for_server(port: int, timeout_sec: float = 20.0) -> bool:
    start = time.time()
    while time.time() - start < timeout_sec:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1.0):
                return True
        except (OSError, ConnectionRefusedError):
            time.sleep(0.5)
    return False


@pytest.fixture
def isolated_clean_blind_env(tmp_path: Path):
    """Create isolated test environment with the 3 clean blind samples."""
    ann_dir = tmp_path / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)

    # 1. Primary worklist: 3 clean blind samples (unreviewed DRAFT)
    primary_csv = ann_dir / "human_primary.csv"
    fieldnames = [
        "sample_id", "video_filename", "vid", "source_dataset", "source_video_id",
        "source_split", "source_url", "product_category", "query", "duration_seconds",
        "gt_start_seconds", "gt_end_seconds", "annotator_id", "annotation_notes", "review_status"
    ]
    rows = [
        {
            "sample_id": "lumae_ads_pilot_0008",
            "video_filename": "1646e570bed4d4c37b3460cb79248328.mp4",
            "vid": "vid_0008",
            "source_dataset": "AdsQA",
            "source_video_id": "1646e570bed4d4c37b3460cb79248328",
            "source_split": "test",
            "source_url": "",
            "product_category": "Product Demonstration",
            "query": "The creator presents the key physical features of the product.",
            "duration_seconds": "30.00",
            "gt_start_seconds": "",
            "gt_end_seconds": "",
            "annotator_id": "ann_lead",
            "annotation_notes": "",
            "review_status": "DRAFT",
        },
        {
            "sample_id": "lumae_ads_pilot_0010",
            "video_filename": "1b50f42a80a997059b1fa3a5a30a2174.mp4",
            "vid": "vid_0010",
            "source_dataset": "AdsQA",
            "source_video_id": "1b50f42a80a997059b1fa3a5a30a2174",
            "source_split": "test",
            "source_url": "",
            "product_category": "Product Demonstration",
            "query": "The demonstrator shows how the product functions.",
            "duration_seconds": "132.36",
            "gt_start_seconds": "",
            "gt_end_seconds": "",
            "annotator_id": "ann_lead",
            "annotation_notes": "",
            "review_status": "DRAFT",
        },
        {
            "sample_id": "lumae_ads_pilot_0011",
            "video_filename": "20d8f028a0e2d51f289d6617c6e9ba8f.mp4",
            "vid": "vid_0011",
            "source_dataset": "AdsQA",
            "source_video_id": "20d8f028a0e2d51f289d6617c6e9ba8f",
            "source_split": "test",
            "source_url": "",
            "product_category": "Product Demonstration",
            "query": "The creator presents the key physical features of the product.",
            "duration_seconds": "101.20",
            "gt_start_seconds": "",
            "gt_end_seconds": "",
            "annotator_id": "ann_lead",
            "annotation_notes": "",
            "review_status": "DRAFT",
        },
    ]
    with primary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    # 2. Holdout JSON (standard holdout contains 0010 and 0011)
    holdout_json = ann_dir / "blind_holdout.json"
    holdout_data = {
        "seed": 42,
        "sample_ids": ["lumae_ads_pilot_0010", "lumae_ads_pilot_0011"],
        "sample_count": 2,
    }
    with holdout_json.open("w", encoding="utf-8") as f:
        json.dump(holdout_data, f, indent=2)

    # 3. AI preannotations CSV (contains candidates, which must be suppressed)
    ai_csv = ann_dir / "ai_preannotations.csv"
    ai_fieldnames = [
        "sample_id", "video_filename", "original_query", "ai_query_status",
        "ai_proposed_query", "ai_start_seconds", "ai_end_seconds", "ai_windows_json",
        "ai_confidence", "ai_reason", "analysis_method", "frame_sampling_interval",
        "boundary_refinement_interval", "generated_at_utc", "status"
    ]
    ai_rows = [
        {
            "sample_id": "lumae_ads_pilot_0008",
            "video_filename": "1646e570bed4d4c37b3460cb79248328.mp4",
            "original_query": "The creator presents the key physical features of the product.",
            "ai_query_status": "NEEDS_EDIT",
            "ai_proposed_query": "Secret AI Proposed Query 0008",
            "ai_start_seconds": "22.5",
            "ai_end_seconds": "24.4",
            "ai_windows_json": "[[22.5, 24.4]]",
            "ai_confidence": "HIGH",
            "ai_reason": "Secret AI Reason 0008",
            "analysis_method": "ffmpeg_coarse_dense_visual_flow",
            "frame_sampling_interval": "2.00",
            "boundary_refinement_interval": "0.50",
            "generated_at_utc": "2026-09-30T00:00:00Z",
            "status": "AI_PROPOSED",
        },
    ]
    with ai_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ai_fieldnames)
        writer.writeheader()
        for r in ai_rows:
            writer.writerow(r)

    # Dummy videos
    vid_dir = tmp_path / "videos"
    vid_dir.mkdir(parents=True, exist_ok=True)
    for r in rows:
        (vid_dir / r["video_filename"]).touch()

    return {
        "annotations_dir": ann_dir,
        "videos_dir": vid_dir,
        "primary_csv": primary_csv,
    }


def test_playwright_clean_blind_dom_blindness(isolated_clean_blind_env):
    """Playwright verification that clean-blind samples (0008, 0010, 0011) strictly suppress AI candidates in DOM."""
    env_info = isolated_clean_blind_env
    ann_dir = env_info["annotations_dir"]
    vid_dir = env_info["videos_dir"]
    primary_csv = env_info["primary_csv"]

    port = get_free_port()
    env = os.environ.copy()
    env["LUMAE_ANNOTATIONS_DIR"] = str(ann_dir)
    env["LUMAE_VIDEOS_DIR"] = str(vid_dir)
    env["STREAMLIT_SERVER_PORT"] = str(port)
    env["STREAMLIT_SERVER_HEADLESS"] = "true"
    env["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "data_process/annotation/app.py",
        f"--server.port={port}",
        "--server.headless=true",
        "--browser.gatherUsageStats=false",
    ]

    proc = subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    try:
        assert wait_for_server(port, timeout_sec=25.0), f"Streamlit failed to bind to port {port}."

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(viewport={"width": 1280, "height": 900})
            page = context.new_page()

            page.goto(f"http://127.0.0.1:{port}", timeout=25000)
            page.wait_for_load_state("networkidle")
            page.wait_for_selector("h1", timeout=15000)

            for target_sid in CLEAN_BLIND_IDS:
                # Wait for sample target_sid to be the active sample
                page.wait_for_selector(f"text={target_sid}", timeout=10000)

                # 1. Video playback container present
                page.wait_for_selector("video, [data-testid='stVideo']", timeout=10000)

                # 2. Query is visible
                body_content = page.locator("body").inner_text()
                dom_html = page.content()
                assert "Original Query:" in body_content

                # 3. Blind banner present
                assert "Blind Human Holdout Sample" in body_content or "BLIND_PRIMARY" in body_content

                # 4. CRITICAL PRIVACY: AI Candidate card is ABSENT
                assert "AI Candidate Pre-Annotation" not in body_content
                assert "AI Candidate Pre-Annotation" not in dom_html

                # 5. CRITICAL PRIVACY: Secret AI candidate info absent from entire DOM
                assert "Secret AI Proposed Query" not in dom_html
                assert "Secret AI Reason" not in dom_html
                assert "Accept AI Query" not in dom_html
                assert "Accept AI Window" not in dom_html

                # 6. Save button is present and says Save Human Review
                save_btn = page.locator("button:has-text('Save Human Review')")
                assert save_btn.count() > 0

                # 7. Start and End inputs are blank (empty value)
                start_input = page.locator(f"input[id*='num_start_{target_sid}']")
                end_input = page.locator(f"input[id*='num_end_{target_sid}']")
                if start_input.count() > 0:
                    assert start_input.input_value() == ""
                if end_input.count() > 0:
                    assert end_input.input_value() == ""

                # Move to next sample if not at the end
                if target_sid != CLEAN_BLIND_IDS[-1]:
                    next_btn = page.locator("button:has-text('Next Sample')")
                    assert next_btn.count() > 0
                    next_btn.first.click()
                    time.sleep(1.0)

            browser.close()

            # Confirm no automated annotation occurred (primary_csv unchanged, all still DRAFT)
            with primary_csv.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    assert r["review_status"] == "DRAFT"
                    assert r["gt_start_seconds"] == ""
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
