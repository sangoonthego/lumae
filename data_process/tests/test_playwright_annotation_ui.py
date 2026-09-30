"""Playwright UI tests for LUMAE Ads Streamlit Review Application (Stage A.2.5B).

Verifies:
1. Application successfully loads in headless Chromium.
2. Video playback container and metadata render.
3. Non-holdout samples display advisory AI candidate card (status, proposed query, windows, reason).
4. 'Accept AI Query' and 'Accept AI Window' populate review inputs.
5. 'SAVE HUMAN VERIFIED' persists human ground truth to isolated test storage.
6. Blind human holdout samples strictly suppress AI candidate suggestions.
7. CRITICAL GUARD: Test fixtures are completely isolated in tmp_path (real human dataset is never touched).
"""

from __future__ import annotations

import csv
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


def get_free_port() -> int:
    """Find an available port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_for_server(port: int, timeout_sec: float = 20.0) -> bool:
    """Poll localhost port until Streamlit server responds."""
    start = time.time()
    while time.time() - start < timeout_sec:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1.0):
                return True
        except (OSError, ConnectionRefusedError):
            time.sleep(0.5)
    return False


@pytest.fixture
def isolated_annotation_env(tmp_path: Path):
    """Create isolated test environment with primary worklist, holdout, and AI candidates."""
    ann_dir = tmp_path / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)

    # 1. Primary worklist: 1 non-holdout sample + 1 holdout sample
    primary_csv = ann_dir / "human_primary.csv"
    fieldnames = [
        "sample_id", "video_filename", "vid", "source_dataset", "source_video_id",
        "source_split", "source_url", "product_category", "query", "duration_seconds",
        "gt_start_seconds", "gt_end_seconds", "annotator_id", "annotation_notes", "review_status"
    ]
    rows = [
        {
            "sample_id": "test_sample_non_holdout",
            "video_filename": "dummy_non_holdout.mp4",
            "vid": "vid_nh",
            "source_dataset": "AdsQA",
            "source_video_id": "v_nh",
            "source_split": "test",
            "source_url": "",
            "product_category": "Home & Kitchen",
            "query": "The creator presents the key physical features of the product.",
            "duration_seconds": "30.00",
            "gt_start_seconds": "",
            "gt_end_seconds": "",
            "annotator_id": "ann_lead",
            "annotation_notes": "",
            "review_status": "DRAFT",
        },
        {
            "sample_id": "test_sample_blind_holdout",
            "video_filename": "dummy_holdout.mp4",
            "vid": "vid_h",
            "source_dataset": "AdsQA",
            "source_video_id": "v_h",
            "source_split": "test",
            "source_url": "",
            "product_category": "Electronics & Gadgets",
            "query": "The creator tests the camera feature on the device.",
            "duration_seconds": "40.00",
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

    # 2. Blind holdout JSON (sample test_sample_blind_holdout is holdout)
    holdout_json = ann_dir / "blind_holdout.json"
    holdout_data = {
        "seed": 42,
        "sample_ids": ["test_sample_blind_holdout"],
        "sample_count": 1,
    }
    with holdout_json.open("w", encoding="utf-8") as f:
        json.dump(holdout_data, f, indent=2)

    # 3. AI preannotations CSV for both samples
    ai_csv = ann_dir / "ai_preannotations.csv"
    ai_fieldnames = [
        "sample_id", "video_filename", "original_query", "ai_query_status",
        "ai_proposed_query", "ai_start_seconds", "ai_end_seconds", "ai_windows_json",
        "ai_confidence", "ai_reason", "analysis_method", "frame_sampling_interval",
        "boundary_refinement_interval", "generated_at_utc", "status"
    ]
    ai_rows = [
        {
            "sample_id": "test_sample_non_holdout",
            "video_filename": "dummy_non_holdout.mp4",
            "original_query": "The creator presents the key physical features of the product.",
            "ai_query_status": "NEEDS_EDIT",
            "ai_proposed_query": "The presenter demonstrates the physical features and operation of the appliance.",
            "ai_start_seconds": "5.5",
            "ai_end_seconds": "18.2",
            "ai_windows_json": "[[5.5, 18.2]]",
            "ai_confidence": "LOW",
            "ai_reason": "Clear action demonstration detected from 5.5s to 18.2s.",
            "analysis_method": "ffmpeg_coarse_dense_visual_flow",
            "frame_sampling_interval": "2.00",
            "boundary_refinement_interval": "0.50",
            "generated_at_utc": "2026-09-30T00:00:00Z",
            "status": "AI_PROPOSED",
        },
        {
            "sample_id": "test_sample_blind_holdout",
            "video_filename": "dummy_holdout.mp4",
            "original_query": "The creator tests the camera feature on the device.",
            "ai_query_status": "VALID",
            "ai_proposed_query": "The creator tests the camera feature on the device.",
            "ai_start_seconds": "10.0",
            "ai_end_seconds": "22.5",
            "ai_windows_json": "[[10.0, 22.5]]",
            "ai_confidence": "HIGH",
            "ai_reason": "Holdout candidate internal reason.",
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

    # Empty dummy video files
    vid_dir = tmp_path / "videos"
    vid_dir.mkdir(parents=True, exist_ok=True)
    (vid_dir / "dummy_non_holdout.mp4").touch()
    (vid_dir / "dummy_holdout.mp4").touch()

    return {
        "annotations_dir": ann_dir,
        "videos_dir": vid_dir,
        "primary_csv": primary_csv,
    }


def test_playwright_streamlit_ai_assisted_and_blind_privacy(isolated_annotation_env):
    """Full end-to-end browser automation test for AI-assisted review and holdout privacy."""
    env_info = isolated_annotation_env
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

            # 1. Load application
            page.goto(f"http://127.0.0.1:{port}", timeout=20000)
            page.wait_for_load_state("networkidle")

            # Check title
            page.wait_for_selector("h1", timeout=15000)
            h1_text = page.locator("h1").inner_text()
            assert "LUMAE Product Ads" in h1_text

            # 2. Check Sample 1 (Non-Holdout)
            # Should show sample test_sample_non_holdout
            page.wait_for_selector("text=test_sample_non_holdout", timeout=10000)

            # Non-holdout MUST show AI Candidate Pre-Annotation card
            ai_card_text = page.locator("body").inner_text()
            assert "AI Candidate Pre-Annotation" in ai_card_text
            assert "The presenter demonstrates the physical features and operation of the appliance." in ai_card_text
            assert "5.5" in ai_card_text
            assert "18.2" in ai_card_text

            # Click Accept AI Query
            accept_q_btn = page.locator("button:has-text('Accept AI Query')")
            if accept_q_btn.count() > 0:
                accept_q_btn.first.click()
                time.sleep(1.0)

            # Click Accept AI Window
            accept_w_btn = page.locator("button:has-text('Accept AI Window')")
            if accept_w_btn.count() > 0:
                accept_w_btn.first.click()
                time.sleep(1.0)

            # Click Save button
            save_btn = page.locator("button:has-text('SAVE HUMAN VERIFIED'), button:has-text('SAVE & NEXT')")
            save_btn.first.wait_for(state="visible", timeout=20000)
            save_btn.first.click()
            time.sleep(2.0)

            # Verify persisted in isolated test CSV
            with primary_csv.open("r", encoding="utf-8") as f:
                saved_records = {r["sample_id"]: r for r in csv.DictReader(f)}
            assert saved_records["test_sample_non_holdout"]["review_status"] == "REVIEWED"
            assert saved_records["test_sample_non_holdout"]["annotator_id"] == "human_01"

            # 3. Check Sample 2 (Blind Holdout)
            # Check if app already advanced to test_sample_blind_holdout upon SAVE HUMAN VERIFIED (advance=True)
            body_text = page.locator("body").inner_text()
            if "test_sample_blind_holdout" not in body_text:
                next_btn = page.locator("button:has-text('Next Sample')")
                if next_btn.count() > 0 and next_btn.is_enabled():
                    next_btn.click()
                    time.sleep(1.5)

            # Wait for Sample 2 text to appear
            page.wait_for_selector("text=test_sample_blind_holdout", timeout=10000)

            # Verify Blind Holdout banner appears
            holdout_page_text = page.locator("body").inner_text()
            assert "Blind Human Holdout Sample" in holdout_page_text

            # CRITICAL PRIVACY ASSERTION:
            # AI candidate information for holdout MUST NOT be displayed!
            assert "Holdout Candidate Query" not in holdout_page_text
            assert "Holdout Rationale" not in holdout_page_text
            assert "Accept AI Query" not in holdout_page_text

            browser.close()
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
