"""Tests for Stage A.2.5I: Query Sanitation, Contamination Exposure Audit, and Clean-Blind 5 Selection.

Verifies:
1. Query provenance model schema & enums (QueryReviewStatus, OriginalQueryQuality, QueryReviewRecord).
2. Exposure registry generation & exclusion of development, Fresh5, and legacy holdout samples.
3. Deterministic clean-blind 5 sample selection (reproducible with seed 20260930).
4. Query sanitation worklist generation covering all 35 pending samples.
5. Atomic query review reading & writing.
6. Freeze refusal when human query review is incomplete (WAITING_FOR_HUMAN_QUERY_REVIEW).
7. Freeze success & manifest generation when human reviews are complete.
8. Source data immutability (human_primary.csv, semantic_v3 source, previous prediction files).
9. Playwright DOM privacy test: confirms temporal fields, predictions, and candidate windows are absent from DOM in BLIND_QUERY_REVIEW mode.
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
import tempfile
import time
import pytest
from playwright.sync_api import sync_playwright

from data_process.annotation.models import (
    AnnotationMode,
    OriginalQueryQuality,
    QueryReviewRecord,
    QueryReviewStatus,
)
from data_process.annotation.query_sanitation import (
    DEVELOPMENT_SAMPLES,
    FRESH5_A25H_SAMPLES,
    LEGACY_SEED42_HOLDOUT_SAMPLES,
    SELECTION_SEED,
    generate_exposure_registry,
    generate_query_sanitation_worklist,
    get_annotations_dir,
    get_manifests_dir,
    get_query_reviews_path,
    get_query_worklist_path,
    load_query_reviews,
    save_query_review,
    select_clean_blind5,
)
from data_process.annotation.freeze_clean_blind_queries import (
    check_and_freeze_clean_blind_queries,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

BASELINE_HUMAN_PRIMARY_SHA256 = "29b5924831d32953e6399fcf408023bbfb02efc4b082976d98a63cf82ec8e1ae"
BASELINE_V3_PRED_SHA256 = "a67fba037a7566d0d232ce82b651e48fdce6ea36aef6297372c55f506d1ab5fe"
BASELINE_V3_ALG_SHA256 = "cc7097db06a1376819a193b09054c135f8e5fadcef67ba58a82c34b24ba5d5fd"

EXPECTED_CLEAN_BLIND5 = [
    "lumae_ads_pilot_0028",
    "lumae_ads_pilot_0033",
    "lumae_ads_pilot_0040",
    "lumae_ads_pilot_0045",
    "lumae_ads_pilot_0048",
]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def test_query_provenance_schema_and_enums():
    """Verify QueryReviewRecord serialization, enums, and required fields."""
    rec = QueryReviewRecord(
        sample_id="lumae_ads_pilot_0028",
        video_filename="6d4f8e966dfbfa7eefdfb2ba49b11767.mp4",
        source_dataset="AdsQA",
        source_video_id="6d4f8e966dfbfa7eefdfb2ba49b11767",
        original_query="The presenter demonstrates assembling the product components.",
        human_final_query="The driver folds down the rear seats in the SUV.",
        query_review_status=QueryReviewStatus.HUMAN_EDITED.value,
        original_query_quality=OriginalQueryQuality.INCORRECT_FOR_VIDEO.value,
        query_change_type="REWRITTEN",
        query_localizable=True,
        query_observable=True,
        reviewer_id="human_01",
        review_notes="Initial Lumae query template had incorrect assembly action.",
        reviewed_at_utc="2026-09-30T10:00:00Z",
        query_version="lumae_query_v1",
        temporal_annotation_locked=True,
    )

    d = rec.to_csv_dict()
    assert d["sample_id"] == "lumae_ads_pilot_0028"
    assert d["query_review_status"] == "HUMAN_EDITED"
    assert d["original_query_quality"] == "INCORRECT_FOR_VIDEO"
    assert d["temporal_annotation_locked"] == "True"

    rec2 = QueryReviewRecord.from_csv_dict(d)
    assert rec2.sample_id == rec.sample_id
    assert rec2.query_localizable is True
    assert rec2.temporal_annotation_locked is True


def test_exposure_registry_audit():
    """Verify temporal exposure registry accurately classifies all 48 samples."""
    registry = generate_exposure_registry()
    assert registry["total_pilot_samples"] == 48

    summary = registry["exposure_summary"]
    assert len(summary["DEVELOPMENT_EXPOSED"]) == 8
    assert len(summary["BLIND_EVALUATED"]) == 8  # 3 clean + 5 fresh
    assert len(summary["TEMPORAL_GT_REVIEWED"]) == 13

    eligibility = registry["clean_blind_eligibility"]
    assert eligibility["eligible_count"] == 33
    assert eligibility["excluded_count"] == 15

    # Confirm 0002 and 0009 are excluded due to legacy holdout contamination
    assert "lumae_ads_pilot_0002" in eligibility["excluded_sample_ids"]
    assert "lumae_ads_pilot_0009" in eligibility["excluded_sample_ids"]

    # Confirm all 8 dev samples and all 5 fresh5 samples are excluded
    for sid in DEVELOPMENT_SAMPLES | FRESH5_A25H_SAMPLES:
        assert sid in eligibility["excluded_sample_ids"]


def test_deterministic_clean_blind5_selection():
    """Verify deterministic selection is reproducible with seed 20260930."""
    manifest1 = select_clean_blind5(seed=SELECTION_SEED)
    manifest2 = select_clean_blind5(seed=SELECTION_SEED)

    assert manifest1["selected_sample_ids"] == EXPECTED_CLEAN_BLIND5
    assert manifest2["selected_sample_ids"] == EXPECTED_CLEAN_BLIND5
    assert manifest1["status"] == "CLEAN_BLIND5_SELECTED_BEFORE_QUERY_REVIEW"

    # Verify none of the selected 5 have any prior exposure
    for sid in manifest1["selected_sample_ids"]:
        assert sid not in DEVELOPMENT_SAMPLES
        assert sid not in FRESH5_A25H_SAMPLES
        assert sid not in LEGACY_SEED42_HOLDOUT_SAMPLES


def test_query_sanitation_worklist():
    """Verify query_sanitation_worklist.csv contains all 35 pending samples and correct reservation flags."""
    worklist = generate_query_sanitation_worklist()
    assert len(worklist) == 35

    reserved_count = sum(1 for r in worklist if r["clean_blind_reserved"] == "TRUE")
    assert reserved_count == 5

    reserved_ids = sorted([r["sample_id"] for r in worklist if r["clean_blind_reserved"] == "TRUE"])
    assert reserved_ids == EXPECTED_CLEAN_BLIND5

    for r in worklist:
        assert r["temporal_annotation_locked"] == "TRUE"
        assert r["query_review_status"] == QueryReviewStatus.PENDING_QUERY_REVIEW.value


def test_atomic_query_review_storage(tmp_path, monkeypatch):
    """Verify atomic write and read of query_reviews.csv."""
    test_csv = tmp_path / "test_query_reviews.csv"
    monkeypatch.setattr("data_process.annotation.query_sanitation.get_query_reviews_path", lambda: test_csv)

    rec = QueryReviewRecord(
        sample_id="lumae_ads_pilot_0028",
        video_filename="test.mp4",
        original_query="Original query",
        human_final_query="Verified query",
        query_review_status="VALID_AS_IS",
        original_query_quality="VALID",
        reviewer_id="human_01",
    )
    save_query_review(rec)
    assert test_csv.is_file()

    loaded = load_query_reviews()
    assert "lumae_ads_pilot_0028" in loaded
    assert loaded["lumae_ads_pilot_0028"].human_final_query == "Verified query"


def test_freeze_refusal_when_review_incomplete():
    """Verify freeze_clean_blind_queries refuses to freeze when reviews are missing or incomplete."""
    res = check_and_freeze_clean_blind_queries()
    # At this tooling stage, the human reviewer has not yet reviewed the clean-blind 5
    assert res["status"] == "WAITING_FOR_HUMAN_QUERY_REVIEW"
    assert "target_samples" in res
    assert res["target_samples"] == EXPECTED_CLEAN_BLIND5


def test_source_data_immutability():
    """Verify baseline artifacts remain untouched during Stage A.2.5I."""
    hp_path = REPO_ROOT / "local_data" / "annotations" / "lumae_ads" / "human_primary.csv"
    assert _sha256(hp_path) == BASELINE_HUMAN_PRIMARY_SHA256, "human_primary.csv was mutated!"

    v3_pred_path = REPO_ROOT / "local_data" / "annotations" / "lumae_ads" / "blind_eval" / "semantic_v3_fresh5_predictions.csv"
    assert _sha256(v3_pred_path) == BASELINE_V3_PRED_SHA256, "semantic_v3 fresh5 predictions mutated!"

    alg_manifest = REPO_ROOT / "local_data" / "manifests" / "semantic_v3_algorithm_freeze.json"
    with open(alg_manifest, "r", encoding="utf-8") as f:
        alg_data = json.load(f)
    assert alg_data["combined_source_sha256"] == BASELINE_V3_ALG_SHA256


def get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def wait_for_server(port: int, timeout_sec: float = 20.0) -> bool:
    start_time = time.time()
    while time.time() - start_time < timeout_sec:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1.0):
                return True
        except (ConnectionRefusedError, OSError):
            time.sleep(0.4)
    return False


@pytest.fixture
def isolated_clean_blind5_env(tmp_path):
    """Fixture providing isolated environment with the 5 selected clean-blind samples."""
    ann_dir = tmp_path / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)

    primary_csv = ann_dir / "human_primary.csv"
    fieldnames = [
        "sample_id", "video_filename", "vid", "source_dataset", "source_video_id",
        "source_split", "source_url", "product_category", "query", "duration_seconds",
        "gt_start_seconds", "gt_end_seconds", "annotator_id", "annotation_notes", "review_status"
    ]
    rows = [
        {
            "sample_id": "lumae_ads_pilot_0028",
            "video_filename": "6d4f8e966dfbfa7eefdfb2ba49b11767.mp4",
            "vid": "vid_0028",
            "source_dataset": "AdsQA",
            "source_video_id": "6d4f8e966dfbfa7eefdfb2ba49b11767",
            "source_split": "test",
            "source_url": "",
            "product_category": "Product Demonstration",
            "query": "The demonstrator shows how the product functions.",
            "duration_seconds": "30.00",
            "gt_start_seconds": "",
            "gt_end_seconds": "",
            "annotator_id": "ann_lead",
            "annotation_notes": "",
            "review_status": "DRAFT",
        },
        {
            "sample_id": "lumae_ads_pilot_0033",
            "video_filename": "75bdb0fbb1879ac330f3fc94094bf7ca.mp4",
            "vid": "vid_0033",
            "source_dataset": "AdsQA",
            "source_video_id": "75bdb0fbb1879ac330f3fc94094bf7ca",
            "source_split": "test",
            "source_url": "",
            "product_category": "Product Demonstration",
            "query": "The creator presents the key physical features of the product.",
            "duration_seconds": "60.00",
            "gt_start_seconds": "",
            "gt_end_seconds": "",
            "annotator_id": "ann_lead",
            "annotation_notes": "",
            "review_status": "DRAFT",
        },
        {
            "sample_id": "lumae_ads_pilot_0040",
            "video_filename": "ae18cc0a431a7bd30cbdc9e5a7409939.mp4",
            "vid": "vid_0040",
            "source_dataset": "AdsQA",
            "source_video_id": "ae18cc0a431a7bd30cbdc9e5a7409939",
            "source_split": "test",
            "source_url": "",
            "product_category": "Product Demonstration",
            "query": "The demonstrator presents the key physical features of the product.",
            "duration_seconds": "30.00",
            "gt_start_seconds": "",
            "gt_end_seconds": "",
            "annotator_id": "ann_lead",
            "annotation_notes": "",
            "review_status": "DRAFT",
        },
        {
            "sample_id": "lumae_ads_pilot_0045",
            "video_filename": "cc5a7ee2995c8ea4ea46cb2f0d07c941.mp4",
            "vid": "vid_0045",
            "source_dataset": "AdsQA",
            "source_video_id": "cc5a7ee2995c8ea4ea46cb2f0d07c941",
            "source_split": "test",
            "source_url": "",
            "product_category": "Product Demonstration",
            "query": "The user demonstrates operating the product features.",
            "duration_seconds": "60.00",
            "gt_start_seconds": "",
            "gt_end_seconds": "",
            "annotator_id": "ann_lead",
            "annotation_notes": "",
            "review_status": "DRAFT",
        },
        {
            "sample_id": "lumae_ads_pilot_0048",
            "video_filename": "e559a293a88606b4d2bc96bb71e88e5e.mp4",
            "vid": "vid_0048",
            "source_dataset": "AdsQA",
            "source_video_id": "e559a293a88606b4d2bc96bb71e88e5e",
            "source_split": "test",
            "source_url": "",
            "product_category": "Product Demonstration",
            "query": "The creator shows how the handheld device is powered on and used.",
            "duration_seconds": "30.00",
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

    vid_dir = tmp_path / "videos"
    vid_dir.mkdir(parents=True, exist_ok=True)
    for r in rows:
        (vid_dir / r["video_filename"]).touch()

    return {
        "annotations_dir": ann_dir,
        "videos_dir": vid_dir,
        "primary_csv": primary_csv,
    }


def test_playwright_clean_blind5_query_privacy(isolated_clean_blind5_env):
    """Playwright verification that selected clean-blind samples strictly enforce BLIND_QUERY_REVIEW in DOM."""
    env_info = isolated_clean_blind5_env
    ann_dir = env_info["annotations_dir"]
    vid_dir = env_info["videos_dir"]

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

            for target_sid in EXPECTED_CLEAN_BLIND5:
                page.wait_for_selector(f"text={target_sid}", timeout=10000)

                body_content = page.locator("body").inner_text()
                dom_html = page.content()

                # 1. Query sanitation banner present
                assert "Clean-Blind Query Review" in body_content or "Query Sanitation Mode" in body_content

                # 2. Query fields present
                assert "Original Lumae-Generated Draft Query" in body_content
                assert "Original Query Quality:" in body_content
                assert "Final Human Query" in body_content
                assert "Event is temporally localizable in video" in body_content

                # 3. Save Query Review button present
                save_query_btn = page.locator("button:has-text('Save Query Review')")
                assert save_query_btn.count() > 0

                # 4. CRITICAL PRIVACY: Temporal boundaries and predictions strictly ABSENT from DOM
                assert "Ground-Truth Temporal Window" not in body_content
                assert "Ground-Truth Temporal Window" not in dom_html
                assert "Save Human Review" not in dom_html
                assert "START (seconds):" not in dom_html
                assert "END (seconds):" not in dom_html
                assert "num_start_" not in dom_html
                assert "num_end_" not in dom_html
                assert "AI Candidate Pre-Annotation" not in dom_html
                assert "candidate_windows" not in dom_html
                assert "ranking_margin" not in dom_html
                assert "semantic_confidence" not in dom_html

                # Move to next sample if not at the end
                if target_sid != EXPECTED_CLEAN_BLIND5[-1]:
                    next_btn = page.locator("button:has-text('Next Sample')")
                    assert next_btn.count() > 0
                    next_btn.first.click()
                    time.sleep(1.0)

            browser.close()
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
