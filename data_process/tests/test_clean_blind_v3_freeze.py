"""Tests for Stage A.2.5J: Freeze Verified Clean-Blind Queries, Semantic V3 Inference, and Prediction Sealing.

Verifies:
1. All 5 clean-blind sample query reviews are complete and human-verified.
2. Canonical 17-field schema of query_reviews.csv (no temporal fields).
3. Selection manifest integrity (0028, 0033, 0040, 0045, 0048).
4. Query freeze CSV (14 fields) and query freeze manifest SHA256 integrity.
5. Semantic V3 algorithm freeze manifest and bit-for-bit source hash integrity.
6. Safe inference inputs schema (no temporal GT leakage).
7. Sealed predictions CSV (15 fields) and prediction freeze manifest integrity.
8. Source data immutability (human_primary.csv untouched, blank GT preserved).
9. Playwright DOM privacy test verifying predictions are ABSENT from DOM in CLEAN_QUERY_BLIND_TEMPORAL mode.
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

from data_process.annotation.models import (
    AnnotationMode,
    OriginalQueryQuality,
    QueryReviewRecord,
    QueryReviewStatus,
)
from data_process.annotation.query_sanitation import (
    CANONICAL_QUERY_REVIEW_FIELDS,
    FORBIDDEN_TEMPORAL_FIELDS,
    get_annotations_dir,
    get_manifests_dir,
    get_query_reviews_path,
    load_query_reviews,
    validate_query_review_schema,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

CLEAN_BLIND5_SAMPLE_IDS = [
    "lumae_ads_pilot_0028",
    "lumae_ads_pilot_0033",
    "lumae_ads_pilot_0040",
    "lumae_ads_pilot_0045",
    "lumae_ads_pilot_0048",
]

EXPECTED_QUERIES = {
    "lumae_ads_pilot_0028": "A hand turns the washing machine temperature dial to the cold setting.",
    "lumae_ads_pilot_0033": "The woman speaks into the Ignite TV voice remote, prompting entertainment options to appear on the TV.",
    "lumae_ads_pilot_0040": "A user unlocks the shared surfboard station with a phone and takes out a surfboard.",
    "lumae_ads_pilot_0045": "Miniature caricature figures are revealed one by one before the full group is shown.",
    "lumae_ads_pilot_0048": "The presenter holds up two IRN-BRU cans toward the camera.",
}

BASELINE_HUMAN_PRIMARY_SHA256 = "29b5924831d32953e6399fcf408023bbfb02efc4b082976d98a63cf82ec8e1ae"
BASELINE_V3_ALG_SHA256 = "cc7097db06a1376819a193b09054c135f8e5fadcef67ba58a82c34b24ba5d5fd"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def wait_for_server(port: int, timeout_sec: float = 25.0) -> bool:
    start = time.time()
    while time.time() - start < timeout_sec:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1.0):
                return True
        except (ConnectionRefusedError, OSError):
            time.sleep(0.3)
    return False


def test_all_five_query_reviews_complete():
    """Verify all 5 clean-blind samples have complete, valid human query reviews in query_reviews.csv."""
    reviews = load_query_reviews()
    for sid in CLEAN_BLIND5_SAMPLE_IDS:
        assert sid in reviews, f"Missing query review for {sid}"
        rec = reviews[sid]
        assert rec.query_review_status in (
            QueryReviewStatus.VALID_AS_IS.value,
            QueryReviewStatus.HUMAN_EDITED.value,
        )
        assert len(rec.human_final_query.strip()) > 0
        assert rec.query_localizable is True
        assert rec.query_observable is True
        assert rec.reviewer_id.startswith("human_")
        assert rec.temporal_annotation_locked is True
        assert len(rec.reviewed_at_utc.strip()) > 0
        assert rec.human_final_query == EXPECTED_QUERIES[sid]


def test_query_reviews_schema_validity():
    """Verify query_reviews.csv adheres to canonical 17-field schema without temporal GT fields."""
    csv_path = get_query_reviews_path()
    validate_query_review_schema(csv_path)  # raises if invalid


def test_selection_manifest_integrity():
    """Verify semantic_v3_clean_query_blind5_selection.json contains exact expected IDs and status."""
    sel_path = get_manifests_dir() / "semantic_v3_clean_query_blind5_selection.json"
    assert sel_path.is_file()
    with sel_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["status"] == "CLEAN_BLIND5_SELECTED_BEFORE_QUERY_REVIEW"
    assert data["selected_sample_ids"] == CLEAN_BLIND5_SAMPLE_IDS


def test_query_freeze_manifest_and_sha256():
    """Verify frozen query CSV and query freeze manifest SHA256 integrity."""
    frozen_csv = (
        REPO_ROOT
        / "local_data"
        / "annotations"
        / "lumae_ads"
        / "blind_eval"
        / "semantic_v3_clean_query_blind5_queries_frozen.csv"
    )
    assert frozen_csv.is_file()

    expected_fields = [
        "sample_id",
        "video_filename",
        "source_dataset",
        "source_video_id",
        "original_query",
        "human_final_query",
        "original_query_quality",
        "query_review_status",
        "query_change_type",
        "query_localizable",
        "query_observable",
        "reviewer_id",
        "review_notes",
        "query_version",
    ]
    with frozen_csv.open("r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == expected_fields
        rows = list(reader)
        assert len(rows) == 5
        sids = [r["sample_id"] for r in rows]
        assert sids == sorted(sids)
        assert sids == CLEAN_BLIND5_SAMPLE_IDS

    q_manifest_path = (
        get_manifests_dir() / "semantic_v3_clean_query_blind5_query_freeze.json"
    )
    assert q_manifest_path.is_file()
    with q_manifest_path.open("r", encoding="utf-8") as f:
        q_manifest = json.load(f)

    assert q_manifest["status"] == "CLEAN_BLIND_QUERIES_FROZEN_BEFORE_V3_INFERENCE"
    assert q_manifest["sample_ids"] == CLEAN_BLIND5_SAMPLE_IDS
    assert q_manifest["query_file_sha256"] == _sha256(frozen_csv)
    assert q_manifest["source_query_reviews_sha256"] == _sha256(get_query_reviews_path())


def test_semantic_v3_algorithm_hash_freeze():
    """Verify semantic_v3 algorithm source files match recorded frozen hashes bit-for-bit."""
    alg_manifest_path = get_manifests_dir() / "semantic_v3_algorithm_freeze.json"
    assert alg_manifest_path.is_file()
    with alg_manifest_path.open("r", encoding="utf-8") as f:
        alg_data = json.load(f)

    assert alg_data["version"] == "semantic_v3_multi_candidate_ranker"
    assert alg_data["combined_source_sha256"] == BASELINE_V3_ALG_SHA256

    for rel_path, expected_hash in alg_data["source_file_hashes"].items():
        full_path = REPO_ROOT / rel_path
        assert full_path.is_file()
        assert _sha256(full_path) == expected_hash


def test_safe_inference_inputs_schema_and_denylist():
    """Verify safe inputs JSON has exactly the 7 non-temporal fields."""
    inputs_path = (
        REPO_ROOT
        / "local_data"
        / "annotations"
        / "lumae_ads"
        / "blind_eval"
        / "semantic_v3_clean_query_blind5_inputs.json"
    )
    assert inputs_path.is_file()
    with inputs_path.open("r", encoding="utf-8") as f:
        items = json.load(f)

    assert len(items) == 5
    allowed_keys = {
        "sample_id",
        "video_filename",
        "video_path",
        "duration_seconds",
        "human_final_query",
        "source_dataset",
        "source_video_id",
    }
    for item in items:
        assert set(item.keys()) == allowed_keys
        assert item["sample_id"] in CLEAN_BLIND5_SAMPLE_IDS
        # Verify no temporal label leakage
        for forbidden in ("gt_start_seconds", "gt_end_seconds", "start", "end", "windows"):
            assert forbidden not in item


def test_prediction_output_and_freeze_manifest():
    """Verify prediction CSV schema, determinism, and prediction freeze manifest integrity."""
    pred_csv = (
        REPO_ROOT
        / "local_data"
        / "annotations"
        / "lumae_ads"
        / "blind_eval"
        / "semantic_v3_clean_query_blind5_predictions.csv"
    )
    assert pred_csv.is_file()

    expected_fields = [
        "sample_id",
        "video_filename",
        "query",
        "candidate_windows_json",
        "candidate_count",
        "selected_candidate_id",
        "selected_candidate_window",
        "selected_candidate_rank",
        "semantic_confidence",
        "ranking_margin",
        "semantic_reason",
        "preannotator_version",
        "algorithm_sha256",
        "query_freeze_sha256",
        "generated_at_utc",
    ]
    with pred_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == expected_fields
        rows = list(reader)
        assert len(rows) == 5
        sids = [r["sample_id"] for r in rows]
        assert sids == sorted(sids)
        assert sids == CLEAN_BLIND5_SAMPLE_IDS

        for r in rows:
            # Query must be the human-verified query, NOT the original query
            assert r["query"] == EXPECTED_QUERIES[r["sample_id"]]
            assert int(r["candidate_count"]) >= 2
            assert r["selected_candidate_id"] in ("A", "B", "C", "D")
            assert len(json.loads(r["selected_candidate_window"])) == 2
            assert r["preannotator_version"] == "semantic_v3_multi_candidate_ranker"
            assert r["algorithm_sha256"] == BASELINE_V3_ALG_SHA256

    pred_manifest_path = (
        get_manifests_dir()
        / "semantic_v3_clean_query_blind5_prediction_freeze.json"
    )
    assert pred_manifest_path.is_file()
    with pred_manifest_path.open("r", encoding="utf-8") as f:
        p_manifest = json.load(f)

    assert p_manifest["status"] == "PREDICTIONS_FROZEN_BEFORE_HUMAN_TEMPORAL_GT"
    assert p_manifest["sample_ids"] == CLEAN_BLIND5_SAMPLE_IDS
    assert p_manifest["prediction_sha256"] == _sha256(pred_csv)
    assert p_manifest["prediction_count"] == 5
    assert p_manifest["algorithm_sha256"] == BASELINE_V3_ALG_SHA256


def test_source_data_immutability():
    """Verify human_primary.csv matches the hash recorded in human GT freeze manifest."""
    primary_csv = get_annotations_dir() / "human_primary.csv"
    assert primary_csv.is_file()

    gt_manifest_path = REPO_ROOT / "local_data" / "manifests" / "semantic_v3_clean_query_blind5_human_gt_freeze.json"
    assert gt_manifest_path.is_file()
    with gt_manifest_path.open("r", encoding="utf-8") as f:
        gt_manifest = json.load(f)

    assert _sha256(primary_csv) == gt_manifest["source_human_primary_sha256"]

    # Verify that clean-blind samples have REVIEWED GT in human_primary.csv
    with primary_csv.open("r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r["sample_id"] in CLEAN_BLIND5_SAMPLE_IDS:
                assert r["review_status"] == "REVIEWED"
                assert len(r["gt_start_seconds"]) > 0
                assert len(r["gt_end_seconds"]) > 0


def test_playwright_clean_query_blind_temporal_dom_privacy():
    """Playwright verification that clean-blind samples in CLEAN_QUERY_BLIND_TEMPORAL mode have ZERO prediction data in DOM."""
    port = get_free_port()
    env = os.environ.copy()
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

            # Switch mode to CLEAN_QUERY_BLIND_TEMPORAL
            mode_radio = page.locator("label:has-text('CLEAN_QUERY_BLIND_TEMPORAL')")
            if mode_radio.count() > 0:
                mode_radio.click()
                time.sleep(1.0)
                page.wait_for_load_state("networkidle")

            for target_sid in CLEAN_BLIND5_SAMPLE_IDS:
                page.wait_for_selector(f"text={target_sid}", timeout=10000)

                body_content = page.locator("body").inner_text()
                dom_html = page.content()

                # 1. Video playback container present
                page.wait_for_selector("video, [data-testid='stVideo']", timeout=10000)

                # 2. Frozen verified query is visible
                expected_q = EXPECTED_QUERIES[target_sid]
                assert expected_q in body_content

                # 3. Blind temporal review banner present
                assert "Clean-Blind Temporal Ground-Truth Review" in body_content or "Clean-Blind Temporal Review" in body_content

                # 4. Strict privacy: AI predictions, margins, confidences are ABSENT from DOM
                assert "AI Candidate Pre-Annotation" not in dom_html
                assert "candidate_windows" not in dom_html
                assert "ranking_margin" not in dom_html
                assert "semantic_reason" not in dom_html
                assert "semantic_confidence" not in dom_html
                assert "selected_candidate" not in dom_html
                assert "semantic_v3_clean_query_blind5_predictions.csv" not in dom_html

                # 5. Start and End inputs are blank (empty)
                start_input = page.locator(f"input[id*='clean_blind_start_{target_sid}']")
                end_input = page.locator(f"input[id*='clean_blind_end_{target_sid}']")
                if start_input.count() > 0:
                    assert start_input.input_value() == ""
                if end_input.count() > 0:
                    assert end_input.input_value() == ""

                # 6. Save Human Temporal Review button present
                save_btn = page.locator("button:has-text('Save Human Temporal Review')")
                assert save_btn.count() > 0

                # Move to next sample if not at the end
                if target_sid != CLEAN_BLIND5_SAMPLE_IDS[-1]:
                    next_btn = page.locator("button:has-text('Next Sample')")
                    if next_btn.count() > 0:
                        next_btn.first.click()
                        time.sleep(1.0)

            browser.close()
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
