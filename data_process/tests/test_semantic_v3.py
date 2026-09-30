"""Tests for STAGE A.2.5G: Semantic V3 Multi-Candidate Ranker, Development Validation, and Fresh5 Blind Prediction Freeze.

Verifies:
1. Candidate enumeration & structured representation.
2. Query decomposition (subject, action, object, context).
3. Action & object specificity scoring rules.
4. Multi-phase video behavior and ranking margin computation.
5. Short-event duration preservation.
6. 8-sample development benchmark regression & sample 0010 failure-mode resolution.
7. Development gate passing (mean tIoU >= 0.70, semantic correct >= 7/8, 0010 != WRONG_EVENT).
8. Algorithm freeze manifest integrity and source SHA256 hashes.
9. Fresh5 safe inputs manifest and GT denylist enforcement.
10. Fresh5 sealed predictions CSV schema and prediction freeze manifest.
11. Prediction immutability & unrevealed predictions.
12. Streamlit DOM blindness for fresh5 samples using Playwright.
13. Source data immutability (human_primary.csv, semantic_v2 artifacts, blind_holdout.json).
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

from data_process.adsqa.semantic_preannotator_v3 import (
    DEVELOPMENT_CANDIDATES_POOL,
    FRESH_BLIND_CANDIDATES_POOL,
    SemanticCandidateEvent,
    decompose_query,
    predict_semantic_v3,
    rank_and_select_candidates,
    score_candidate_match,
)
from data_process.adsqa.dev_benchmark_v3 import (
    DEV_SAMPLE_IDS as BENCHMARK_DEV_SAMPLE_IDS,
    run_development_benchmark,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

FRESH5_SAMPLE_IDS = [
    "lumae_ads_pilot_0017",
    "lumae_ads_pilot_0018",
    "lumae_ads_pilot_0020",
    "lumae_ads_pilot_0037",
    "lumae_ads_pilot_0043",
]

DEV_SAMPLE_IDS = [
    "lumae_ads_pilot_0004",
    "lumae_ads_pilot_0006",
    "lumae_ads_pilot_0007",
    "lumae_ads_pilot_0008",
    "lumae_ads_pilot_0010",
    "lumae_ads_pilot_0011",
    "lumae_ads_pilot_0012",
    "lumae_ads_pilot_0013",
]

BASELINE_HUMAN_PRIMARY_SHA256 = "4ac2aa9ec0de00795cf335d785db513f79608ca097b3e5d3bfae4610924f13b3"
STAGE_A25H_HUMAN_PRIMARY_SHA256 = "29b5924831d32953e6399fcf408023bbfb02efc4b082976d98a63cf82ec8e1ae"
BASELINE_BLIND_HOLDOUT_SHA256 = "ec551df79eac67b5984190d6efab730c173e3c394cbeea1761632ab247a52387"
BASELINE_V2_CLEAN3_PRED_SHA256 = "bfec7d5c91e699ffb52315bcb1a97e9e03b4473304f5644f92203d19190bf8b2"
BASELINE_V2_PREANNOTATIONS_SHA256 = "e3ead17dd58c988c700e80e0800caaa4996bd081112946ed41ad51a03eebf8a1"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def test_candidate_enumeration_and_structure():
    """Verify all development and fresh blind samples have structured candidates with required fields."""
    for sid in DEV_SAMPLE_IDS:
        assert sid in DEVELOPMENT_CANDIDATES_POOL, f"{sid} missing in development pool"
        cands = DEVELOPMENT_CANDIDATES_POOL[sid]
        assert len(cands) >= 2, f"{sid} should have >= 2 enumerated candidates for ranking"
        for c in cands:
            assert isinstance(c, SemanticCandidateEvent)
            assert c.candidate_id in {"A", "B", "C", "D"}
            assert c.start_coarse <= c.end_coarse
            assert len(c.visible_subject) > 0
            assert len(c.visible_action) > 0
            assert len(c.visible_object) > 0
            assert len(c.product_function) > 0
            assert len(c.supporting_visual_evidence) > 0

    for sid in FRESH5_SAMPLE_IDS:
        assert sid in FRESH_BLIND_CANDIDATES_POOL, f"{sid} missing in fresh blind pool"
        cands = FRESH_BLIND_CANDIDATES_POOL[sid]
        assert len(cands) >= 2, f"{sid} should have >= 2 enumerated candidates"
        for c in cands:
            assert isinstance(c, SemanticCandidateEvent)
            assert c.start_coarse <= c.end_coarse


def test_query_decomposition():
    """Verify structured parsing into subject, action, object, context components."""
    q_0010 = "The Unfear system detects and identifies surrounding sounds while the user wears earbuds."
    decomp = decompose_query(q_0010)
    assert "unfear" in decomp.subject.lower() or "system" in decomp.subject.lower()
    assert "detect" in decomp.action.lower() or "identif" in decomp.action.lower()
    assert "sound" in decomp.object.lower()
    assert "earbud" in decomp.context.lower() or "wear" in decomp.context.lower()

    q_generic = "The demonstrator shows how the product functions."
    decomp_gen = decompose_query(q_generic)
    assert len(decomp_gen.action) > 0
    assert len(decomp_gen.object) > 0


def test_action_and_object_specificity_rules():
    """Verify candidate scoring prioritizes matching action and object over generic product activity."""
    decomp = decompose_query("The Unfear system detects and identifies surrounding sounds while the user wears earbuds.")
    
    cand_sound_detect = SemanticCandidateEvent(
        candidate_id="B",
        start_coarse=39.1,
        end_coarse=47.6,
        visible_subject="earbuds system HUD",
        visible_action="detects and identifies surrounding sounds with visual audio cards",
        visible_object="surrounding sounds (dog bark, siren, crowd noise)",
        visible_state_change="real-time sound identification card pops up",
        product_function="sound identification feature in action",
        supporting_visual_evidence="display graphic shows sound detection label",
    )
    
    cand_earbud_insert = SemanticCandidateEvent(
        candidate_id="A",
        start_coarse=27.0,
        end_coarse=30.5,
        visible_subject="young man",
        visible_action="opens charging case and inserts earbuds into ears",
        visible_object="Samsung Galaxy Buds case and earbuds",
        visible_state_change="earbuds inserted into ears",
        product_function="wearing/fit demonstration",
        supporting_visual_evidence="close up of earbud insertion",
    )
    
    score_b = score_candidate_match(decomp, cand_sound_detect)
    score_a = score_candidate_match(decomp, cand_earbud_insert)
    assert score_b > score_a + 0.3, f"Action and object specificity should strongly prefer sound detection (got {score_b} vs {score_a})"


def test_ranking_margin_and_uncertainty():
    """Verify candidate ranking computes ranking margin and sets appropriate confidence."""
    decomp = decompose_query("The creator presents the key physical features of the product.")
    cands = [
        SemanticCandidateEvent(
            candidate_id="A",
            start_coarse=10.0,
            end_coarse=15.0,
            visible_subject="creator",
            visible_action="presents physical features of product",
            visible_object="product exterior",
            product_function="feature display",
            supporting_visual_evidence="close up physical details",
        ),
        SemanticCandidateEvent(
            candidate_id="B",
            start_coarse=0.0,
            end_coarse=5.0,
            visible_subject="creator",
            visible_action="smiling at camera",
            visible_object="studio",
            product_function="intro",
            supporting_visual_evidence="talking intro",
        ),
    ]
    top, margin, conf, reason = rank_and_select_candidates(decomp, cands)
    assert top.candidate_id == "A"
    assert margin > 0.0
    assert conf in {"HIGH", "MEDIUM", "LOW"}


def test_short_event_preservation():
    """Verify dense refinement preserves short event durations without excessive padding."""
    cand = SemanticCandidateEvent(
        candidate_id="B",
        start_coarse=10.0,
        end_coarse=20.0,
        start_dense=12.2,
        end_dense=13.5,
        visible_subject="user",
        visible_action="clicks button",
        visible_object="device",
        product_function="button click",
        supporting_visual_evidence="quick button press",
    )
    window = cand.get_effective_window()
    duration = window[1] - window[0]
    assert 1.0 <= duration <= 2.0, f"Expected short event window of ~1.3s, got {duration}s"
    assert window == [12.2, 13.5]


def test_development_benchmark_gate_and_0010_resolution():
    """Verify development benchmark achieves >= 0.70 mean tIoU, >= 7/8 semantic correct, and fixes 0010."""
    report = run_development_benchmark()
    agg = report["aggregate"]
    gate = report["quality_gate"]

    assert gate["gate_result"] == "PASS_DEVELOPMENT_GATE"
    assert agg["v3_mean_tiou"] >= 0.70
    assert agg["v3_semantic_correct"] >= 7

    # 0010 resolution verification: V2 failed (tIoU 0.0, WRONG_EVENT), V3 must be CORRECT_EVENT
    s0010 = agg["sample_0010_result"]
    assert s0010["v2_semantic"] == "WRONG_EVENT"
    assert s0010["v3_semantic"] in {"CORRECT_EVENT", "PARTIAL_EVENT"}
    assert s0010["v3_tiou"] > 0.5


def test_semantic_v3_algorithm_freeze_manifest():
    """Verify semantic_v3 algorithm freeze manifest exists, matches current hashes, and has required status."""
    freeze_path = REPO_ROOT / "local_data" / "manifests" / "semantic_v3_algorithm_freeze.json"
    assert freeze_path.is_file(), f"Missing algorithm freeze: {freeze_path}"

    with freeze_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["version"] == "semantic_v3_multi_candidate_ranker"
    assert data["status"] == "SEMANTIC_V3_FROZEN_BEFORE_FRESH_BLIND_VALIDATION"
    assert data["development_sample_ids"] == DEV_SAMPLE_IDS

    # Check source file hashes match
    for rel_path, expected_hash in data["source_file_hashes"].items():
        p = REPO_ROOT / rel_path
        assert p.is_file(), f"Source file {p} does not exist"
        assert _sha256(p) == expected_hash, f"Source file {rel_path} was modified after algorithm freeze!"


def test_fresh5_safe_inputs_no_gt_leakage():
    """Verify fresh5 inputs manifest was sourced safely without GT fields."""
    inputs_path = REPO_ROOT / "local_data" / "annotations" / "lumae_ads" / "blind_eval" / "semantic_v3_fresh5_inputs.json"
    assert inputs_path.is_file(), f"Missing inputs: {inputs_path}"

    with inputs_path.open("r", encoding="utf-8") as f:
        samples = json.load(f)

    assert len(samples) == 5
    sample_ids = [s["sample_id"] for s in samples]
    assert sample_ids == FRESH5_SAMPLE_IDS

    prohibited_keys = {
        "gt_start_seconds", "gt_end_seconds", "annotator_id", "review_status",
        "annotation_notes", "human_notes", "adjudicated", "multi_windows"
    }
    for s in samples:
        keys = set(s.keys())
        assert not keys.intersection(prohibited_keys), f"GT leakage in {s['sample_id']}: {keys.intersection(prohibited_keys)}"
        assert s["duration_seconds"] > 0
        assert len(s["query"]) > 5


def test_fresh5_prediction_freeze_and_immutability():
    """Verify fresh5 predictions CSV is sealed, correctly hashed, and freeze manifest is valid."""
    pred_path = REPO_ROOT / "local_data" / "annotations" / "lumae_ads" / "blind_eval" / "semantic_v3_fresh5_predictions.csv"
    assert pred_path.is_file()

    freeze_path = REPO_ROOT / "local_data" / "manifests" / "semantic_v3_fresh5_prediction_freeze.json"
    assert freeze_path.is_file()

    with freeze_path.open("r", encoding="utf-8") as f:
        freeze_data = json.load(f)

    assert freeze_data["status"] == "PREDICTIONS_FROZEN_BEFORE_HUMAN_GT"
    assert freeze_data["prediction_count"] == 5
    assert freeze_data["sample_ids"] == FRESH5_SAMPLE_IDS

    actual_sha = _sha256(pred_path)
    assert actual_sha == freeze_data["prediction_sha256"], "Prediction SHA256 does not match freeze manifest!"

    # Verify CSV schema
    with pred_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == 5
    expected_fields = [
        "sample_id", "video_filename", "query", "candidate_windows_json",
        "selected_candidate_window", "candidate_count", "semantic_confidence",
        "ranking_margin", "semantic_reason", "preannotator_version", "generated_at_utc"
    ]
    assert list(rows[0].keys()) == expected_fields


def test_source_data_immutability():
    """Verify human_primary.csv, blind_holdout.json, and semantic_v2 artifacts are completely unchanged."""
    human_primary_path = REPO_ROOT / "local_data" / "annotations" / "lumae_ads" / "human_primary.csv"
    assert human_primary_path.is_file()
    a25k_freeze = json.loads((REPO_ROOT / "local_data/manifests/semantic_v3_clean_query_blind5_human_gt_freeze.json").read_text())
    assert _sha256(human_primary_path) == a25k_freeze["source_human_primary_sha256"], "human_primary.csv differs from A.2.5K human freeze!"

    holdout_path = REPO_ROOT / "local_data" / "annotations" / "lumae_ads" / "blind_holdout.json"
    assert holdout_path.is_file()
    assert _sha256(holdout_path) == BASELINE_BLIND_HOLDOUT_SHA256, "blind_holdout.json was mutated!"

    v2_clean3_pred = REPO_ROOT / "local_data" / "annotations" / "lumae_ads" / "blind_eval" / "semantic_v2_clean3_predictions.csv"
    if v2_clean3_pred.is_file():
        assert _sha256(v2_clean3_pred) == BASELINE_V2_CLEAN3_PRED_SHA256, "semantic_v2 clean3 predictions mutated!"

    v2_preann = REPO_ROOT / "local_data" / "annotations" / "lumae_ads" / "semantic_preannotations_v2.csv"
    if v2_preann.is_file():
        assert _sha256(v2_preann) == BASELINE_V2_PREANNOTATIONS_SHA256, "semantic_preannotations_v2.csv mutated!"


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
def isolated_fresh5_env(tmp_path):
    """Fixture providing isolated environment for Playwright UI validation of fresh5 blindness."""
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
            "sample_id": "lumae_ads_pilot_0017",
            "video_filename": "2e3fd9fcc7064cbea237f4da9b8583fb.mp4",
            "vid": "vid_0017",
            "source_dataset": "AdsQA",
            "source_video_id": "2e3fd9fcc7064cbea237f4da9b8583fb",
            "source_split": "test",
            "source_url": "",
            "product_category": "Product Demonstration",
            "query": "The creator presents the key physical features of the product.",
            "duration_seconds": "40.01",
            "gt_start_seconds": "",
            "gt_end_seconds": "",
            "annotator_id": "ann_lead",
            "annotation_notes": "",
            "review_status": "DRAFT",
        },
        {
            "sample_id": "lumae_ads_pilot_0018",
            "video_filename": "355d38abd561eed3e330bf6b9a64d1cc.mp4",
            "vid": "vid_0018",
            "source_dataset": "AdsQA",
            "source_video_id": "355d38abd561eed3e330bf6b9a64d1cc",
            "source_split": "test",
            "source_url": "",
            "product_category": "Product Demonstration",
            "query": "The presenter demonstrates assembling the product components.",
            "duration_seconds": "24.75",
            "gt_start_seconds": "",
            "gt_end_seconds": "",
            "annotator_id": "ann_lead",
            "annotation_notes": "",
            "review_status": "DRAFT",
        },
        {
            "sample_id": "lumae_ads_pilot_0020",
            "video_filename": "3a06ef4e81885cfbedd14d5f15a025f3.mp4",
            "vid": "vid_0020",
            "source_dataset": "AdsQA",
            "source_video_id": "3a06ef4e81885cfbedd14d5f15a025f3",
            "source_split": "test",
            "source_url": "",
            "product_category": "Product Demonstration",
            "query": "The creator presents the key physical features of the product.",
            "duration_seconds": "31.05",
            "gt_start_seconds": "",
            "gt_end_seconds": "",
            "annotator_id": "ann_lead",
            "annotation_notes": "",
            "review_status": "DRAFT",
        },
        {
            "sample_id": "lumae_ads_pilot_0037",
            "video_filename": "974e5395d04e9f39df700af341a647f4.mp4",
            "vid": "vid_0037",
            "source_dataset": "AdsQA",
            "source_video_id": "974e5395d04e9f39df700af341a647f4",
            "source_split": "test",
            "source_url": "",
            "product_category": "Electronics & Gadgets",
            "query": "The creator shows how the handheld device is powered on and used.",
            "duration_seconds": "40.00",
            "gt_start_seconds": "",
            "gt_end_seconds": "",
            "annotator_id": "ann_lead",
            "annotation_notes": "",
            "review_status": "DRAFT",
        },
        {
            "sample_id": "lumae_ads_pilot_0043",
            "video_filename": "b55612a7531872c6ce921bb78958fd78.mp4",
            "vid": "vid_0043",
            "source_dataset": "AdsQA",
            "source_video_id": "b55612a7531872c6ce921bb78958fd78",
            "source_split": "test",
            "source_url": "",
            "product_category": "Electronics & Gadgets",
            "query": "The user connects the electronic accessory to the main unit.",
            "duration_seconds": "51.04",
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


def test_playwright_fresh5_dom_blindness(isolated_fresh5_env):
    """Playwright verification that fresh 5 samples (0017, 0018, 0020, 0037, 0043) strictly enforce BLIND_PRIMARY in DOM."""
    env_info = isolated_fresh5_env
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

            for target_sid in FRESH5_SAMPLE_IDS:
                page.wait_for_selector(f"text={target_sid}", timeout=10000)

                body_content = page.locator("body").inner_text()
                dom_html = page.content()

                # 1. Video playback container present
                page.wait_for_selector("video, [data-testid='stVideo']", timeout=10000)

                # 2. Query is visible
                assert "Original Query:" in body_content

                # 3. Blind banner present
                assert "Blind Human Holdout Sample" in body_content or "BLIND_PRIMARY" in body_content

                # 4. Strict privacy: AI candidate card absent from body and DOM
                assert "AI Candidate Pre-Annotation" not in body_content
                assert "AI Candidate Pre-Annotation" not in dom_html

                # 5. Semantic V3 prediction fields absent from DOM
                assert "candidate_windows" not in dom_html
                assert "ranking_margin" not in dom_html
                assert "semantic_reason" not in dom_html
                assert "semantic_confidence" not in dom_html
                assert "selected_candidate" not in dom_html

                # 6. Start and End inputs are blank (empty)
                start_input = page.locator(f"input[id*='num_start_{target_sid}']")
                end_input = page.locator(f"input[id*='num_end_{target_sid}']")
                if start_input.count() > 0:
                    assert start_input.input_value() == ""
                if end_input.count() > 0:
                    assert end_input.input_value() == ""

                # 7. Save button present
                save_btn = page.locator("button:has-text('Save Human Review')")
                assert save_btn.count() > 0

                # Move to next sample if not at the end
                if target_sid != FRESH5_SAMPLE_IDS[-1]:
                    next_btn = page.locator("button:has-text('Next Sample')")
                    assert next_btn.count() > 0
                    next_btn.first.click()
                    time.sleep(1.0)

            browser.close()

            # Confirm no automated human annotation took place
            with primary_csv.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    assert r["review_status"] == "DRAFT"
                    assert r["gt_start_seconds"] == ""
                    assert r["gt_end_seconds"] == ""
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
