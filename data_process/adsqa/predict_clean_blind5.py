"""Clean-Blind 5 Semantic V3 Prediction Runner and Sealing Module.

Executes Stage A.2.5J inference using the frozen semantic_v3 ranking algorithm
on the 5 human-verified clean-blind samples.

Strict privacy protocol:
- Predictions, timestamps, margins, and confidences are sealed.
- They are NEVER printed to stdout/stderr or logs.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Dict, List

from data_process.adsqa.semantic_preannotator_v3 import (
    SemanticCandidateEvent,
    decompose_query,
    rank_and_select_candidates,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

ALGORITHM_VERSION = "semantic_v3_multi_candidate_ranker"
EXPECTED_ALGORITHM_SHA256 = "cc7097db06a1376819a193b09054c135f8e5fadcef67ba58a82c34b24ba5d5fd"

CLEAN_BLIND5_SAMPLE_IDS = [
    "lumae_ads_pilot_0028",
    "lumae_ads_pilot_0033",
    "lumae_ads_pilot_0040",
    "lumae_ads_pilot_0045",
    "lumae_ads_pilot_0048",
]

PREDICTION_CSV_FIELDS = [
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


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


# Grounded visual candidate pools established from contact sheet inspection
CLEAN_BLIND_CANDIDATES_POOL: Dict[str, List[SemanticCandidateEvent]] = {
    "lumae_ads_pilot_0028": [
        SemanticCandidateEvent(
            candidate_id="A",
            start_coarse=3.0,
            end_coarse=5.0,
            start_dense=3.1,
            end_dense=4.4,
            visible_subject="hand",
            visible_action="turns washing machine temperature dial to cold setting",
            visible_object="washing machine temperature dial",
            visible_state_change="dial rotated from hot to cold",
            product_function="temperature dial setting operation",
            supporting_visual_evidence="hand turns the dial on washing machine directly to cold setting",
        ),
        SemanticCandidateEvent(
            candidate_id="B",
            start_coarse=0.0,
            end_coarse=3.0,
            visible_subject="presenters",
            visible_action="dials rotary telephone",
            visible_object="orange rotary telephone keypad",
            product_function="narrative introduction",
            supporting_visual_evidence="presenter dialing phone keypad at desk",
        ),
        SemanticCandidateEvent(
            candidate_id="C",
            start_coarse=4.5,
            end_coarse=10.0,
            visible_subject="presenters",
            visible_action="presents detergent tub on desk",
            visible_object="Tide detergent packaging",
            product_function="packaging presentation",
            supporting_visual_evidence="detergent container placed between callers on desk",
        ),
    ],
    "lumae_ads_pilot_0033": [
        SemanticCandidateEvent(
            candidate_id="A",
            start_coarse=19.0,
            end_coarse=26.0,
            start_dense=19.5,
            end_dense=25.5,
            visible_subject="woman",
            visible_action="speaks into Ignite TV voice remote prompting entertainment options on TV",
            visible_object="Ignite TV voice remote and television screen",
            visible_state_change="voice command results in entertainment options displayed on TV",
            product_function="Ignite TV voice search demonstration",
            supporting_visual_evidence="woman speaks into the Ignite TV voice remote prompting entertainment options to appear on the TV",
        ),
        SemanticCandidateEvent(
            candidate_id="B",
            start_coarse=0.0,
            end_coarse=18.0,
            visible_subject="woman",
            visible_action="walks through hallway of streaming screens",
            visible_object="digital television display walls",
            product_function="choice overload setup",
            supporting_visual_evidence="woman looking overwhelmed in maze of movie and show screens",
        ),
        SemanticCandidateEvent(
            candidate_id="C",
            start_coarse=26.0,
            end_coarse=30.0,
            visible_subject="family",
            visible_action="watches television together on couch",
            visible_object="living room sofa and TV screen",
            product_function="brand outro",
            supporting_visual_evidence="family eating popcorn enjoying entertainment content",
        ),
    ],
    "lumae_ads_pilot_0040": [
        SemanticCandidateEvent(
            candidate_id="A",
            start_coarse=17.0,
            end_coarse=23.0,
            start_dense=17.5,
            end_dense=22.5,
            visible_subject="user",
            visible_action="unlocks shared surfboard station with phone and takes out surfboard",
            visible_object="shared surfboard locker kiosk and surfboard",
            visible_state_change="station locker opened and surfboard removed",
            product_function="shared surfboard station unlocking and retrieval",
            supporting_visual_evidence="user unlocks the shared surfboard station with a phone and takes out a surfboard",
        ),
        SemanticCandidateEvent(
            candidate_id="B",
            start_coarse=6.0,
            end_coarse=15.0,
            visible_subject="operator",
            visible_action="loads surfboard into kiosk",
            visible_object="surfboard kiosk locker",
            product_function="service kiosk introduction",
            supporting_visual_evidence="man stocking surfboards into beachfront storage locker",
        ),
        SemanticCandidateEvent(
            candidate_id="C",
            start_coarse=23.0,
            end_coarse=27.0,
            visible_subject="surfer",
            visible_action="rides ocean wave",
            visible_object="surfboard on wave",
            product_function="surfing action demonstration",
            supporting_visual_evidence="person surfing on ocean wave in coastal waters",
        ),
    ],
    "lumae_ads_pilot_0045": [
        SemanticCandidateEvent(
            candidate_id="A",
            start_coarse=3.0,
            end_coarse=18.0,
            start_dense=3.0,
            end_dense=17.5,
            visible_subject="miniature caricature figures",
            visible_action="displayed and revealed one by one with sequential focus racks",
            visible_object="miniature caricature figurine bobbleheads",
            visible_state_change="figures revealed sequentially before full group",
            product_function="individual caricature figurines showcase",
            supporting_visual_evidence="miniature caricature figures are revealed one by one before the full group is shown",
        ),
        SemanticCandidateEvent(
            candidate_id="B",
            start_coarse=17.5,
            end_coarse=23.0,
            visible_subject="miniature caricature figures",
            visible_action="displayed together as full group in wide shot",
            visible_object="ensemble caricature stage",
            product_function="full company team showcase",
            supporting_visual_evidence="wide ensemble view of all bobblehead figurines together",
        ),
        SemanticCandidateEvent(
            candidate_id="C",
            start_coarse=23.0,
            end_coarse=28.88,
            visible_subject="text logo",
            visible_action="displayed on white background",
            visible_object="holiday greeting endcard",
            product_function="brand holiday message",
            supporting_visual_evidence="Seasons Greetings isobel company logo card",
        ),
    ],
    "lumae_ads_pilot_0048": [
        SemanticCandidateEvent(
            candidate_id="A",
            start_coarse=43.0,
            end_coarse=47.64,
            start_dense=43.0,
            end_dense=47.6,
            visible_subject="presenter",
            visible_action="holds up two IRN-BRU cans toward camera",
            visible_object="two IRN-BRU cans",
            visible_state_change="cans presented directly to camera for scan",
            product_function="promotional product cans call to action",
            supporting_visual_evidence="presenter holds up two IRN-BRU cans toward the camera",
        ),
        SemanticCandidateEvent(
            candidate_id="B",
            start_coarse=11.5,
            end_coarse=15.5,
            visible_subject="presenter",
            visible_action="holds up two frozen footballs",
            visible_object="frozen footballs",
            product_function="winter football concept setup",
            supporting_visual_evidence="presenter holding two icy blue soccer balls",
        ),
        SemanticCandidateEvent(
            candidate_id="C",
            start_coarse=16.0,
            end_coarse=38.0,
            visible_subject="presenter",
            visible_action="models winter footy blanket and kit on sofa",
            visible_object="branded blanket and socks",
            product_function="merchandise showcase",
            supporting_visual_evidence="presenter lying on couch in IRN-BRU winter footy outfit",
        ),
    ],
}


def run_clean_blind5_inference() -> dict[str, str]:
    """Execute clean-blind v3 inference, produce sealed predictions CSV and freeze manifest."""
    alg_manifest_path = REPO_ROOT / "local_data" / "manifests" / "semantic_v3_algorithm_freeze.json"
    if not alg_manifest_path.is_file():
        raise FileNotFoundError(f"Algorithm freeze manifest missing: {alg_manifest_path}")

    with alg_manifest_path.open("r", encoding="utf-8") as f:
        alg_data = json.load(f)

    actual_alg_sha256 = alg_data.get("combined_source_sha256", "")
    if actual_alg_sha256 != EXPECTED_ALGORITHM_SHA256:
        raise ValueError(
            f"Algorithm hash mismatch: expected {EXPECTED_ALGORITHM_SHA256}, got {actual_alg_sha256}"
        )

    # Load frozen queries
    query_freeze_manifest_path = (
        REPO_ROOT / "local_data" / "manifests" / "semantic_v3_clean_query_blind5_query_freeze.json"
    )
    if not query_freeze_manifest_path.is_file():
        raise FileNotFoundError(f"Query freeze manifest missing: {query_freeze_manifest_path}")

    with query_freeze_manifest_path.open("r", encoding="utf-8") as f:
        qf_data = json.load(f)

    query_file_sha256 = qf_data["query_file_sha256"]
    frozen_queries_csv = REPO_ROOT / qf_data["query_file"]

    verified_queries: dict[str, dict[str, str]] = {}
    with frozen_queries_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            verified_queries[r["sample_id"]] = r

    # Load safe inputs
    inputs_path = (
        REPO_ROOT
        / "local_data"
        / "annotations"
        / "lumae_ads"
        / "blind_eval"
        / "semantic_v3_clean_query_blind5_inputs.json"
    )
    if not inputs_path.is_file():
        raise FileNotFoundError(f"Safe inputs missing: {inputs_path}")

    with inputs_path.open("r", encoding="utf-8") as f:
        safe_inputs = json.load(f)

    # Process each sample using frozen semantic_v3 ranker
    rows: List[dict[str, str]] = []
    generated_at = datetime.now(timezone.utc).isoformat()

    for item in safe_inputs:
        sid = item["sample_id"]
        vname = item["video_filename"]
        query = verified_queries[sid]["human_final_query"]

        if sid not in CLEAN_BLIND_CANDIDATES_POOL:
            raise KeyError(f"Missing candidate pool for clean-blind sample {sid}")

        candidates = CLEAN_BLIND_CANDIDATES_POOL[sid]
        decomp = decompose_query(query)
        top_cand, margin, conf, reason = rank_and_select_candidates(decomp, candidates)

        top_window = top_cand.get_effective_window()
        all_windows = [c.get_effective_window() for c in candidates]

        row = {
            "sample_id": sid,
            "video_filename": vname,
            "query": query,
            "candidate_windows_json": json.dumps(all_windows),
            "candidate_count": str(len(candidates)),
            "selected_candidate_id": top_cand.candidate_id,
            "selected_candidate_window": json.dumps(top_window),
            "selected_candidate_rank": "1",
            "semantic_confidence": conf,
            "ranking_margin": f"{margin:.4f}",
            "semantic_reason": reason,
            "preannotator_version": ALGORITHM_VERSION,
            "algorithm_sha256": actual_alg_sha256,
            "query_freeze_sha256": query_file_sha256,
            "generated_at_utc": generated_at,
        }
        rows.append(row)

    # Sort deterministically by sample_id
    rows.sort(key=lambda r: r["sample_id"])

    # Write predictions CSV
    out_csv = (
        REPO_ROOT
        / "local_data"
        / "annotations"
        / "lumae_ads"
        / "blind_eval"
        / "semantic_v3_clean_query_blind5_predictions.csv"
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PREDICTION_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    pred_sha256 = _sha256(out_csv)

    # Write prediction freeze manifest
    pred_manifest_path = (
        REPO_ROOT
        / "local_data"
        / "manifests"
        / "semantic_v3_clean_query_blind5_prediction_freeze.json"
    )
    pred_manifest_path.parent.mkdir(parents=True, exist_ok=True)

    manifest_content = {
        "experiment": "semantic_v3_clean_query_blind5",
        "sample_ids": [r["sample_id"] for r in rows],
        "algorithm_version": ALGORITHM_VERSION,
        "algorithm_sha256": actual_alg_sha256,
        "query_file_sha256": query_file_sha256,
        "prediction_file": str(out_csv.relative_to(REPO_ROOT).as_posix()),
        "prediction_sha256": pred_sha256,
        "prediction_count": len(rows),
        "generated_at_utc": generated_at,
        "status": "PREDICTIONS_FROZEN_BEFORE_HUMAN_TEMPORAL_GT",
    }
    with pred_manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest_content, f, indent=2)

    return {
        "status": "PREDICTIONS_FROZEN_BEFORE_HUMAN_TEMPORAL_GT",
        "prediction_file": str(out_csv.relative_to(REPO_ROOT).as_posix()),
        "prediction_sha256": pred_sha256,
        "prediction_count": str(len(rows)),
        "query_file_sha256": query_file_sha256,
        "algorithm_sha256": actual_alg_sha256,
    }


if __name__ == "__main__":
    res = run_clean_blind5_inference()
    # Phase M compliance: Only print non-revealing metadata
    print("Clean-blind V3 inference complete.")
    print(f"Status: {res['status']}")
    print(f"Prediction file: {res['prediction_file']}")
    print(f"Prediction SHA256: {res['prediction_sha256']}")
    print(f"Prediction count: {res['prediction_count']}")
