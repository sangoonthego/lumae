"""Script to freeze fresh5 human ground truth before revealing AI predictions.

Stage A.2.5H Protocol:
1. Extract the 5 fresh blind samples from human_primary.csv.
2. Read original queries from semantic_v3_fresh5_inputs.json.
3. Write local_data/annotations/lumae_ads/blind_eval/semantic_v3_fresh5_human_gt_frozen.csv.
4. Calculate SHA256 of human GT file and human_primary.csv.
5. Create local_data/manifests/semantic_v3_fresh5_human_gt_freeze.json.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

FRESH5_SAMPLE_IDS = [
    "lumae_ads_pilot_0017",
    "lumae_ads_pilot_0018",
    "lumae_ads_pilot_0020",
    "lumae_ads_pilot_0037",
    "lumae_ads_pilot_0043",
]


def freeze_fresh5_human_gt() -> dict:
    hp_path = REPO_ROOT / "local_data" / "annotations" / "lumae_ads" / "human_primary.csv"
    assert hp_path.is_file(), f"Missing human_primary.csv at {hp_path}"
    hp_bytes = hp_path.read_bytes()
    source_hp_sha256 = hashlib.sha256(hp_bytes).hexdigest()

    df_hp = pd.read_csv(hp_path)

    inputs_path = REPO_ROOT / "local_data" / "annotations" / "lumae_ads" / "blind_eval" / "semantic_v3_fresh5_inputs.json"
    assert inputs_path.is_file(), f"Missing inputs at {inputs_path}"
    with open(inputs_path, "r", encoding="utf-8") as f:
        inputs = json.load(f)
    orig_queries = {item["sample_id"]: item["query"] for item in inputs}

    sub = df_hp[df_hp["sample_id"].isin(FRESH5_SAMPLE_IDS)].copy()
    assert len(sub) == 5, f"Expected 5 samples, found {len(sub)}"

    # Sort strictly by FRESH5_SAMPLE_IDS
    sub["sort_order"] = sub["sample_id"].apply(lambda x: FRESH5_SAMPLE_IDS.index(x))
    sub = sub.sort_values("sort_order").drop(columns=["sort_order"])

    rows = []
    for _, r in sub.iterrows():
        sid = str(r["sample_id"])
        start_sec = float(r["gt_start_seconds"])
        end_sec = float(r["gt_end_seconds"])
        assert 0.0 <= start_sec < end_sec <= float(r["duration_seconds"]), f"Invalid boundaries for {sid}"
        assert str(r["review_status"]) == "REVIEWED", f"{sid} review_status is not REVIEWED"
        assert str(r["annotator_id"]).startswith("human_"), f"{sid} annotator_id must start with human_"

        rows.append({
            "sample_id": sid,
            "video_filename": str(r["video_filename"]),
            "original_query": orig_queries.get(sid, str(r["query"])),
            "final_query": str(r["query"]),
            "human_start_seconds": f"{start_sec:.1f}",
            "human_end_seconds": f"{end_sec:.1f}",
            "annotator_id": str(r["annotator_id"]),
            "review_status": str(r["review_status"]),
            "annotation_notes": str(r["annotation_notes"]),
        })

    out_dir = REPO_ROOT / "local_data" / "annotations" / "lumae_ads" / "blind_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    frozen_csv = out_dir / "semantic_v3_fresh5_human_gt_frozen.csv"

    fieldnames = [
        "sample_id",
        "video_filename",
        "original_query",
        "final_query",
        "human_start_seconds",
        "human_end_seconds",
        "annotator_id",
        "review_status",
        "annotation_notes",
    ]

    with open(frozen_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    frozen_bytes = frozen_csv.read_bytes()
    human_gt_sha256 = hashlib.sha256(frozen_bytes).hexdigest()

    manifest = {
        "experiment": "semantic_v3_fresh5_blind_validation",
        "sample_ids": FRESH5_SAMPLE_IDS,
        "human_gt_file": "local_data/annotations/lumae_ads/blind_eval/semantic_v3_fresh5_human_gt_frozen.csv",
        "human_gt_sha256": human_gt_sha256,
        "source_human_primary_sha256": source_hp_sha256,
        "freeze_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "HUMAN_GT_FROZEN_BEFORE_AI_REVEAL",
    }

    manifest_path = REPO_ROOT / "local_data" / "manifests" / "semantic_v3_fresh5_human_gt_freeze.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest


if __name__ == "__main__":
    m = freeze_fresh5_human_gt()
    print("FRESH5 HUMAN GT FROZEN SUCCESSFULLY")
    print(json.dumps(m, indent=2))
