"""Product-ad candidate selection and review catalog generator for AdsQA."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
from typing import Any

# Preferred candidate category keyword patterns
CATEGORY_RULES: list[tuple[str, list[str], str]] = [
    (
        "Cleaning & Household Supply",
        [
            r"\bvacuum\b",
            r"\bclean\b",
            r"\bcleaning\b",
            r"\bmop\b",
            r"\bsweep\b",
            r"\blaundry\b",
            r"\bdetergent\b",
            r"\bdisinfect\b",
            r"\bstain\b",
            r"\bsponge\b",
            r"\bwipe\b",
            r"\bsoap\b",
            r"\bbleach\b",
            r"\btrash\b",
        ],
        "ACCEPT",
    ),
    (
        "Home & Kitchen Appliance",
        [
            r"\bblender\b",
            r"\bfridge\b",
            r"\brefrigerator\b",
            r"\boven\b",
            r"\bcooker\b",
            r"\bmicrowave\b",
            r"\bcoffee\b",
            r"\bdishwasher\b",
            r"\btoaster\b",
            r"\bkettle\b",
            r"\biron\b",
            r"\bsteamer\b",
            r"\bappliance\b",
            r"\bkitchen\b",
            r"\bpot\b",
            r"\bpan\b",
        ],
        "ACCEPT",
    ),
    (
        "Electronics & Gadgets",
        [
            r"\bphone\b",
            r"\bsmartphone\b",
            r"\bscreen\b",
            r"\btablet\b",
            r"\blaptop\b",
            r"\bspeaker\b",
            r"\bheadphone\b",
            r"\bearphone\b",
            r"\bcamera\b",
            r"\bcharger\b",
            r"\bbattery\b",
            r"\bdisplay\b",
            r"\bmonitor\b",
            r"\baudio\b",
            r"\bdrone\b",
            r"\belectronic\b",
            r"\bgadget\b",
        ],
        "ACCEPT",
    ),
    (
        "Tools & Hardware",
        [
            r"\btool\b",
            r"\bhardware\b",
            r"\bdrill\b",
            r"\bscrew\b",
            r"\bhammer\b",
            r"\bwrench\b",
            r"\bsaw\b",
            r"\btape\b",
            r"\bpaint\b",
            r"\bscrewfix\b",
        ],
        "ACCEPT",
    ),
    (
        "Personal Care & Grooming",
        [
            r"\bshaver\b",
            r"\brazor\b",
            r"\bshave\b",
            r"\btoothbrush\b",
            r"\bbrush\b",
            r"\bhairdryer\b",
            r"\bstraightener\b",
            r"\bskincare\b",
            r"\blotion\b",
            r"\bcream\b",
            r"\bserum\b",
        ],
        "ACCEPT",
    ),
    (
        "Product Demonstration",
        [
            r"\bdemonstrat\w*\b",
            r"\bdemo\b",
            r"\bhow it works\b",
            r"\bhow to use\b",
            r"\bfeatures?\b",
            r"\bassembl\w*\b",
            r"\boperate\b",
            r"\bfunctions?\b",
            r"\bunboxing\b",
        ],
        "ACCEPT",
    ),
]

REJECT_RULES: list[tuple[str, list[str]]] = [
    (
        "Awareness & Non-Product Campaign",
        [
            r"\bfundraising\b",
            r"\bcharity\b",
            r"\bngo\b",
            r"\bbank\b",
            r"\binsurance\b",
            r"\bpeta\b",
            r"\bpolitical\b",
            r"\belection\b",
            r"\bgovernment\b",
            r"\bwildlife\b",
            r"\bclimate change\b",
            r"\bhuman rights\b",
        ],
    ),
]


def classify_text(text: str) -> tuple[str, str, str]:
    """Classify video text into category, status, and reason."""
    lower_text = text.lower()

    # Check for direct product matches
    for cat_name, patterns, status in CATEGORY_RULES:
        for pat in patterns:
            if re.search(pat, lower_text):
                match_word = re.findall(pat, lower_text)[0]
                reason = f"Matches {cat_name} product demonstration criteria (keyword '{match_word}')"
                return cat_name, status, reason

    # Check for explicit rejection
    for rej_cat, patterns in REJECT_RULES:
        for pat in patterns:
            if re.search(pat, lower_text):
                match_word = re.findall(pat, lower_text)[0]
                return rej_cat, "REJECT", f"Excluded domain: {rej_cat} ('{match_word}')"

    # Fallback to general category / pending review
    return "General Commercial", "PENDING", "Ambiguous category; requires human visual verification"


def select_candidates(
    raw_dir: Path | str,
    output_csv: Path | str | None = None,
) -> list[dict[str, Any]]:
    """Generate candidate review list from AdsQA metadata."""
    raw = Path(raw_dir)
    manifest_path = raw / "source_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Source manifest not found at: {manifest_path}")

    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    # Collect texts per video ID from train and test metadata
    video_texts: dict[str, list[str]] = {}

    train_path = raw / "train.json"
    if train_path.is_file():
        with train_path.open("r", encoding="utf-8") as f:
            train_data = json.load(f)
        for item in train_data:
            vid = item.get("video")
            if vid:
                meta = item.get("meta_info", "")
                q = item.get("question", "")
                video_texts.setdefault(vid, []).append(f"{meta} {q}")

    test_path = raw / "testset_question.json"
    if test_path.is_file():
        with test_path.open("r", encoding="utf-8") as f:
            test_data = json.load(f)
        for item in test_data:
            vid = item.get("video")
            if vid:
                q = item.get("question", "")
                video_texts.setdefault(vid, []).append(q)

    videos = manifest.get("videos", [])
    records: list[dict[str, Any]] = []

    for v in videos:
        vid_id = v["source_video_id"]
        split = v.get("source_split", "unassigned")
        url = v.get("source_url", "")
        texts = video_texts.get(vid_id, [])
        combined_text = " ".join(texts)

        category, status, reason = classify_text(combined_text)

        # Store note with snippet of metadata or question
        snippet = ""
        if texts:
            # First 120 chars of first entry
            cleaned = " ".join(texts[0].split())
            snippet = cleaned[:120]

        record = {
            "source_video_id": vid_id,
            "source_split": split,
            "source_url": url,
            "category": category,
            "candidate_status": status,
            "candidate_reason": reason,
            "duration_seconds": "",
            "download_status": "PENDING",
            "notes": snippet,
        }
        records.append(record)

    if output_csv:
        out_path = Path(output_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "source_video_id",
            "source_split",
            "source_url",
            "category",
            "candidate_status",
            "candidate_reason",
            "duration_seconds",
            "download_status",
            "notes",
        ]
        with out_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in records:
                writer.writerow(r)
        print(f"Generated candidate review CSV: {out_path} ({len(records):,} records).")

    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Select product-ad candidates from AdsQA metadata.")
    parser.add_argument(
        "--raw-dir",
        default="local_data/raw/adsqa",
        help="Directory containing source_manifest.json and raw annotations",
    )
    parser.add_argument(
        "--output-csv",
        default="local_data/reports/adsqa/candidate_review.csv",
        help="Output CSV path for human review and pipeline filtering",
    )
    args = parser.parse_args()

    records = select_candidates(args.raw_dir, args.output_csv)
    accepted = [r for r in records if r["candidate_status"] == "ACCEPT"]
    rejected = [r for r in records if r["candidate_status"] == "REJECT"]
    pending = [r for r in records if r["candidate_status"] == "PENDING"]
    print(f"Candidate Summary: {len(accepted):,} ACCEPT, {len(pending):,} PENDING, {len(rejected):,} REJECT.")


if __name__ == "__main__":
    main()
