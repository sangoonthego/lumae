"""Stable 10% HIGH review sample, independent of source processing order."""

from __future__ import annotations

import hashlib
import math

from .models import ROOT, SEED, WORK, atomic_json, read_json

REPORT = ROOT / "local_data/reports/lumae_ads_d200/d200_high_random_audit.json"


def select_audit_ids(high_video_ids: list[str], seed: int = SEED) -> list[str]:
    unique = set(high_video_ids)
    if len(unique) != len(high_video_ids):
        raise ValueError("Duplicate HIGH video IDs")
    return sorted(unique, key=lambda video_id: hashlib.sha256(
        f"{seed}:{video_id}".encode()).hexdigest())[:math.ceil(len(unique) * 0.1)]


def write_audit_report() -> dict:
    high = []
    for path in sorted((WORK / "accepted").glob("*.json")):
        record = read_json(path)
        if record.get("router", {}).get("route") == "HIGH":
            high.append(record["source_video_id"])
    selected = select_audit_ids(high)
    reviews = {}
    for video_id in selected:
        path = WORK / "visual_index" / video_id / "high_audit.json"
        if path.exists():
            reviews[video_id] = read_json(path)
    agreement = [x.get("correct") is True for x in reviews.values()]
    report = {"seed": SEED, "high_count": len(high),
              "selected_for_audit": selected, "selected_count": len(selected),
              "reviewed_count": len(reviews), "correct_count": sum(agreement),
              "disagreement_count": sum(not x for x in agreement),
              "accuracy": sum(agreement) / len(agreement) if agreement else None,
              "auto_accept_paused": len(reviews) < len(selected) or any(not x for x in agreement)}
    atomic_json(REPORT, report)
    return report
