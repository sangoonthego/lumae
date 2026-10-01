"""Read-only accounting for the first 20 post-baseline source candidates."""

from __future__ import annotations

from .models import ROOT, WORK, atomic_json, read_json

REPORT = ROOT / "local_data/reports/lumae_ads_d200/d200_shadow_review.json"


def write_shadow_report() -> dict:
    rows = read_json(WORK / "source_selection.json")["candidates"][11:31]
    cases = []
    for row in rows:
        video_id = row["source_video_id"]
        route_file = WORK / "visual_index" / video_id / "router.json"
        route = read_json(route_file) if route_file.exists() else None
        verdict = ("accepted" if row["status"] == "accepted" else
                   "rejected" if row["status"] == "rejected" else None)
        if route is not None or verdict is not None:
            cases.append({"source_video_id": video_id,
                          "automatic_route": route["route"] if route else None,
                          "automatic_reason": route["reason"] if route else None,
                          "review_verdict": verdict,
                          "shadow_complete": route is not None and verdict is not None})
    completed = [c for c in cases if c["shadow_complete"]]
    high = [c for c in completed if c["automatic_route"] == "HIGH"]
    report = {"target_candidate_count": 20, "cases": cases,
              "completed_with_auto_route_and_review": len(completed),
              "high_cases": len(high),
              "high_precision": (sum(c["review_verdict"] == "accepted" for c in high) / len(high)
                                 if high else None),
              "broad_auto_accept_enabled": False,
              "note": "MEDIUM is an abstention requiring agent review; it is not counted as an automatic pass/reject agreement."}
    atomic_json(REPORT, report)
    return report
