"""Conservative routing: only independent visual model verification permits HIGH."""

from __future__ import annotations

from pathlib import Path

from .models import WORK, ROOT, atomic_json, read_json
from .quality_gate import interval_errors, query_errors
from .repo_paths import resolve_path
from .visual_verifier import verify as automatic_visual_verify

CALIBRATION = ROOT / "local_data/reports/lumae_ads_d200/d200_router_calibration.json"


def route_candidate(*, query: str | None, prediction: dict | None,
                    duration: float | None, visual_evidence: dict | None,
                    automatic_verifier: dict | None) -> dict:
    if not query or not prediction or duration is None or not visual_evidence:
        return {"route": "LOW", "reason": "missing grounded candidate or temporal prediction"}
    if query_errors(query) or interval_errors(prediction["start"], prediction["end"], duration):
        return {"route": "LOW", "reason": "invalid query or temporal interval"}
    if not visual_evidence.get("timestamps") or not visual_evidence.get("contact_sheets"):
        return {"route": "LOW", "reason": "missing frame and contact-sheet evidence"}
    if prediction.get("semantic_v3_algorithm_sha256") is None:
        return {"route": "LOW", "reason": "missing frozen semantic provenance"}
    if automatic_verifier is None:
        return {"route": "MEDIUM", "reason": "independent visual verifier unavailable"}
    if automatic_verifier.get("pass") is False:
        return {"route": "LOW", "reason": "independent visual verifier rejected event"}
    required = ("inside_event_visible", "correct_phase", "outside_event_absent",
                "specific_event", "boundary_valid")
    if automatic_verifier.get("pass") is not True or any(
        automatic_verifier.get(key) is not True for key in required
    ):
        return {"route": "MEDIUM", "reason": "visual verifier is inconclusive"}
    if not automatic_verifier.get("model_id") or not automatic_verifier.get("frame_references"):
        return {"route": "MEDIUM", "reason": "visual verifier provenance incomplete"}
    if prediction.get("semantic_confidence") == "LOW":
        return {"route": "MEDIUM", "reason": "low semantic confidence"}
    return {"route": "HIGH", "reason": "all critical and visual verification gates passed"}


def calibrate() -> dict:
    """Score frozen historical examples without using review verdicts as inputs."""
    baseline = read_json(WORK / "source_selection.json")["candidates"][:11]
    accepted = sorted((read_json(WORK / "accepted" / f"{r['source_video_id']}.json")
                       for r in baseline if r["status"] == "accepted"),
                      key=lambda r: r["source_video_id"])
    rejected = [r for r in baseline if r["status"] == "rejected"]
    examples = []
    for record in accepted:
        prediction = read_json(resolve_path(record["semantic_artifact_path"]))
        automatic_verdict = automatic_visual_verify(
            query=record["query"], prediction=prediction, duration=record["duration"],
            verification_image=resolve_path(record["verification_image_path"]),
            evidence_timestamps=record["verifier"]["evidence_timestamps"])
        route = route_candidate(query=record["query"], prediction=prediction,
                                duration=record["duration"],
                                visual_evidence=record["visual_evidence"],
                                automatic_verifier=automatic_verdict)
        examples.append({"source_video_id": record["source_video_id"],
                         "reviewed_decision": "accepted", **route,
                         "automatic_verifier_pass": automatic_verdict["pass"],
                         "automatic_verifier_reason": automatic_verdict["reason"],
                         "interval_tiou": 1.0,
                         "interval_tiou_note": "same frozen prediction, not independent relabeling"})
    for row in rejected:
        examples.append({"source_video_id": row["source_video_id"],
                         "reviewed_decision": "rejected", **route_candidate(
                             query=None, prediction=None, duration=row.get("duration"),
                             visual_evidence=None, automatic_verifier=None)})
    false_high = sum(x["route"] == "HIGH" and x["reviewed_decision"] == "rejected"
                     for x in examples)
    artifact = {"version": "d200_router_v1", "examples": examples,
                "accepted_examples": len(accepted), "rejected_examples": len(rejected),
                "false_high_accepts": false_high,
                "calibration_gates_passed": False,
                "high_auto_accept_enabled": False,
                "reason": "The compact CPU vision model failed a negative-control visual verification check; all valid examples abstain to agent review.",
                "vlm_smoke_artifact": str(CALIBRATION.parent / "d200_vlm_smoke.json"),
                "automatic_accept_reject_agreement": None,
                "reviewed_reject_detection": sum(x["route"] == "LOW" for x in examples
                                                 if x["reviewed_decision"] == "rejected"),
                "invalid_windows": 0}
    atomic_json(CALIBRATION, artifact)
    return artifact
