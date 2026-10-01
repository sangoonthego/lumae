"""Automatic evidence checks with explicit abstention when visual semantics are unknown."""

from __future__ import annotations

from pathlib import Path

from .quality_gate import interval_errors, query_errors


def verify(*, query: str, prediction: dict, duration: float,
           verification_image: Path, evidence_timestamps: dict) -> dict:
    valid_image = verification_image.is_file() and verification_image.stat().st_size > 0
    valid_frame_roles = all(evidence_timestamps.get(k) for k in ("before", "inside", "after"))
    structural_errors = query_errors(query) + interval_errors(
        prediction["start"], prediction["end"], duration)
    if not valid_image:
        structural_errors.append("missing before/inside/after image")
    if not valid_frame_roles:
        structural_errors.append("incomplete before/inside/after evidence")
    if structural_errors:
        return {"pass": False, "reason": "; ".join(structural_errors),
                "frame_references": [str(verification_image)] if valid_image else [],
                "evidence_timestamps": evidence_timestamps,
                "inside_event_visible": None, "correct_phase": None,
                "outside_event_absent": None, "specific_event": None,
                "boundary_valid": False}
    return {"pass": None, "reason": "Visual semantics abstained: the tested compact CPU VLM failed a negative control.",
            "frame_references": [str(verification_image)],
            "evidence_timestamps": evidence_timestamps,
            "inside_event_visible": None, "correct_phase": None,
            "outside_event_absent": None, "specific_event": None,
            "boundary_valid": True,
            "clip_signals": prediction.get("clip_signals"),
            "clip_warning": "CLIP similarity is an auxiliary signal, not visual verification or a calibrated probability."}
