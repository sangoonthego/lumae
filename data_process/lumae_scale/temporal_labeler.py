"""Frozen semantic_v3 ranking plus separately versioned CLIP boundary wrapper."""

from __future__ import annotations

import numpy as np
from contextlib import nullcontext

from data_process.adsqa.semantic_preannotator_v3 import (
    SemanticCandidateEvent, decompose_query, rank_and_select_candidates,
)
from data_process.annotation.assisted_deployment import ALGORITHM_SHA256, verify_frozen_ranker

from .models import ROOT, VISUAL_CACHE
from .quality_gate import interval_errors

WRAPPER_VERSION = "d200_clip_boundary_v3_evidence_anchored"


def localize(video_id: str, query: str, events: list[dict], duration: float,
             expected_event_index: int, timer=None, clip_runtime=None) -> dict:
    verify_frozen_ranker(ROOT)
    if not events:
        raise ValueError("No visually inspected candidates")
    candidates = [SemanticCandidateEvent(
        candidate_id=chr(65 + i), start_coarse=float(e["coarse_start"]),
        end_coarse=float(e["coarse_end"]), visible_subject=e["visible_subject"],
        visible_action=e["visible_action"], visible_object=e["visible_object"],
        visible_state_change=e.get("visible_state_change", ""),
        supporting_visual_evidence=e.get("evidence_note", ""),
    ) for i, e in enumerate(events)]
    with timer.stage("semantic") if timer else nullcontext():
        selected, margin, confidence, reason = rank_and_select_candidates(
            decompose_query(query), candidates)
    if selected.candidate_id != chr(65 + expected_event_index):
        raise ValueError("Frozen ranker selected a different visual event phase")
    # Boundary refinement is query-conditioned CLIP similarity on real frame features.
    # It is a separate wrapper; the frozen ranker source remains untouched.
    with timer.stage("refinement") if timer else nullcontext():
        return _refine(video_id, query, candidates, selected, margin, confidence,
                       reason, duration, events[expected_event_index],
                       clip_runtime=clip_runtime)


def _refine(video_id, query, candidates, selected, margin, confidence, reason,
            duration, selected_event, clip_runtime=None):
    from .clip_runtime import get_clip_runtime
    from .visual_index import clip_consistency

    array = np.load(VISUAL_CACHE / video_id / "clip_features.npz")
    features, times = array["features"], array["timestamps"]
    runtime = clip_runtime or get_clip_runtime("ViT-B/32", device="cpu")
    vector = runtime.encode_text(query)[0]
    similarity = features @ vector
    coarse_start, coarse_end = selected.get_effective_window()
    allowed = (times >= coarse_start) & (times <= coarse_end)
    if not allowed.any():
        raise ValueError("No CLIP frame in selected visual phase")
    best = int(np.argmax(np.where(allowed, similarity, -999.0)))
    # Expand from the peak while neighboring similarities remain close to its
    # local score. Coarse visual bounds constrain search, but are never copied
    # straight into the final label.
    threshold = float(similarity[best] - 0.03)
    left = right = best
    while left > 0 and allowed[left - 1] and similarity[left - 1] >= threshold:
        left -= 1
    while right + 1 < len(times) and allowed[right + 1] and similarity[right + 1] >= threshold:
        right += 1
    start = max(0.0, round(float(times[left] - 1.0), 1))
    end = min(duration, round(float(times[right] + 1.0), 1))
    if end - start < 4.0:
        center = float(times[best])
        start = max(0.0, round(center - 2.0, 1))
        end = min(duration, round(center + 2.0, 1))
    # The visual evidence timestamps identify frames where the action was
    # actually seen. Keep those frames inside the result with one second of
    # sampling context; this repairs short CLIP peaks without expanding every
    # interval to the entire coarse phase.
    evidence_times = sorted(float(t) for t in selected_event["evidence_timestamps"])
    start = max(start, max(0.0, evidence_times[0] - 1.0))
    end = max(end, min(duration, evidence_times[-1] + 1.0))
    end = min(end, max(float(selected_event["coarse_end"]),
                       min(duration, evidence_times[-1] + 1.0)))
    if interval_errors(start, end, duration):
        raise ValueError("CLIP refined interval failed validity gate")
    return {"query": query, "start": start, "end": end,
            "semantic_v3_algorithm_sha256": ALGORITHM_SHA256,
            "boundary_wrapper_version": WRAPPER_VERSION,
            "semantic_v3_selected_candidate_id": selected.candidate_id,
            "semantic_v3_candidate_windows": [c.get_effective_window() for c in candidates],
            "semantic_confidence": confidence, "ranking_score": selected.query_match_score,
            "ranking_margin": margin, "semantic_reason": reason,
            "clip_peak_timestamp": float(times[best]),
            "refinement_evidence_timestamps": evidence_times,
            "clip_signals": clip_consistency(video_id, query, start, end,
                                             clip_runtime=runtime, query_vector=vector)}
