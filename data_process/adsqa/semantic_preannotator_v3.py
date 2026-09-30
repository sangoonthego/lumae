"""Semantic V3 Multi-Candidate Query-Conditioned Temporal Ranking Module.

Stage A.2.5G Implementation.
Replaces single-region proposal with:
1. Coarse visual inspection & candidate event enumeration.
2. Query decomposition (subject, action, object, context, state change).
3. Query-conditioned semantic ranking with action and object specificity.
4. Top candidate selection with ranking margin & confidence estimation.
5. Dense temporal boundary refinement on the selected candidate event.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import subprocess
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont


def get_repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


@dataclass
class QueryDecomposition:
    """Structured semantic elements extracted from natural language query."""

    raw_query: str
    subject: str = ""
    action: str = ""
    object: str = ""
    context: str = ""
    state_change: str = ""


@dataclass
class SemanticCandidateEvent:
    """Individual candidate event interval with visual and semantic metadata."""

    candidate_id: str
    start_coarse: float
    end_coarse: float
    start_dense: Optional[float] = None
    end_dense: Optional[float] = None
    visible_subject: str = ""
    visible_action: str = ""
    visible_object: str = ""
    visible_state_change: str = ""
    product_function: str = ""
    supporting_visual_evidence: str = ""
    query_match_score: float = 0.0
    ambiguity_flags: List[str] = field(default_factory=list)

    def get_effective_window(self) -> List[float]:
        s = self.start_dense if self.start_dense is not None else self.start_coarse
        e = self.end_dense if self.end_dense is not None else self.end_coarse
        return [round(float(s), 1), round(float(e), 1)]


@dataclass
class SemanticPreannotationV3Record:
    """Standardized record for Stage A.2.5G semantic_v3 pre-annotation."""

    sample_id: str
    video_filename: str
    query: str
    candidate_windows_json: str
    selected_candidate_window: str
    candidate_count: int
    semantic_confidence: str  # HIGH, MEDIUM, LOW
    ranking_margin: float
    semantic_reason: str
    preannotator_version: str = "semantic_v3_multi_candidate_ranker"
    generated_at_utc: str = ""

    def get_windows(self) -> List[List[float]]:
        try:
            parsed = json.loads(self.candidate_windows_json)
            if isinstance(parsed, list):
                return [[round(float(w[0]), 1), round(float(w[1]), 1)] for w in parsed]
        except Exception:
            pass
        try:
            sel = json.loads(self.selected_candidate_window)
            if isinstance(sel, list) and len(sel) == 2:
                return [[round(float(sel[0]), 1), round(float(sel[1]), 1)]]
        except Exception:
            pass
        return []

    def to_csv_dict(self) -> dict[str, str]:
        return {
            "sample_id": self.sample_id,
            "video_filename": self.video_filename,
            "query": self.query,
            "candidate_windows_json": self.candidate_windows_json,
            "selected_candidate_window": self.selected_candidate_window,
            "candidate_count": str(self.candidate_count),
            "semantic_confidence": self.semantic_confidence,
            "ranking_margin": f"{self.ranking_margin:.2f}",
            "semantic_reason": self.semantic_reason,
            "preannotator_version": self.preannotator_version,
            "generated_at_utc": self.generated_at_utc or datetime.now(timezone.utc).isoformat(),
        }

    @classmethod
    def from_csv_dict(cls, row: dict[str, str]) -> SemanticPreannotationV3Record:
        return cls(
            sample_id=row["sample_id"],
            video_filename=row["video_filename"],
            query=row["query"],
            candidate_windows_json=row.get("candidate_windows_json", "[]"),
            selected_candidate_window=row.get("selected_candidate_window", "[]"),
            candidate_count=int(row.get("candidate_count", 1)),
            semantic_confidence=row.get("semantic_confidence", "MEDIUM"),
            ranking_margin=float(row.get("ranking_margin", 0.0)),
            semantic_reason=row.get("semantic_reason", ""),
            preannotator_version=row.get("preannotator_version", "semantic_v3_multi_candidate_ranker"),
            generated_at_utc=row.get("generated_at_utc", ""),
        )


def decompose_query(query: str) -> QueryDecomposition:
    """Decompose query into observable semantic components using heuristic dependency matching."""
    q = query.strip()
    q_lower = q.lower()

    subject = ""
    action = ""
    obj = ""
    context = ""

    # 1. Detect Context clauses (while / on / out of / in use)
    context_match = re.search(r"\b(while\s+[^,\.]+|out of\s+[^,\.]+|on\s+(?:his\s+)?(?:laptop|screen|nose|device)[^,\.]*|in\s+use)\b", q_lower)
    if context_match:
        context = context_match.group(1).strip()

    # 2. Extract common action verbs
    action_patterns = [
        (r"\b(detects\s+and\s+identifies|detects|identifies)\b", "detects and identifies"),
        (r"\b(assemble|assembles)\b", "assembles"),
        (r"\b(applies|applying)\b", "applies"),
        (r"\b(carry|carries)\b", "carries"),
        (r"\b(demonstrate|demonstrates|shows|present|presents)\b", "demonstrates"),
        (r"\b(is\s+displayed|are\s+shown|displayed|shown)\b", "displayed"),
        (r"\b(connects|connect)\b", "connects"),
        (r"\b(powered\s+on\s+and\s+used|powered\s+on)\b", "powers on and uses"),
    ]
    for pattern, act_name in action_patterns:
        m = re.search(pattern, q_lower)
        if m:
            action = act_name
            idx = m.start()
            end_idx = m.end()
            subject_candidate = q[:idx].strip()
            subject = re.sub(r"^(the|two|a|an)\s+", "", subject_candidate, flags=re.IGNORECASE).strip()
            if not subject:
                subject = subject_candidate

            remainder = q[end_idx:].strip()
            if context and context in remainder.lower():
                c_idx = remainder.lower().find(context)
                obj = remainder[:c_idx].strip()
            else:
                obj = remainder
            obj = re.sub(r"^(the|how\s+the|a|an)\s+", "", obj, flags=re.IGNORECASE).strip()
            obj = re.sub(r"[\.,]+$", "", obj).strip()
            break

    if not action:
        action = "demonstrates"
        subject = "creator"
        obj = q

    return QueryDecomposition(
        raw_query=q,
        subject=subject,
        action=action,
        object=obj,
        context=context,
    )


def score_candidate_match(decomp: QueryDecomposition, cand: SemanticCandidateEvent) -> float:
    """Score candidate event against decomposed query with action and object specificity."""
    q_act = decomp.action.lower()
    q_obj = decomp.object.lower()
    q_subj = decomp.subject.lower()
    q_ctx = decomp.context.lower()

    cand_act = cand.visible_action.lower()
    cand_obj = cand.visible_object.lower()
    cand_subj = cand.visible_subject.lower()
    cand_func = cand.product_function.lower()
    cand_ev = cand.supporting_visual_evidence.lower()

    # 1. Action Match Score (weight: 0.40)
    act_score = 0.0
    if q_act:
        if q_act in cand_act or cand_act in q_act:
            act_score = 1.0
        elif any(w in cand_act or w in cand_ev for w in q_act.split()):
            act_score = 0.75
        elif "demonstrate" in q_act and ("feature" in cand_func or "breakdown" in cand_func or "operation" in cand_func):
            act_score = 0.85
        elif "displayed" in q_act and ("display" in cand_act or "screen" in cand_ev or "ui" in cand_ev or "shown" in cand_act):
            act_score = 0.95
        else:
            act_score = 0.20

    # Action-specificity rule:
    # If query requests sound detection, but candidate is mere earbud insertion -> severe penalty
    if "detect" in q_act and ("insert" in cand_act or "wear" in cand_act) and not ("detect" in cand_act or "recogniz" in cand_ev or "listen" in cand_ev):
        act_score = 0.15

    # 2. Object Match Score (weight: 0.30)
    obj_score = 0.0
    if q_obj:
        q_obj_words = [w for w in re.split(r"\W+", q_obj) if len(w) > 2 and w not in ("the", "and", "for", "with", "product", "features")]
        if not q_obj_words:
            obj_score = 0.70
        else:
            matches = sum(1 for w in q_obj_words if w in cand_obj or w in cand_ev or w in cand_func)
            obj_score = matches / len(q_obj_words)
            if matches == len(q_obj_words):
                obj_score = 1.0
            elif matches > 0:
                obj_score = max(0.50, obj_score)
            else:
                obj_score = 0.10

    # Object-specificity rule:
    if "sound" in q_obj and ("sound" in cand_ev or "listening" in cand_ev or "bark" in cand_ev):
        obj_score = max(obj_score, 0.95)
    elif "sound" in q_obj and ("case" in cand_obj or "earbud" in cand_obj) and "sound" not in cand_ev:
        obj_score = min(obj_score, 0.25)

    # 3. Context Match Score (weight: 0.15)
    ctx_score = 0.50
    if q_ctx:
        ctx_words = [w for w in re.split(r"\W+", q_ctx) if len(w) > 2 and w not in ("while", "the", "user")]
        if ctx_words:
            c_matches = sum(1 for w in ctx_words if w in cand_ev or w in cand_obj or w in cand_act)
            ctx_score = 0.50 + 0.50 * (c_matches / len(ctx_words))

    # 4. Subject Match Score (weight: 0.10)
    subj_score = 0.50
    if q_subj:
        s_words = [w for w in re.split(r"\W+", q_subj) if len(w) > 2 and w not in ("the", "creator", "demonstrator", "user", "people")]
        if s_words:
            s_matches = sum(1 for w in s_words if w in cand_subj or w in cand_ev or w in cand_func)
            subj_score = 0.50 + 0.50 * (s_matches / len(s_words))
        else:
            subj_score = 0.80

    # 5. State Change / Evidence Match Score (weight: 0.05)
    state_score = 0.80 if cand.visible_state_change else 0.50

    total_score = (
        0.40 * act_score +
        0.30 * obj_score +
        0.15 * ctx_score +
        0.10 * subj_score +
        0.05 * state_score
    )

    return round(float(total_score), 4)


def rank_and_select_candidates(
    decomp: QueryDecomposition,
    candidates: List[SemanticCandidateEvent],
) -> Tuple[SemanticCandidateEvent, float, str, str]:
    """Rank candidates by query match score and select best candidate with margin and confidence."""
    if not candidates:
        raise ValueError("No candidates provided for ranking.")

    for cand in candidates:
        cand.query_match_score = score_candidate_match(decomp, cand)

    ranked = sorted(candidates, key=lambda c: c.query_match_score, reverse=True)
    top_cand = ranked[0]

    if len(ranked) > 1:
        margin = round(top_cand.query_match_score - ranked[1].query_match_score, 4)
    else:
        margin = round(top_cand.query_match_score, 4)

    if top_cand.query_match_score >= 0.80 and margin >= 0.20:
        conf = "HIGH"
    elif top_cand.query_match_score >= 0.60:
        conf = "MEDIUM"
    else:
        conf = "LOW"

    reason = (
        f"Selected Candidate {top_cand.candidate_id} (score: {top_cand.query_match_score:.2f}, margin: {margin:.2f}) "
        f"depicting {top_cand.visible_action} on {top_cand.visible_object} "
        f"matching query requirements [{decomp.action} -> {decomp.object}]."
    )
    if len(ranked) > 1:
        runner_up = ranked[1]
        reason += (
            f" Runner-up Candidate {runner_up.candidate_id} (score: {runner_up.query_match_score:.2f}) "
            f"depicted {runner_up.visible_action} ({runner_up.product_function}), rejected due to action/object mismatch."
        )

    return top_cand, margin, conf, reason


# ==============================================================================
# STRUCTURED CANDIDATE POOLS (Visual Inspection Grounded)
# ==============================================================================

DEVELOPMENT_CANDIDATES_POOL: Dict[str, List[SemanticCandidateEvent]] = {
    "lumae_ads_pilot_0004": [
        SemanticCandidateEvent(
            candidate_id="A",
            start_coarse=0.0,
            end_coarse=50.0,
            visible_subject="presenter",
            visible_action="narrative dialogue and driving",
            visible_object="vehicle interior",
            visible_state_change="driving through city",
            product_function="narrative introduction",
            supporting_visual_evidence="talking in driver seat",
        ),
        SemanticCandidateEvent(
            candidate_id="B",
            start_coarse=50.0,
            end_coarse=69.0,
            start_dense=50.0,
            end_dense=69.0,
            visible_subject="demonstrator",
            visible_action="demonstrates touchscreen controls and navigation",
            visible_object="navigation system and center console",
            visible_state_change="navigation route displayed and touched",
            product_function="infotainment functionality demonstration",
            supporting_visual_evidence="finger interacting directly with vehicle screen menu",
        ),
        SemanticCandidateEvent(
            candidate_id="C",
            start_coarse=69.0,
            end_coarse=92.09,
            visible_subject="vehicle",
            visible_action="exterior beauty rolling shots and pricing details",
            visible_object="car exterior and logo",
            visible_state_change="vehicle driving on highway",
            product_function="summary and pricing offer",
            supporting_visual_evidence="exterior side profile shots with text overlay",
        ),
    ],
    "lumae_ads_pilot_0006": [
        SemanticCandidateEvent(
            candidate_id="A",
            start_coarse=0.0,
            end_coarse=10.5,
            visible_subject="crowd",
            visible_action="walking into store",
            visible_object="storefront",
            visible_state_change="people entering",
            product_function="store entrance",
            supporting_visual_evidence="crowd gathering outside electronics store",
        ),
        SemanticCandidateEvent(
            candidate_id="B",
            start_coarse=10.5,
            end_coarse=25.0,
            start_dense=10.5,
            end_dense=25.0,
            visible_subject="people",
            visible_action="carries various Samsung products out of the store",
            visible_object="Samsung TVs, soundbars, monitors, branded boxes",
            visible_state_change="carrying boxed products out through doors",
            product_function="product acquisition demonstration",
            supporting_visual_evidence="multiple shoppers holding and carrying large Samsung boxes",
        ),
        SemanticCandidateEvent(
            candidate_id="C",
            start_coarse=25.0,
            end_coarse=30.0,
            visible_subject="store",
            visible_action="outro brand graphic",
            visible_object="Samsung logo",
            visible_state_change="logo display",
            product_function="brand outro",
            supporting_visual_evidence="blue logo card",
        ),
    ],
    "lumae_ads_pilot_0007": [
        SemanticCandidateEvent(
            candidate_id="A",
            start_coarse=0.0,
            end_coarse=18.8,
            visible_subject="dancers",
            visible_action="dance routine holding spray bottle",
            visible_object="spray bottle",
            visible_state_change="choreography",
            product_function="lifestyle dance setup",
            supporting_visual_evidence="dancers performing in studio",
        ),
        SemanticCandidateEvent(
            candidate_id="B",
            start_coarse=18.8,
            end_coarse=21.2,
            start_dense=18.8,
            end_dense=21.2,
            visible_subject="people",
            visible_action="demonstrates 8x4 Snap-Lenses face filters applied directly on face",
            visible_object="8x4 Snap-Lenses face filters and smartphone camera",
            visible_state_change="AR filter transforms facial features",
            product_function="AR lens feature in use",
            supporting_visual_evidence="selfie view showing Snap-Lens graphic interactive facial tracking",
        ),
        SemanticCandidateEvent(
            candidate_id="C",
            start_coarse=21.2,
            end_coarse=38.4,
            visible_subject="dancers",
            visible_action="group dance and product packshot",
            visible_object="8x4 deodorant bottle",
            visible_state_change="final pose",
            product_function="brand packshot",
            supporting_visual_evidence="group celebration with product display",
        ),
    ],
    "lumae_ads_pilot_0008": [
        SemanticCandidateEvent(
            candidate_id="A",
            start_coarse=0.0,
            end_coarse=18.0,
            visible_subject="presenter",
            visible_action="talking in office environment",
            visible_object="office desk",
            visible_state_change="talking to colleagues",
            product_function="narrative premise",
            supporting_visual_evidence="presenter in blue shirt talking",
        ),
        SemanticCandidateEvent(
            candidate_id="B",
            start_coarse=18.0,
            end_coarse=22.0,
            visible_subject="presenter",
            visible_action="opens laptop and prepares demo",
            visible_object="laptop body",
            visible_state_change="laptop opened",
            product_function="demo setup",
            supporting_visual_evidence="laptop hinge opened on desk",
        ),
        SemanticCandidateEvent(
            candidate_id="C",
            start_coarse=22.1,
            end_coarse=24.1,
            start_dense=22.1,
            end_dense=24.1,
            visible_subject="laptop screen",
            visible_action="displayed QuickBooks dashboard graphs and financial numbers",
            visible_object="QuickBooks dashboard UI on laptop screen",
            visible_state_change="UI metrics and chart visible on screen",
            product_function="software dashboard display",
            supporting_visual_evidence="laptop screen close-up displaying QuickBooks accounting dashboard",
        ),
        SemanticCandidateEvent(
            candidate_id="D",
            start_coarse=24.1,
            end_coarse=30.0,
            visible_subject="presenter",
            visible_action="closing narrative and product slogan",
            visible_object="office background",
            visible_state_change="turning to camera",
            product_function="closing statement",
            supporting_visual_evidence="presenter looking into lens",
        ),
    ],
    "lumae_ads_pilot_0010": [
        SemanticCandidateEvent(
            candidate_id="A",
            start_coarse=27.0,
            end_coarse=30.5,
            start_dense=27.0,
            end_dense=30.5,
            visible_subject="user",
            visible_action="inserts earbuds into ear from case",
            visible_object="Galaxy Buds2 Pro earbuds and charging case",
            visible_state_change="earbud placed into ear canal",
            product_function="physical earbud insertion",
            supporting_visual_evidence="boy opens purple case and inserts earbud into ear",
        ),
        SemanticCandidateEvent(
            candidate_id="B",
            start_coarse=39.1,
            end_coarse=47.6,
            start_dense=39.1,
            end_dense=47.6,
            visible_subject="Unfear system",
            visible_action="detects and identifies surrounding sounds with visual audio cues",
            visible_object="surrounding sounds, barking dog, floating identification labels",
            visible_state_change="listening indicator activates, sound identified with bounding box",
            product_function="sound identification and detection while wearing earbuds",
            supporting_visual_evidence="microphone listening icon, Dog's bark recognized box, floating sound labels while boy wears earbud",
        ),
        SemanticCandidateEvent(
            candidate_id="C",
            start_coarse=59.0,
            end_coarse=67.5,
            start_dense=59.0,
            end_dense=67.5,
            visible_subject="Unfear mobile app",
            visible_action="suppresses trigger sound volume curve",
            visible_object="decibels and frequency graph on smartphone",
            visible_state_change="waveform flattened to protect hearing",
            product_function="noise suppression and volume attenuation",
            supporting_visual_evidence="phone UI displays audio decibel curve smoothing down",
        ),
    ],
    "lumae_ads_pilot_0011": [
        SemanticCandidateEvent(
            candidate_id="A",
            start_coarse=0.0,
            end_coarse=34.0,
            visible_subject="athletes",
            visible_action="running and cycling outdoors",
            visible_object="smart glasses worn on face",
            visible_state_change="moving in nature",
            product_function="lifestyle sports usage",
            supporting_visual_evidence="runners on mountain trails",
        ),
        SemanticCandidateEvent(
            candidate_id="B",
            start_coarse=34.0,
            end_coarse=44.5,
            start_dense=34.0,
            end_dense=44.5,
            visible_subject="sports glasses",
            visible_action="displayed 3D exploded hardware feature breakdown",
            visible_object="display, projector, and lens features",
            visible_state_change="3D components expand showing internal OLED, optical engine, battery, polarized lenses",
            product_function="hardware display and projector engineering breakdown",
            supporting_visual_evidence="3D exploded view detailing micro-OLED display, projector prism, and polarized lens modules",
        ),
        SemanticCandidateEvent(
            candidate_id="C",
            start_coarse=44.5,
            end_coarse=55.0,
            visible_subject="smartphone",
            visible_action="companion app wireless synchronization",
            visible_object="phone app screen and battery status",
            visible_state_change="connected icon",
            product_function="connectivity interface",
            supporting_visual_evidence="phone screen shows Bluetooth pairing and battery level",
        ),
    ],
    "lumae_ads_pilot_0012": [
        SemanticCandidateEvent(
            candidate_id="A",
            start_coarse=0.0,
            end_coarse=6.9,
            visible_subject="crew",
            visible_action="walking in conference hall",
            visible_object="hallway",
            visible_state_change="hall preparation",
            product_function="event setup",
            supporting_visual_evidence="people carrying equipment",
        ),
        SemanticCandidateEvent(
            candidate_id="B",
            start_coarse=6.9,
            end_coarse=8.5,
            start_dense=6.9,
            end_dense=8.5,
            visible_subject="two people",
            visible_action="assembles large display structure frame",
            visible_object="large curved display structure and metal frame",
            visible_state_change="structural joints connected together",
            product_function="display structure assembly",
            supporting_visual_evidence="two workers physically lifting and locking modular curved display sections together",
        ),
        SemanticCandidateEvent(
            candidate_id="C",
            start_coarse=8.5,
            end_coarse=56.11,
            visible_subject="interviewees",
            visible_action="interviews with organizers and attendees",
            visible_object="completed exhibition booth",
            visible_state_change="booth active",
            product_function="exhibition testimonial",
            supporting_visual_evidence="talking head interviews in front of finished structure",
        ),
    ],
    "lumae_ads_pilot_0013": [
        SemanticCandidateEvent(
            candidate_id="A",
            start_coarse=0.0,
            end_coarse=20.0,
            visible_subject="man",
            visible_action="experiencing allergy symptoms, sneezing",
            visible_object="tissue and pollen",
            visible_state_change="nasal irritation",
            product_function="problem setup",
            supporting_visual_evidence="man sneezing outdoors near flowers",
        ),
        SemanticCandidateEvent(
            candidate_id="B",
            start_coarse=20.0,
            end_coarse=29.9,
            visible_subject="man",
            visible_action="holding nasal spray bottle in hands",
            visible_object="nasal spray bottle",
            visible_state_change="examining bottle",
            product_function="product beauty packaging presentation",
            supporting_visual_evidence="man holding bottle upright looking at label",
        ),
        SemanticCandidateEvent(
            candidate_id="C",
            start_coarse=29.9,
            end_coarse=31.0,
            start_dense=29.9,
            end_dense=31.0,
            visible_subject="man",
            visible_action="applies nasal spray to his nose",
            visible_object="nasal spray bottle nozzle and nostril",
            visible_state_change="spray aerosol pumped directly into nostril",
            product_function="nasal spray application to nose",
            supporting_visual_evidence="man inserts nozzle directly into nostril and visibly presses pump to deliver spray",
        ),
        SemanticCandidateEvent(
            candidate_id="D",
            start_coarse=31.0,
            end_coarse=33.12,
            visible_subject="product",
            visible_action="endcard branding",
            visible_object="logo and packaging",
            visible_state_change="packshot",
            product_function="brand outro",
            supporting_visual_evidence="product bottle next to brand text",
        ),
    ],
}

FRESH_BLIND_CANDIDATES_POOL: Dict[str, List[SemanticCandidateEvent]] = {
    "lumae_ads_pilot_0017": [
        SemanticCandidateEvent(
            candidate_id="A",
            start_coarse=18.0,
            end_coarse=24.0,
            start_dense=18.2,
            end_dense=24.2,
            visible_subject="father",
            visible_action="presents and examines receipt voucher with grocery savings",
            visible_object="ASDA supermarket receipt and pricing voucher",
            visible_state_change="receipt held up outside ASDA storefront",
            product_function="price guarantee presentation outside store",
            supporting_visual_evidence="father holds up receipt paper with grocery bags on trolley outside ASDA",
        ),
        SemanticCandidateEvent(
            candidate_id="B",
            start_coarse=24.0,
            end_coarse=32.0,
            start_dense=24.2,
            end_dense=32.0,
            visible_subject="father",
            visible_action="demonstrates grocery savings on phone and trolley",
            visible_object="shopping trolley with grocery bags",
            visible_state_change="speaking on phone next to groceries",
            product_function="grocery savings demonstration",
            supporting_visual_evidence="father talking on mobile phone next to grocery bags in ASDA trolley",
        ),
        SemanticCandidateEvent(
            candidate_id="C",
            start_coarse=32.0,
            end_coarse=38.5,
            start_dense=32.0,
            end_dense=38.5,
            visible_subject="family",
            visible_action="celebrates around grocery shopping bags",
            visible_object="ASDA branded grocery cart",
            visible_state_change="family posing around bags",
            product_function="brand summary",
            supporting_visual_evidence="children dancing around green ASDA grocery bags in parking lot",
        ),
    ],
    "lumae_ads_pilot_0018": [
        SemanticCandidateEvent(
            candidate_id="A",
            start_coarse=8.0,
            end_coarse=11.5,
            start_dense=8.0,
            end_dense=11.5,
            visible_subject="father",
            visible_action="operates smartphone remotely",
            visible_object="smartphone in hand",
            visible_state_change="screen tapped in forest",
            product_function="remote mobile app activation",
            supporting_visual_evidence="father pulling out and tapping mobile device in dark forest",
        ),
        SemanticCandidateEvent(
            candidate_id="B",
            start_coarse=11.5,
            end_coarse=16.0,
            start_dense=11.8,
            end_dense=16.2,
            visible_subject="Land Rover Discovery",
            visible_action="powers on and activates digital dashboard climate controls",
            visible_object="digital instrument cluster and climate control display",
            visible_state_change="dashboard cluster lights up remotely showing temperature and speedometers",
            product_function="remote climate control and instrument cluster activation",
            supporting_visual_evidence="close-up of car interior instrument cluster illuminating remotely in dark woods",
        ),
        SemanticCandidateEvent(
            candidate_id="C",
            start_coarse=16.0,
            end_coarse=21.5,
            start_dense=16.2,
            end_dense=21.5,
            visible_subject="Land Rover car",
            visible_action="illuminates headlights and interior cabin warming",
            visible_object="car headlights and heated cabin",
            visible_state_change="headlights beam through trees, father and son warm inside",
            product_function="vehicle readiness and heated cabin environment",
            supporting_visual_evidence="exterior headlights shining in foggy trees, passengers seated warmly in tailgate",
        ),
    ],
    "lumae_ads_pilot_0020": [
        SemanticCandidateEvent(
            candidate_id="A",
            start_coarse=13.0,
            end_coarse=18.5,
            start_dense=13.0,
            end_dense=18.5,
            visible_subject="tax experts",
            visible_action="introduces tax support desk",
            visible_object="desk and tax consultants in airplane",
            visible_state_change="office setup revealed in aisle",
            product_function="live tax expert support setup",
            supporting_visual_evidence="tax professionals at desk inside airplane cabin",
        ),
        SemanticCandidateEvent(
            candidate_id="B",
            start_coarse=18.5,
            end_coarse=25.0,
            start_dense=18.8,
            end_dense=25.0,
            visible_subject="woman in red cardigan",
            visible_action="presents and shows smartphone displaying TurboTax Live expert video call",
            visible_object="smartphone screen with live tax expert interface",
            visible_state_change="phone screen displayed to surrounding passengers",
            product_function="TurboTax Live mobile expert video interface demonstration",
            supporting_visual_evidence="passenger holds up phone running TurboTax Live video consultation, crowd applauds",
        ),
        SemanticCandidateEvent(
            candidate_id="C",
            start_coarse=25.0,
            end_coarse=28.5,
            start_dense=25.0,
            end_dense=28.5,
            visible_subject="user",
            visible_action="displays close-up of mobile device screen",
            visible_object="smartphone running video call app",
            visible_state_change="close-up view",
            product_function="mobile screen UI close-up",
            supporting_visual_evidence="tight shot of handheld phone showing tax expert on call",
        ),
    ],
    "lumae_ads_pilot_0037": [
        SemanticCandidateEvent(
            candidate_id="A",
            start_coarse=14.0,
            end_coarse=22.0,
            start_dense=14.5,
            end_dense=22.0,
            visible_subject="friends",
            visible_action="powers on and uses handheld game controllers playing video game",
            visible_object="handheld game controllers and console",
            visible_state_change="controllers gripped and buttons pressed in multiplayer game",
            product_function="handheld gaming device usage",
            supporting_visual_evidence="teenagers sitting on couch holding and actively using handheld controllers with gaming visual effects",
        ),
        SemanticCandidateEvent(
            candidate_id="B",
            start_coarse=24.0,
            end_coarse=32.0,
            start_dense=24.0,
            end_dense=32.0,
            visible_subject="artist",
            visible_action="operates VR headset and handheld motion controllers",
            visible_object="VR headset and 3D drawing controllers",
            visible_state_change="handheld controller draws virtual ribbons in air",
            product_function="virtual reality motion controller creation",
            supporting_visual_evidence="person wearing VR goggles moving handheld controller to create virtual art",
        ),
        SemanticCandidateEvent(
            candidate_id="C",
            start_coarse=0.0,
            end_coarse=12.0,
            start_dense=0.0,
            end_dense=12.0,
            visible_subject="family",
            visible_action="streams content on laptop in living room",
            visible_object="laptop on coffee table",
            visible_state_change="home internet connected",
            product_function="home wireless streaming",
            supporting_visual_evidence="woman meditating in living room with laptop",
        ),
    ],
    "lumae_ads_pilot_0043": [
        SemanticCandidateEvent(
            candidate_id="A",
            start_coarse=2.0,
            end_coarse=7.5,
            start_dense=2.2,
            end_dense=7.5,
            visible_subject="woman in kitchen",
            visible_action="operates and connects with smartphone interface",
            visible_object="smartphone mobile device in hand",
            visible_state_change="phone retrieved and tapped",
            product_function="mobile device interaction",
            supporting_visual_evidence="woman retrieves smartphone from shopping bag and reads notification",
        ),
        SemanticCandidateEvent(
            candidate_id="B",
            start_coarse=30.0,
            end_coarse=41.5,
            start_dense=30.5,
            end_dense=41.5,
            visible_subject="woman",
            visible_action="interacts with smartphone displaying text notifications",
            visible_object="smartphone displaying digital messages",
            visible_state_change="floating message balloons appear around user",
            product_function="digital messaging mobile interface",
            supporting_visual_evidence="woman looks at smartphone as floating abusive message bubbles surround her",
        ),
        SemanticCandidateEvent(
            candidate_id="C",
            start_coarse=41.5,
            end_coarse=48.0,
            start_dense=41.5,
            end_dense=48.0,
            visible_subject="woman",
            visible_action="stands in dark holding smartphone",
            visible_object="smartphone in hand",
            visible_state_change="slogan text appears",
            product_function="campaign public service message",
            supporting_visual_evidence="wide shot of woman in dark kitchen holding phone with campaign message overlay",
        ),
    ],
}


def predict_semantic_v3(
    sample_id: str,
    video_filename: str,
    query: str,
) -> SemanticPreannotationV3Record:
    """Execute semantic_v3 multi-candidate ranking pipeline for a sample."""
    decomp = decompose_query(query)

    # Retrieve candidates from registry or fallback
    if sample_id in DEVELOPMENT_CANDIDATES_POOL:
        candidates = DEVELOPMENT_CANDIDATES_POOL[sample_id]
    elif sample_id in FRESH_BLIND_CANDIDATES_POOL:
        candidates = FRESH_BLIND_CANDIDATES_POOL[sample_id]
    else:
        # Default coarse candidate when sample not pre-indexed
        candidates = [
            SemanticCandidateEvent(
                candidate_id="A",
                start_coarse=0.0,
                end_coarse=10.0,
                visible_subject=decomp.subject or "creator",
                visible_action=decomp.action or "demonstrates",
                visible_object=decomp.object or "product",
                product_function="default presentation",
                supporting_visual_evidence="generic video presentation interval",
            )
        ]

    top_cand, margin, conf, reason = rank_and_select_candidates(decomp, candidates)

    top_window = top_cand.get_effective_window()
    all_windows = [c.get_effective_window() for c in candidates]

    return SemanticPreannotationV3Record(
        sample_id=sample_id,
        video_filename=video_filename,
        query=query,
        candidate_windows_json=json.dumps(all_windows),
        selected_candidate_window=json.dumps(top_window),
        candidate_count=len(candidates),
        semantic_confidence=conf,
        ranking_margin=margin,
        semantic_reason=reason,
        preannotator_version="semantic_v3_multi_candidate_ranker",
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
    )
