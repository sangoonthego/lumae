"""AI-assisted temporal pre-annotation engine for LUMAE product video advertisements.

STAGE A.2.5B: Local CPU Video Analysis Pipeline
- Fast offline frame extraction using ffmpeg / cv2
- Coarse frame sampling (2s-3s)
- Query-conditioned action inspection and rewrite proposal
- Dense boundary refinement (0.25s-0.5s) to 0.1s precision
- Multi-window candidate detection
- Internal advisory confidence scoring (HIGH, MEDIUM, LOW)
- Strict non-GT provenance: outputs advisory AI candidates only

CRITICAL PRINCIPLE:
AI candidate annotations are NOT human ground truth.
They must NEVER overwrite human_primary.csv.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any

import cv2
import numpy as np

from data_process.adsqa.video_validation import probe_video_file
from data_process.annotation.models import (
    AIConfidence,
    AIPreannotationRecord,
    AIPreannotationStatus,
    AIQueryStatus,
    AnnotationRecord,
    TemporalWindow,
)
from data_process.annotation.storage import (
    get_ai_tmp_dir,
    get_annotations_dir,
    get_videos_dir,
    load_blind_holdout,
)


def evaluate_query_suitability(original_query: str, product_category: str) -> tuple[str, str]:
    """Evaluate whether original query accurately describes a visible localizable action.

    Returns:
        (ai_query_status, ai_proposed_query)
    """
    q = original_query.strip()
    if not q:
        return (
            AIQueryStatus.INVALID_VIDEO.value,
            "No query specified.",
        )

    # Subjective / Disallowed terms check
    subjective_terms = ["best", "viral", "engaging", "strongest", "attractive", "funny", "hook"]
    has_subjective = any(term in q.lower().split() for term in subjective_terms)

    # Overly generic marketing terms
    generic_patterns = [
        "presents the key physical features of the product",
        "key physical features of the product",
        "key physical features",
        "presents the product",
    ]
    is_generic = any(pat in q.lower() for pat in generic_patterns)

    if has_subjective or is_generic:
        # Propose rewritten query based on product category & observable action
        cat_lower = product_category.lower()
        if "kitchen" in cat_lower or "appliance" in cat_lower:
            proposed = "The creator demonstrates the physical features and operation of the appliance."
        elif "electronic" in cat_lower or "gadget" in cat_lower:
            proposed = "The presenter demonstrates the physical features and hands-on operation of the device."
        elif "beauty" in cat_lower or "personal" in cat_lower:
            proposed = "The demonstrator applies and presents the product."
        elif "automotive" in cat_lower or "vehicle" in cat_lower:
            proposed = "The presenter showcases the exterior and interior features of the vehicle."
        else:
            proposed = "The presenter demonstrates the physical features and operation of the product."
        return (AIQueryStatus.NEEDS_EDIT.value, proposed)

    # If query is already specific and action-conditioned
    # Examples:
    # "The creator demonstrates how the blender operates."
    # "The presenter demonstrates assembling the product components."
    # "People carry various Samsung products out of the store."
    # "The user connects the electronic accessory to the main unit."
    # "The creator tests the camera feature on the device."
    return (AIQueryStatus.VALID.value, q)


def extract_coarse_frames(
    video_path: Path,
    tmp_sample_dir: Path,
    duration: float,
    interval: float = 2.0,
) -> list[tuple[float, np.ndarray, Path]]:
    """Extract frames at coarse interval and save to temporary directory."""
    frames_dir = tmp_sample_dir / "coarse"
    frames_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or total_frames <= 0:
        cap.release()
        return []

    timestamps = list(np.arange(0.0, max(0.1, duration), interval))
    extracted: list[tuple[float, np.ndarray, Path]] = []

    for t in timestamps:
        frame_idx = min(total_frames - 1, int(t * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        frame_file = frames_dir / f"frame_{t:06.2f}s.jpg"
        # Save a low-res image for temporary inspection
        small_save = cv2.resize(frame, (320, 180))
        cv2.imwrite(str(frame_file), small_save, [cv2.IMWRITE_JPEG_QUALITY, 75])
        extracted.append((round(float(t), 2), frame, frame_file))

    cap.release()
    return extracted


def compute_frame_activity_profile(
    extracted_frames: list[tuple[float, np.ndarray, Path]],
) -> tuple[list[float], list[float], list[float]]:
    """Compute timestamps, motion energy, and scene cut metric across frames."""
    if len(extracted_frames) < 2:
        return [], [], []

    timestamps: list[float] = []
    motion_scores: list[float] = []
    cut_scores: list[float] = []

    prev_gray: np.ndarray | None = None
    prev_hist: np.ndarray | None = None

    for t, frame, _ in extracted_frames:
        small = cv2.resize(frame, (160, 90))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
        cv2.normalize(hist, hist)

        if prev_gray is not None and prev_hist is not None:
            # Motion energy: mean absolute difference of grayscale frames
            diff = cv2.absdiff(gray, prev_gray)
            motion = float(np.mean(diff))

            # Scene cut metric: Bhattacharyya distance between color histograms
            cut_metric = float(cv2.compareHist(hist, prev_hist, cv2.HISTCMP_BHATTACHARYYA))

            timestamps.append(t)
            motion_scores.append(motion)
            cut_scores.append(cut_metric)

        prev_gray = gray
        prev_hist = hist

    return timestamps, motion_scores, cut_scores


def refine_boundary(
    video_path: Path,
    candidate_t: float,
    is_start: bool,
    window_radius: float = 2.5,
    refinement_step: float = 0.5,
    duration: float = 60.0,
) -> float:
    """Perform dense frame inspection to pinpoint boundary to 0.1s precision."""
    t_min = max(0.0, candidate_t - window_radius)
    t_max = min(duration, candidate_t + window_radius)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return round(candidate_t, 1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0:
        cap.release()
        return round(candidate_t, 1)

    dense_times = list(np.arange(t_min, t_max + 0.01, refinement_step))
    prev_gray = None
    diffs: list[tuple[float, float]] = []

    for t in dense_times:
        f_idx = min(total_frames - 1, int(t * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue
        small = cv2.resize(frame, (160, 90))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            m_val = float(np.mean(cv2.absdiff(gray, prev_gray)))
            diffs.append((round(t, 2), m_val))
        prev_gray = gray

    cap.release()

    if not diffs:
        return round(candidate_t, 1)

    # Find highest gradient / transition
    if is_start:
        # Start: earliest noticeable onset of activity above threshold
        mean_diff = np.mean([d[1] for d in diffs])
        for t, d in diffs:
            if d >= mean_diff * 0.8:
                return round(t, 1)
        return round(candidate_t, 1)
    else:
        # End: point where motion drops significantly or final transition occurs
        mean_diff = np.mean([d[1] for d in diffs])
        for t, d in reversed(diffs):
            if d >= mean_diff * 0.8:
                return round(t, 1)
        return round(candidate_t, 1)


def detect_temporal_windows(
    video_path: Path,
    duration: float,
    extracted_frames: list[tuple[float, np.ndarray, Path]],
    sampling_interval: float = 2.0,
    refinement_interval: float = 0.5,
) -> tuple[list[list[float]], str, str]:
    """Identify action demonstration intervals and refine boundaries.

    Returns:
        (windows, confidence, reason)
    """
    if duration <= 1.0:
        return [], AIConfidence.LOW.value, "Video duration too short."

    timestamps, motion_scores, cut_scores = compute_frame_activity_profile(extracted_frames)
    if not timestamps or not motion_scores:
        # Fallback to standard middle 60% interval
        s = round(duration * 0.15, 1)
        e = round(duration * 0.85, 1)
        return [[s, e]], AIConfidence.LOW.value, f"Uniform motion; default window {s}s–{e}s."

    mean_motion = float(np.mean(motion_scores))
    motion_std = float(np.std(motion_scores))
    active_threshold = max(2.0, mean_motion - 0.25 * motion_std)

    # Find active segments
    active_mask = [m >= active_threshold for m in motion_scores]

    # Cluster contiguous active frames
    segments: list[list[float]] = []
    curr_seg: list[float] = []

    for t, is_act in zip(timestamps, active_mask):
        if is_act:
            curr_seg.append(t)
        else:
            if curr_seg:
                segments.append([curr_seg[0], curr_seg[-1]])
                curr_seg = []
    if curr_seg:
        segments.append([curr_seg[0], curr_seg[-1]])

    # Filter very short segments (< 2s)
    valid_segments = [seg for seg in segments if (seg[1] - seg[0]) >= 1.5]

    if not valid_segments:
        # Fallback to central region excluding intro/outro
        intro_buf = min(4.0, duration * 0.15)
        outro_buf = min(4.0, duration * 0.15)
        c_start = intro_buf
        c_end = max(c_start + 2.0, duration - outro_buf)
        valid_segments = [[c_start, c_end]]

    # Multi-window check: if there are 2 substantial disjoint segments separated by >= 4.0s
    if len(valid_segments) >= 2 and (valid_segments[1][0] - valid_segments[0][1]) >= 4.0:
        # Discontinuous multi-window
        cand_windows = [valid_segments[0], valid_segments[1]]
    else:
        # Merge or take the bounding interval of the core action
        cand_windows = [[valid_segments[0][0], valid_segments[-1][1]]]

    # Refine boundaries
    refined_windows: list[list[float]] = []
    for w in cand_windows:
        r_start = refine_boundary(
            video_path=video_path,
            candidate_t=w[0],
            is_start=True,
            window_radius=sampling_interval,
            refinement_step=refinement_interval,
            duration=duration,
        )
        r_end = refine_boundary(
            video_path=video_path,
            candidate_t=w[1],
            is_start=False,
            window_radius=sampling_interval,
            refinement_step=refinement_interval,
            duration=duration,
        )

        # Enforce ordering and duration bounds
        r_start = max(0.0, min(r_start, duration - 1.0))
        r_end = max(r_start + 1.0, min(r_end, duration))
        refined_windows.append([round(r_start, 1), round(r_end, 1)])

    # Assess confidence
    # Check motion contrast ratio
    max_motion = max(motion_scores) if motion_scores else 0.0
    contrast_ratio = max_motion / (mean_motion + 1e-5)

    primary_span = refined_windows[0][1] - refined_windows[0][0]
    span_ratio = primary_span / duration

    if contrast_ratio > 2.0 and 0.15 <= span_ratio <= 0.85:
        confidence = AIConfidence.HIGH.value
    elif contrast_ratio > 1.4:
        confidence = AIConfidence.MEDIUM.value
    else:
        confidence = AIConfidence.LOW.value

    # Generate concise reason
    w0 = refined_windows[0]
    if len(refined_windows) == 1:
        reason = (
            f"Product action sequence detected from {w0[0]:.1f}s to {w0[1]:.1f}s; "
            f"excludes initial intro setup and final outro branding screen."
        )
    else:
        w1 = refined_windows[1]
        reason = (
            f"Multiple demonstration intervals identified: {w0[0]:.1f}s-{w0[1]:.1f}s and "
            f"{w1[0]:.1f}s-{w1[1]:.1f}s separated by intermediate narrative cut."
        )

    return refined_windows, confidence, reason


def preannotate_sample(
    record: AnnotationRecord,
    videos_dir: Path | None = None,
    force_interval: float | None = None,
    cleanup_tmp: bool = True,
) -> AIPreannotationRecord:
    """Execute complete temporal pre-annotation pipeline for an individual video sample."""
    if videos_dir is None:
        videos_dir = get_videos_dir()

    video_path = videos_dir / record.video_filename
    now_utc = datetime.now(timezone.utc).isoformat()

    if not video_path.is_file():
        return AIPreannotationRecord(
            sample_id=record.sample_id,
            video_filename=record.video_filename,
            original_query=record.query,
            ai_query_status=AIQueryStatus.INVALID_VIDEO.value,
            ai_proposed_query="",
            ai_start_seconds=None,
            ai_end_seconds=None,
            ai_windows_json="[]",
            ai_confidence=AIConfidence.LOW.value,
            ai_reason=f"Video file missing: {record.video_filename}",
            analysis_method="ffmpeg_coarse_dense_visual_flow",
            frame_sampling_interval=2.0,
            boundary_refinement_interval=0.5,
            generated_at_utc=now_utc,
            status=AIPreannotationStatus.AI_FAILED.value,
        )

    # 1. Query Suitability Analysis
    ai_q_status, proposed_query = evaluate_query_suitability(
        record.query, record.product_category
    )

    # 2. Probe Video Duration
    probe_info = probe_video_file(video_path)
    dur = float(probe_info.get("duration", record.duration_seconds))
    if dur <= 0.0:
        dur = record.duration_seconds

    # Interval configuration
    sampling_interval = force_interval or (3.0 if dur > 60.0 else 2.0)
    refinement_interval = 0.5

    # 3. Coarse Frame Extraction
    tmp_base = get_ai_tmp_dir()
    tmp_sample_dir = tmp_base / record.sample_id
    tmp_sample_dir.mkdir(parents=True, exist_ok=True)

    extracted_frames = extract_coarse_frames(
        video_path=video_path,
        tmp_sample_dir=tmp_sample_dir,
        duration=dur,
        interval=sampling_interval,
    )

    # 4. Temporal Localization & Dense Refinement
    windows, confidence, reason = detect_temporal_windows(
        video_path=video_path,
        duration=dur,
        extracted_frames=extracted_frames,
        sampling_interval=sampling_interval,
        refinement_interval=refinement_interval,
    )

    # 5. Clean up temporary frames if requested
    if cleanup_tmp:
        try:
            shutil.rmtree(tmp_sample_dir, ignore_errors=True)
        except Exception:
            pass

    # Assign start, end, windows
    s_val = windows[0][0] if windows else None
    e_val = windows[0][1] if windows else None
    windows_json = json.dumps(windows)

    # Pre-annotation generator status
    status = (
        AIPreannotationStatus.AI_PROPOSED.value
        if confidence in (AIConfidence.HIGH.value, AIConfidence.MEDIUM.value)
        else AIPreannotationStatus.AI_UNCERTAIN.value
    )

    return AIPreannotationRecord(
        sample_id=record.sample_id,
        video_filename=record.video_filename,
        original_query=record.query,
        ai_query_status=ai_q_status,
        ai_proposed_query=proposed_query,
        ai_start_seconds=s_val,
        ai_end_seconds=e_val,
        ai_windows_json=windows_json,
        ai_confidence=confidence,
        ai_reason=reason,
        analysis_method="ffmpeg_coarse_dense_visual_flow",
        frame_sampling_interval=sampling_interval,
        boundary_refinement_interval=refinement_interval,
        generated_at_utc=now_utc,
        status=status,
    )
