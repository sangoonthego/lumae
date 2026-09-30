"""Streamlit Human Temporal Ground-Truth Review Application for LUMAE Ads Pilot.

Stage A.2.5B Protocol:
- AI-Assisted Human Verification Mode (AI candidate suggestion with accept/edit/reject actions)
- Blind Human Holdout Privacy (AI candidates strictly suppressed until human review is submitted)
- Prioritized review order (LOW confidence -> NEEDS_EDIT -> INVALID_VIDEO -> MEDIUM -> HIGH)
- Append-only provenance logging to human_review_log.jsonl
- Secondary independent review (human_02) and Adjudication (tIoU < 0.70)
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
import sys
import streamlit as st

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_process.adsqa.annotation_qc import compute_temporal_iou, evaluate_dual_annotations
from data_process.annotation.models import (
    AIConfidence,
    AIPreannotationRecord,
    AIQueryStatus,
    AnnotationMode,
    AnnotationRecord,
    OriginalQueryQuality,
    QueryAction,
    QueryReviewRecord,
    QueryReviewStatus,
    QueryStatus,
    ReviewLogEntry,
    ReviewStatus,
    TemporalWindow,
    WindowAction,
)
from data_process.annotation.storage import (
    append_review_audit_log,
    generate_secondary_worklist,
    get_annotations_dir,
    get_reports_dir,
    get_videos_dir,
    is_blind_holdout,
    load_adjudicated_records,
    load_ai_preannotations,
    load_blind_holdout,
    load_multi_windows,
    load_primary_records,
    load_secondary_records,
    save_adjudicated_record,
    save_multi_window,
    save_primary_records,
    save_secondary_records,
)
from data_process.annotation.validation import (
    validate_multi_windows,
    validate_query,
    validate_temporal_boundaries,
)

st.set_page_config(
    page_title="LUMAE Ads Human Ground-Truth Annotation",
    page_icon="🎬",
    layout="wide",
)


def initialize_session_state() -> None:
    """Initialize session state parameters."""
    if "mode" not in st.session_state:
        requested_mode = st.query_params.get("mode", "")
        st.session_state["mode"] = (
            AnnotationMode.QUERY_SANITATION.value
            if requested_mode == AnnotationMode.QUERY_SANITATION.value
            else AnnotationMode.AI_ASSISTED_PRIMARY.value
        )
    if "filter_status" not in st.session_state:
        st.session_state["filter_status"] = "ALL"
    if "current_index" not in st.session_state:
        st.session_state["current_index"] = 0
    if "has_unsaved_changes" not in st.session_state:
        st.session_state["has_unsaved_changes"] = False


def get_ai_priority_key(sample_id: str, ai_map: dict[str, AIPreannotationRecord]) -> tuple[int, str]:
    """Sort priority in AI-assisted mode (Step 24):
    1. LOW confidence
    2. NEEDS_EDIT
    3. INVALID_VIDEO
    4. MEDIUM confidence
    5. HIGH confidence
    99. Default / unknown
    """
    cand = ai_map.get(sample_id)
    if not cand:
        return (99, sample_id)
    if cand.ai_confidence == AIConfidence.LOW.value:
        return (1, sample_id)
    if cand.ai_query_status == AIQueryStatus.NEEDS_EDIT.value:
        return (2, sample_id)
    if cand.ai_query_status == AIQueryStatus.INVALID_VIDEO.value:
        return (3, sample_id)
    if cand.ai_confidence == AIConfidence.MEDIUM.value:
        return (4, sample_id)
    if cand.ai_confidence == AIConfidence.HIGH.value:
        return (5, sample_id)
    return (6, sample_id)


CLEAN_BLIND_SAMPLES: set[str] = {
    "lumae_ads_pilot_0008",
    "lumae_ads_pilot_0010",
    "lumae_ads_pilot_0011",
    "lumae_ads_pilot_0017",
    "lumae_ads_pilot_0018",
    "lumae_ads_pilot_0020",
    "lumae_ads_pilot_0037",
    "lumae_ads_pilot_0043",
}

CLEAN_QUERY_BLIND5_SAMPLES: set[str] = {
    "lumae_ads_pilot_0028",
    "lumae_ads_pilot_0033",
    "lumae_ads_pilot_0040",
    "lumae_ads_pilot_0045",
    "lumae_ads_pilot_0048",
}


def load_clean_blind_frozen_queries() -> dict[str, str]:
    """Load verified frozen queries for clean-blind 5 samples."""
    frozen_csv = (
        REPO_ROOT
        / "local_data"
        / "annotations"
        / "lumae_ads"
        / "blind_eval"
        / "semantic_v3_clean_query_blind5_queries_frozen.csv"
    )
    if not frozen_csv.is_file():
        return {}
    mapping = {}
    with frozen_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            mapping[r["sample_id"]] = r["human_final_query"]
    return mapping


def main() -> None:
    initialize_session_state()

    st.title("🎬 LUMAE Product Ads — Human Temporal Ground-Truth Review Tool")
    st.caption("Stage A.2.5B/E/G/I Protocol • Local CPU Only • AI-Assisted + Blind Holdout + Query Sanitation • 0.1s Precision")

    # 1. Sidebar - Mode & Progress Dashboard
    with st.sidebar:
        st.header("⚙️ Annotation Controls")

        mode_options = [
            AnnotationMode.AI_ASSISTED_PRIMARY.value,
            AnnotationMode.PRIMARY.value,
            AnnotationMode.BLIND_PRIMARY.value,
            AnnotationMode.QUERY_SANITATION.value,
            AnnotationMode.BLIND_QUERY_REVIEW.value,
            AnnotationMode.CLEAN_QUERY_BLIND_TEMPORAL.value,
            AnnotationMode.ASSISTED_TEMPORAL_REVIEW.value,
            AnnotationMode.SECONDARY.value,
            AnnotationMode.ADJUDICATION.value,
        ]
        mode_choice = st.radio(
            "Workflow Mode:",
            mode_options,
            index=mode_options.index(st.session_state["mode"]) if st.session_state["mode"] in mode_options else 0,
            help=(
                "AI_ASSISTED_PRIMARY: AI candidate pre-annotations with blind holdout privacy;\n"
                "PRIMARY: Manual primary review;\n"
                "BLIND_PRIMARY: Strict blind human review (AI pre-annotations completely suppressed);\n"
                "SECONDARY: Independent blind review (human_02);\n"
                "ADJUDICATION: Adjudication for tIoU < 0.70"
            ),
        )
        if mode_choice != st.session_state["mode"]:
            st.session_state["mode"] = mode_choice
            st.session_state["current_index"] = 0
            st.rerun()

        mode = st.session_state["mode"]

        # Load pre-annotations & holdout
        ai_candidates = load_ai_preannotations()
        holdout_info = load_blind_holdout()
        holdout_sample_ids = set(holdout_info.get("sample_ids", [])) | CLEAN_BLIND_SAMPLES

        # Load records depending on mode
        if mode in (
            AnnotationMode.AI_ASSISTED_PRIMARY.value,
            AnnotationMode.PRIMARY.value,
            AnnotationMode.BLIND_PRIMARY.value,
            AnnotationMode.QUERY_SANITATION.value,
            AnnotationMode.BLIND_QUERY_REVIEW.value,
        ):
            records = load_primary_records()
            annotator_label = "human_01 (Lead Reviewer)"
        elif mode == AnnotationMode.CLEAN_QUERY_BLIND_TEMPORAL.value:
            records = [r for r in load_primary_records() if r.sample_id in CLEAN_QUERY_BLIND5_SAMPLES]
            records.sort(key=lambda r: r.sample_id)
            annotator_label = "human_01 (Clean Blind Temporal Reviewer)"
        elif mode == AnnotationMode.ASSISTED_TEMPORAL_REVIEW.value:
            from data_process.annotation.assisted_deployment import load_state
            deployment_state = load_state()
            records = [r for r in load_primary_records() if r.sample_id in deployment_state["eligible"]]
            records.sort(key=lambda r: r.sample_id)
            annotator_label = "human_01 (Assisted Temporal Reviewer)"
        elif mode == AnnotationMode.SECONDARY.value:
            records = load_secondary_records()
            annotator_label = "human_02 (Secondary Independent Reviewer)"
        else:
            primary_all = load_primary_records()
            records = primary_all
            annotator_label = "human_adjudicator"

        # Apply prioritized sorting if in AI_ASSISTED_PRIMARY mode (Step 24)
        if mode == AnnotationMode.AI_ASSISTED_PRIMARY.value:
            records = sorted(records, key=lambda r: get_ai_priority_key(r.sample_id, ai_candidates))

        total_count = len(records)
        reviewed_count = sum(1 for r in records if r.is_human_reviewed())
        excluded_count = sum(1 for r in records if r.is_excluded())
        unreviewed_count = total_count - (reviewed_count + excluded_count)
        if mode == AnnotationMode.QUERY_SANITATION.value:
            from data_process.annotation.fastlane import progress, recover_incomplete_commits
            recover_incomplete_commits(REPO_ROOT)
            query_state = progress(REPO_ROOT)
            total_count = query_state["fastlane_total"]
            reviewed_count = query_state["temporal_reviewed"]
            excluded_count = query_state["excluded"]
            unreviewed_count = query_state["remaining"]
        completed_count = reviewed_count + excluded_count
        progress_pct = completed_count / total_count * 100 if total_count > 0 else 0.0

        st.markdown("---")
        st.subheader("📊 Progress Dashboard")
        st.metric("Query Worklist Samples" if mode == AnnotationMode.QUERY_SANITATION.value else "Total Pilot Samples", total_count)
        col_m1, col_m2 = st.columns(2)
        col_m1.metric("Reviewed", reviewed_count)
        col_m2.metric("Excluded", excluded_count)
        st.metric("Pending Review", unreviewed_count)
        st.progress(min(1.0, progress_pct / 100.0), text=f"Completion: {progress_pct:.1f}%")

        if mode == AnnotationMode.AI_ASSISTED_PRIMARY.value:
            st.markdown("---")
            st.subheader("🤖 AI Pre-Annotation State")
            st.caption(f"Candidates generated: **{len(ai_candidates)}** / {total_count}")
            st.caption(f"Blind holdout pool: **{len(holdout_sample_ids)}** samples")

        st.markdown("---")
        st.subheader("🔍 Filter Worklist")
        filter_opt = "ALL" if mode == AnnotationMode.QUERY_SANITATION.value else st.selectbox(
            "Filter by Status:",
            ["ALL", "UNREVIEWED", "REVIEWED", "EXCLUDED"],
            index=["ALL", "UNREVIEWED", "REVIEWED", "EXCLUDED"].index(st.session_state["filter_status"]),
        )
        if filter_opt != st.session_state["filter_status"]:
            st.session_state["filter_status"] = filter_opt
            st.session_state["current_index"] = 0
            st.rerun()

        # Secondary Worklist Generator
        if mode in (AnnotationMode.AI_ASSISTED_PRIMARY.value, AnnotationMode.PRIMARY.value) and reviewed_count >= 10:
            st.markdown("---")
            st.subheader("🎯 Dual-Annotation Gate")
            if st.button("Generate Secondary Worklist (20%)", help="Extracts >=20% usable samples with BLANK GT"):
                sec_rows = generate_secondary_worklist(records, fraction=0.20, seed=42)
                st.success(f"Generated {len(sec_rows)} secondary samples in secondary_worklist.csv!")
                st.rerun()

    if not records:
        if mode == AnnotationMode.SECONDARY.value:
            st.warning("Secondary worklist has not been generated yet. Complete primary reviews first.")
        else:
            st.error("No annotation records found. Check local_data/annotations/lumae_ads/.")
        return

    if mode == AnnotationMode.QUERY_SANITATION.value:
        from data_process.annotation.fastlane_ui import render_fastlane

        render_fastlane(records, REPO_ROOT)
        return

    # Apply filter
    filtered_records: list[tuple[int, AnnotationRecord]] = []
    for idx, r in enumerate(records):
        if filter_opt == "UNREVIEWED" and (r.is_human_reviewed() or r.is_excluded()):
            continue
        if filter_opt == "REVIEWED" and not r.is_human_reviewed():
            continue
        if filter_opt == "EXCLUDED" and not r.is_excluded():
            continue
        filtered_records.append((idx, r))

    if not filtered_records:
        st.info(f"No records matching filter '{filter_opt}'.")
        return

    # Clamp index
    if st.session_state["current_index"] >= len(filtered_records):
        st.session_state["current_index"] = max(0, len(filtered_records) - 1)

    curr_list_idx = st.session_state["current_index"]
    orig_idx, current_record = filtered_records[curr_list_idx]

    if mode == AnnotationMode.ASSISTED_TEMPORAL_REVIEW.value:
        from data_process.annotation.assisted_ui import render_assisted_temporal_review
        render_assisted_temporal_review(current_record, curr_list_idx, len(filtered_records))
        return

    # Check Blind Holdout, Clean Blind, and Clean Query Blind Condition (Stage A.2.5E/G/I/J)
    is_clean_blind = current_record.sample_id in CLEAN_BLIND_SAMPLES
    is_clean_query_blind = current_record.sample_id in CLEAN_QUERY_BLIND5_SAMPLES
    is_query_sanitation_mode = mode in (
        AnnotationMode.QUERY_SANITATION.value,
        AnnotationMode.BLIND_QUERY_REVIEW.value,
    )
    is_clean_query_blind_temporal = mode == AnnotationMode.CLEAN_QUERY_BLIND_TEMPORAL.value
    is_holdout = is_blind_holdout(current_record.sample_id) or is_clean_blind or is_clean_query_blind
    is_reviewed = current_record.is_human_reviewed()
    is_blind_mode = (
        is_query_sanitation_mode
        or is_clean_query_blind_temporal
        or (is_clean_blind and not is_reviewed)
        or (mode == AnnotationMode.BLIND_PRIMARY.value and not is_reviewed)
        or (mode == AnnotationMode.AI_ASSISTED_PRIMARY.value and is_holdout and not is_reviewed)
    )
    # Strict privacy: candidate pre-annotations completely suppressed in blind mode
    ai_cand = None if is_blind_mode else ai_candidates.get(current_record.sample_id)

    # 2. Main Content Header
    st.markdown(f"### Sample {curr_list_idx + 1} of {len(filtered_records)} — `{current_record.sample_id}`")

    # Badges row
    c_b1, c_b2, c_b3, c_b4 = st.columns(4)
    c_b1.info(f"**Category:** {current_record.product_category}")
    c_b2.info(f"**Duration:** {current_record.duration_seconds:.2f}s")
    c_b3.info(f"**Split:** {current_record.source_split}")
    status_color = "🟢" if current_record.is_human_reviewed() else ("🔴" if current_record.is_excluded() else "⚪")
    c_b4.info(f"**Status:** {status_color} {current_record.review_status}")

    # Layout: Video on Left, Review Form on Right
    col_video, col_form = st.columns([1.1, 1.0], gap="large")

    video_file = get_videos_dir() / current_record.video_filename

    with col_video:
        st.subheader("📹 Video Playback")
        if video_file.is_file():
            st.video(str(video_file))
            st.caption(f"File: `{current_record.video_filename}` • Total Duration: {current_record.duration_seconds:.2f}s")
        else:
            st.error(f"Physical video file not found at: `{video_file}`")

        # In ADJUDICATION mode, display primary vs secondary comparisons
        if mode == AnnotationMode.ADJUDICATION.value:
            st.markdown("---")
            st.subheader("⚖️ Dual-Annotation Comparison")
            sec_records = {r.sample_id: r for r in load_secondary_records()}
            sec_r = sec_records.get(current_record.sample_id)
            if sec_r and sec_r.is_human_reviewed():
                w_prim = current_record.get_effective_windows()
                w_sec = sec_r.get_effective_windows()
                tiou = compute_temporal_iou(w_prim, w_sec)
                st.write(f"**Primary Window (human_01):** `{w_prim}`")
                st.write(f"**Secondary Window (human_02):** `{w_sec}`")
                st.metric("Pairwise tIoU", f"{tiou:.4f}")
                if tiou >= 0.70:
                    st.success("✅ Agreement PASS (tIoU >= 0.70)")
                else:
                    st.warning("⚠️ ADJUDICATION REQUIRED (tIoU < 0.70)")
            else:
                st.info("No secondary review completed for this sample yet.")

    with col_form:
        if is_clean_query_blind_temporal:
            st.subheader("⏱️ Clean-Blind Temporal Review")
            st.warning(
                "🔒 **Stage A.2.5J Protocol • Clean-Blind Temporal Ground-Truth Review** — "
                "Human blind temporal annotation using verified queries only. "
                "AI candidate predictions, confidence values, ranking margins, and intervals "
                "are strictly suppressed from the interface and DOM."
            )
            frozen_queries = load_clean_blind_frozen_queries()
            display_query = frozen_queries.get(current_record.sample_id, current_record.query)

            st.markdown("**Frozen Human-Verified Query:**")
            st.markdown(f"> *{display_query}*")

            st.markdown("---")
            st.markdown("#### ⏱️ Temporal Ground-Truth Boundary (Seconds)")
            st.caption("Continuous 0.1s precision. Rule: earliest visual start to conclusion.")

            c_t1, c_t2 = st.columns(2)
            with c_t1:
                start_val = st.number_input(
                    "START (seconds):",
                    min_value=0.0,
                    max_value=max(0.0, current_record.duration_seconds),
                    value=None,
                    step=0.1,
                    format="%.1f",
                    key=f"clean_blind_start_{current_record.sample_id}",
                )
            with c_t2:
                end_val = st.number_input(
                    "END (seconds):",
                    min_value=0.0,
                    max_value=max(0.0, current_record.duration_seconds),
                    value=None,
                    step=0.1,
                    format="%.1f",
                    key=f"clean_blind_end_{current_record.sample_id}",
                )

            notes_input = st.text_area(
                "Annotation Notes:",
                value="",
                help="Note visual cues or boundary rationale.",
                key=f"notes_clean_blind_{current_record.sample_id}",
            )

            st.markdown("---")
            if st.button("Save Human Temporal Review", type="primary", use_container_width=True, key=f"btn_save_clean_blind_temporal_{current_record.sample_id}"):
                final_windows = [[round(start_val, 1), round(end_val, 1)]] if start_val is not None and end_val is not None else []
                success = handle_save(
                    record=current_record,
                    records=records,
                    query_status=QueryStatus.VALID.value,
                    final_query=display_query,
                    windows=final_windows,
                    notes=notes_input,
                    mode=mode,
                    is_blind_holdout_sample=True,
                    query_action=None,
                    window_action=None,
                    ai_candidate=None,
                    advance=True,
                )
                if success:
                    st.rerun()

            # Navigation buttons
            st.markdown("---")
            c_prev, c_next = st.columns(2)
            with c_prev:
                if st.button("⬅️ Previous Sample", key=f"btn_prev_{current_record.sample_id}", use_container_width=True, disabled=(curr_list_idx == 0)):
                    st.session_state["current_index"] = max(0, curr_list_idx - 1)
                    st.rerun()
            with c_next:
                if st.button("Next Sample ➡️", key=f"btn_next_{current_record.sample_id}", use_container_width=True, disabled=(curr_list_idx >= len(filtered_records) - 1)):
                    st.session_state["current_index"] = min(len(filtered_records) - 1, curr_list_idx + 1)
                    st.rerun()
            return

        if is_query_sanitation_mode:
            st.subheader("🔍 Query Sanitation & Verification")
            if is_clean_query_blind:
                st.warning(
                    "🔒 **Stage A.2.5I Protocol • Clean-Blind Query Review (BLIND_QUERY_REVIEW)** — "
                    "Independent human query verification only. Temporal boundary annotations and AI temporal "
                    "predictions are strictly suppressed from the interface and DOM."
                )
            else:
                st.info(
                    "📝 **Query Sanitation Mode** — Human verification of temporal query suitability. "
                    "Temporal boundary annotations are locked until query is verified."
                )

            from data_process.annotation.query_sanitation import (
                load_query_reviews,
                save_query_review,
            )

            existing_reviews = load_query_reviews()
            existing_q_rec = existing_reviews.get(current_record.sample_id)

            orig_query_val = current_record.original_query or current_record.query
            st.markdown("**Original Lumae-Generated Draft Query:**")
            st.markdown(f"> *{orig_query_val}*")

            suggestion = None
            if mode == AnnotationMode.QUERY_SANITATION.value:
                from data_process.annotation.assisted_deployment import _paths, _read_csv
                suggestion_path = _paths(REPO_ROOT)[0] / "ai_query_suggestions.csv"
                suggestion = next((r for r in _read_csv(suggestion_path)
                                   if r["sample_id"] == current_record.sample_id), None)
                if suggestion:
                    st.markdown("**AI query suggestion (advisory):**")
                    st.info(suggestion["ai_suggested_query"])
                    st.caption(f"Video evidence: {suggestion['observable_evidence']}")

            qual_options = [
                OriginalQueryQuality.VAGUE_BUT_RELEVANT.value,
                OriginalQueryQuality.VALID.value,
                OriginalQueryQuality.INCORRECT_FOR_VIDEO.value,
                OriginalQueryQuality.INVALID_OR_UNLOCALIZABLE.value,
            ]
            default_qual_idx = 0
            if existing_q_rec and existing_q_rec.original_query_quality in qual_options:
                default_qual_idx = qual_options.index(existing_q_rec.original_query_quality)
            elif suggestion and suggestion["query_quality_suggestion"] in qual_options:
                default_qual_idx = qual_options.index(suggestion["query_quality_suggestion"])

            q_qual_choice = st.radio(
                "Original Query Quality:",
                qual_options,
                index=default_qual_idx,
                key=f"q_qual_{current_record.sample_id}",
                help="Classify suitability of initial Lumae draft query template for the video.",
            )

            q_default = existing_q_rec.human_final_query if (existing_q_rec and existing_q_rec.human_final_query) else (suggestion["ai_suggested_query"] if suggestion else current_record.query)
            q_input_val = st.session_state.get(f"q_input_{current_record.sample_id}", q_default)
            final_query = st.text_area(
                "Final Human Query (Action-Conditioned & Observable):",
                value=q_input_val,
                help="Visible, observable, temporally localizable natural language query.",
                key=f"final_query_san_{current_record.sample_id}",
            )

            change_options = [
                "UNCHANGED",
                "NARROWED_INTENT",
                "REWRITTEN",
                "CORRECTED_ACTION",
                "SPECIFIED_OBJECT",
            ]
            default_change_idx = 0
            if existing_q_rec and existing_q_rec.query_change_type in change_options:
                default_change_idx = change_options.index(existing_q_rec.query_change_type)
            elif final_query.strip() != orig_query_val.strip():
                default_change_idx = 2  # REWRITTEN

            q_change_choice = st.selectbox(
                "Query Change Type:",
                change_options,
                index=default_change_idx,
                key=f"q_change_{current_record.sample_id}",
                help="Provenance category describing how the original draft was revised.",
            )

            default_localizable = existing_q_rec.query_localizable is True if existing_q_rec else False
            is_localizable = st.checkbox(
                "Event is temporally localizable in video",
                value=default_localizable,
                key=f"chk_localizable_{current_record.sample_id}",
            )

            is_observable = st.checkbox(
                "Event is visually observable in video",
                value=existing_q_rec.query_observable is True if existing_q_rec else False,
                key=f"chk_observable_{current_record.sample_id}",
            )

            human_reviewer = st.text_input(
                "Human reviewer ID:", value=existing_q_rec.reviewer_id if existing_q_rec and existing_q_rec.reviewer_id else "human_01",
                key=f"query_reviewer_{current_record.sample_id}",
            )

            default_notes = existing_q_rec.review_notes if (existing_q_rec and existing_q_rec.review_notes) else current_record.annotation_notes
            notes_input = st.text_area(
                "Query Review Notes:",
                value=default_notes,
                help="Note rationale for query classification or rewrite.",
                key=f"notes_san_{current_record.sample_id}",
            )

            if st.button("💾 Save Query Review", type="primary", use_container_width=True, key=f"btn_save_query_{current_record.sample_id}"):
                from datetime import datetime, timezone
                import time

                if final_query.strip() == orig_query_val.strip() and q_qual_choice == OriginalQueryQuality.VALID.value:
                    q_status = QueryReviewStatus.VALID_AS_IS.value
                    final_change_type = "UNCHANGED"
                elif q_qual_choice == OriginalQueryQuality.INVALID_OR_UNLOCALIZABLE.value:
                    q_status = QueryReviewStatus.EXCLUDE.value
                    final_change_type = q_change_choice if q_change_choice != "UNCHANGED" else "EXCLUDED"
                else:
                    q_status = QueryReviewStatus.HUMAN_EDITED.value
                    final_change_type = q_change_choice if q_change_choice != "UNCHANGED" else "REWRITTEN"

                if not human_reviewer.startswith("human_"):
                    st.error("A human_ reviewer ID is required.")
                    return
                if q_status != QueryReviewStatus.EXCLUDE.value and (not final_query.strip() or not is_localizable or not is_observable):
                    st.error("Approve only a nonempty, localizable, observable query.")
                    return

                q_rec = QueryReviewRecord(
                    sample_id=current_record.sample_id,
                    video_filename=current_record.video_filename,
                    source_dataset=current_record.source_dataset,
                    source_video_id=current_record.source_video_id,
                    original_query=orig_query_val,
                    ai_suggested_query=suggestion["ai_suggested_query"] if suggestion else "",
                    human_final_query=final_query.strip(),
                    query_review_status=q_status,
                    original_query_quality=q_qual_choice,
                    query_change_type=final_change_type,
                    query_localizable=is_localizable,
                    query_observable=is_observable,
                    reviewer_id=human_reviewer.strip(),
                    review_notes=notes_input.strip(),
                    reviewed_at_utc=datetime.now(timezone.utc).isoformat(),
                    query_version="1",
                    temporal_annotation_locked=True,
                )
                save_query_review(q_rec)
                st.success("✅ Query review saved successfully to query_reviews.csv!")
                time.sleep(0.4)
                st.rerun()

            # Navigation buttons
            st.markdown("---")
            c_prev, c_next = st.columns(2)
            with c_prev:
                if st.button("⬅️ Previous Sample", key=f"btn_prev_{current_record.sample_id}", use_container_width=True, disabled=(curr_list_idx == 0)):
                    st.session_state["current_index"] = max(0, curr_list_idx - 1)
                    st.rerun()
            with c_next:
                if st.button("Next Sample ➡️", key=f"btn_next_{current_record.sample_id}", use_container_width=True, disabled=(curr_list_idx >= len(filtered_records) - 1)):
                    st.session_state["current_index"] = min(len(filtered_records) - 1, curr_list_idx + 1)
                    st.rerun()
            return

        st.subheader("📝 Query & Temporal Boundaries")

        # BLIND HOLDOUT SUPPRESSION (Step 17, 21, Stage A.2.5E)
        if is_blind_mode:
            st.warning(
                "🔒 **Blind Human Holdout Sample (BLIND_PRIMARY)** — AI candidate pre-annotations are strictly suppressed "
                "until your independent primary human review is submitted."
            )
        elif mode == AnnotationMode.AI_ASSISTED_PRIMARY.value and ai_cand:
            # AI CANDIDATE CARD (Step 18)
            with st.container():
                st.markdown("#### 🤖 AI Candidate Pre-Annotation (Advisory Assistance)")
                c_ai1, c_ai2, c_ai3 = st.columns([1, 1, 1.2])

                q_color = "🟢" if ai_cand.ai_query_status == "VALID" else ("🟡" if ai_cand.ai_query_status == "NEEDS_EDIT" else "🔴")
                c_ai1.markdown(f"**AI Query:** {q_color} `{ai_cand.ai_query_status}`")

                conf_color = "🟢" if ai_cand.ai_confidence == "HIGH" else ("🟡" if ai_cand.ai_confidence == "MEDIUM" else "🟠")
                c_ai2.markdown(f"**Confidence:** {conf_color} `{ai_cand.ai_confidence}`")

                c_ai3.markdown(f"**Candidate Window:** `{ai_cand.get_windows()}`")

                if ai_cand.ai_proposed_query and ai_cand.ai_query_status == "NEEDS_EDIT":
                    st.info(f"**Proposed Query:** *{ai_cand.ai_proposed_query}*")

                if ai_cand.ai_reason:
                    st.caption(f"💡 *Rationale:* {ai_cand.ai_reason}")

                # Quick assist buttons
                c_btn1, c_btn2, c_btn3 = st.columns(3)
                with c_btn1:
                    if st.button("✨ Accept AI Query", key=f"btn_accept_q_{current_record.sample_id}", use_container_width=True):
                        st.session_state[f"q_input_{current_record.sample_id}"] = ai_cand.ai_proposed_query or ai_cand.original_query
                        st.session_state[f"q_action_{current_record.sample_id}"] = QueryAction.ACCEPTED_AI.value
                        st.rerun()

                with c_btn2:
                    if st.button("⏱️ Accept AI Window", key=f"btn_accept_w_{current_record.sample_id}", use_container_width=True):
                        cand_w = ai_cand.get_windows()
                        if len(cand_w) == 1:
                            st.session_state[f"start_{current_record.sample_id}"] = float(cand_w[0][0])
                            st.session_state[f"end_{current_record.sample_id}"] = float(cand_w[0][1])
                            st.session_state[f"multi_{current_record.sample_id}"] = False
                        elif len(cand_w) > 1:
                            st.session_state[f"multi_{current_record.sample_id}"] = True
                            st.session_state[f"mw_count_{current_record.sample_id}"] = len(cand_w)
                            for w_i, w in enumerate(cand_w):
                                st.session_state[f"mw_s_{w_i}_{current_record.sample_id}"] = float(w[0])
                                st.session_state[f"mw_e_{w_i}_{current_record.sample_id}"] = float(w[1])
                        st.session_state[f"w_action_{current_record.sample_id}"] = WindowAction.ACCEPTED_AI.value
                        st.rerun()

                with c_btn3:
                    if st.button("❌ Reject AI", key=f"btn_reject_ai_{current_record.sample_id}", use_container_width=True):
                        st.session_state[f"q_action_{current_record.sample_id}"] = QueryAction.REJECTED_AI.value
                        st.session_state[f"w_action_{current_record.sample_id}"] = WindowAction.REJECTED_AI.value
                        st.rerun()

                st.markdown("---")

        # Query verification
        st.markdown("**Original Query:**")
        st.markdown(f"> *{current_record.original_query or current_record.query}*")

        q_status_choice = st.radio(
            "Query Classification:",
            [QueryStatus.VALID.value, QueryStatus.NEEDS_EDIT.value, QueryStatus.INVALID_VIDEO.value],
            index=0 if current_record.review_status != ReviewStatus.EXCLUDED.value else 2,
            horizontal=True,
            help="VALID = observable event; NEEDS_EDIT = refine query; INVALID_VIDEO = unusable video",
            key=f"q_status_radio_{current_record.sample_id}",
        )

        # Default query input value
        q_default = st.session_state.get(f"q_input_{current_record.sample_id}", current_record.query)
        final_query = st.text_input(
            "Final Verified Query (Action-Conditioned):",
            value=q_default,
            disabled=(q_status_choice == QueryStatus.INVALID_VIDEO.value),
            help="Descriptive query describing visible product demonstration",
            key=f"final_query_{current_record.sample_id}",
        )

        st.markdown("---")
        st.markdown("#### ⏱️ Ground-Truth Temporal Window (Seconds)")
        st.caption("Continuous 0.1s precision. Rule: earliest visual start to conclusion.")

        # Multi-window support
        existing_windows = current_record.get_effective_windows()
        is_multi_default = len(existing_windows) > 1
        is_multi_state = st.session_state.get(f"multi_{current_record.sample_id}", is_multi_default)

        enable_multi = st.checkbox(
            "Multiple Discontinuous Windows",
            value=is_multi_state,
            key=f"multi_chk_{current_record.sample_id}",
        )

        if not enable_multi:
            # Single window inputs
            if is_blind_mode and not is_reviewed:
                default_start = st.session_state.get(f"start_{current_record.sample_id}", None)
                default_end = st.session_state.get(f"end_{current_record.sample_id}", None)
            else:
                default_start = st.session_state.get(
                    f"start_{current_record.sample_id}",
                    existing_windows[0][0] if existing_windows else 0.0,
                )
                default_end = st.session_state.get(
                    f"end_{current_record.sample_id}",
                    existing_windows[0][1] if existing_windows else min(current_record.duration_seconds, 10.0),
                )

            c_t1, c_t2 = st.columns(2)
            with c_t1:
                start_val = st.number_input(
                    "START (seconds):",
                    min_value=0.0,
                    max_value=max(0.0, current_record.duration_seconds),
                    value=float(default_start) if default_start is not None else None,
                    step=0.1,
                    format="%.1f",
                    disabled=(q_status_choice == QueryStatus.INVALID_VIDEO.value),
                    key=f"num_start_{current_record.sample_id}",
                )
            with c_t2:
                end_val = st.number_input(
                    "END (seconds):",
                    min_value=0.0,
                    max_value=max(0.0, current_record.duration_seconds),
                    value=float(default_end) if default_end is not None else None,
                    step=0.1,
                    format="%.1f",
                    disabled=(q_status_choice == QueryStatus.INVALID_VIDEO.value),
                    key=f"num_end_{current_record.sample_id}",
                )
            if start_val is not None and end_val is not None and q_status_choice != QueryStatus.INVALID_VIDEO.value:
                final_windows = [[round(start_val, 1), round(end_val, 1)]]
            else:
                final_windows = []
        else:
            # Multi-window editing
            st.info("Enter start and end for each discontinuous window segment.")
            mw_count_default = st.session_state.get(f"mw_count_{current_record.sample_id}", max(1, len(existing_windows)))
            num_windows = st.number_input(
                "Number of Discontinuous Windows:",
                min_value=1,
                max_value=5,
                value=int(mw_count_default),
                key=f"num_mw_{current_record.sample_id}",
            )
            final_windows = []
            for w_i in range(int(num_windows)):
                c_mw1, c_mw2 = st.columns(2)
                init_s = st.session_state.get(
                    f"mw_s_{w_i}_{current_record.sample_id}",
                    existing_windows[w_i][0] if w_i < len(existing_windows) else 0.0,
                )
                init_e = st.session_state.get(
                    f"mw_e_{w_i}_{current_record.sample_id}",
                    existing_windows[w_i][1] if w_i < len(existing_windows) else min(current_record.duration_seconds, 5.0),
                )
                with c_mw1:
                    w_s = st.number_input(
                        f"Window {w_i+1} START:",
                        min_value=0.0,
                        max_value=current_record.duration_seconds,
                        value=float(init_s),
                        step=0.1,
                        format="%.1f",
                        key=f"mw_s_{w_i}_{current_record.sample_id}",
                    )
                with c_mw2:
                    w_e = st.number_input(
                        f"Window {w_i+1} END:",
                        min_value=0.0,
                        max_value=current_record.duration_seconds,
                        value=float(init_e),
                        step=0.1,
                        format="%.1f",
                        key=f"mw_e_{w_i}_{current_record.sample_id}",
                    )
                final_windows.append([round(w_s, 1), round(w_e, 1)])

        notes_input = st.text_area(
            "Annotation & Adjudication Notes:",
            value=current_record.annotation_notes,
            help="Note visual cues, query revision rationale, or reason for exclusion.",
            key=f"notes_{current_record.sample_id}",
        )

        st.markdown("---")
        # Action buttons
        c_act1, c_act2, c_act3 = st.columns([1.3, 1.0, 1.2])

        # Button label adapts to mode
        if is_blind_mode:
            save_label = "💾 Save Human Review"
        elif mode == AnnotationMode.AI_ASSISTED_PRIMARY.value:
            save_label = "💾 SAVE HUMAN VERIFIED"
        else:
            save_label = "💾 SAVE & NEXT"

        with c_act1:
            if st.button(save_label, type="primary", use_container_width=True):
                # Retrieve explicit or inferred actions
                q_act = st.session_state.get(f"q_action_{current_record.sample_id}")
                w_act = st.session_state.get(f"w_action_{current_record.sample_id}")

                success = handle_save(
                    record=current_record,
                    records=records,
                    query_status=q_status_choice,
                    final_query=final_query,
                    windows=final_windows,
                    notes=notes_input,
                    mode=mode,
                    is_blind_holdout_sample=is_blind_mode,
                    query_action=q_act,
                    window_action=w_act,
                    ai_candidate=ai_cand if not is_blind_mode else None,
                    advance=True,
                )
                if success:
                    st.rerun()

        with c_act2:
            if st.button("💾 SAVE", use_container_width=True):
                q_act = st.session_state.get(f"q_action_{current_record.sample_id}")
                w_act = st.session_state.get(f"w_action_{current_record.sample_id}")
                success = handle_save(
                    record=current_record,
                    records=records,
                    query_status=q_status_choice,
                    final_query=final_query,
                    windows=final_windows,
                    notes=notes_input,
                    mode=mode,
                    is_blind_holdout_sample=is_blind_mode,
                    query_action=q_act,
                    window_action=w_act,
                    ai_candidate=ai_cand if not is_blind_mode else None,
                    advance=False,
                )
                if success:
                    st.success("Annotation saved!")

        with c_act3:
            if st.button("🚫 EXCLUDE VIDEO", use_container_width=True):
                handle_exclude(current_record, records, notes_input, mode)
                st.warning(f"Sample {current_record.sample_id} marked as EXCLUDED.")
                if curr_list_idx < len(filtered_records) - 1:
                    st.session_state["current_index"] += 1
                st.rerun()

    # Navigation Footer
    st.markdown("---")
    col_nav1, col_nav2, col_nav3 = st.columns([1, 2, 1])
    with col_nav1:
        if st.button("⬅ Previous Sample", disabled=(curr_list_idx == 0), use_container_width=True):
            st.session_state["current_index"] = max(0, curr_list_idx - 1)
            st.rerun()

    with col_nav2:
        st.write(f"<center>Sample <b>{curr_list_idx + 1}</b> of <b>{len(filtered_records)}</b></center>", unsafe_allow_html=True)

    with col_nav3:
        if st.button("Next Sample ➡", disabled=(curr_list_idx >= len(filtered_records) - 1), use_container_width=True):
            st.session_state["current_index"] = min(len(filtered_records) - 1, curr_list_idx + 1)
            st.rerun()


def handle_save(
    record: AnnotationRecord,
    records: list[AnnotationRecord],
    query_status: str,
    final_query: str,
    windows: list[list[float]],
    notes: str,
    mode: str,
    is_blind_holdout_sample: bool = False,
    query_action: str | None = None,
    window_action: str | None = None,
    ai_candidate: AIPreannotationRecord | None = None,
    advance: bool = False,
) -> bool:
    """Validate and persist human annotation with audit logging."""
    # 1. Validate query
    q_ok, q_err = validate_query(final_query, query_status)
    if not q_ok:
        st.error(f"Query Error: {q_err}")
        return False

    # 2. Validate temporal boundaries
    if query_status != QueryStatus.INVALID_VIDEO.value:
        if len(windows) == 1:
            w_ok, w_err = validate_temporal_boundaries(windows[0][0], windows[0][1], record.duration_seconds)
        else:
            w_ok, w_err = validate_multi_windows(windows, record.duration_seconds)
        if not w_ok:
            st.error(f"Temporal Window Error: {w_err}")
            return False

    # 3. Update record values
    prev_status = record.review_status
    record.query = final_query.strip()
    record.annotation_notes = notes.strip()
    record.review_status = ReviewStatus.REVIEWED.value if query_status != QueryStatus.INVALID_VIDEO.value else ReviewStatus.EXCLUDED.value

    # Assign Annotator ID and Annotation Mode Provenance
    if mode == AnnotationMode.SECONDARY.value:
        record.annotator_id = "human_02"
        annotation_mode_tag = AnnotationMode.SECONDARY.value
    elif mode == AnnotationMode.ADJUDICATION.value:
        record.annotator_id = "human_adjudicator"
        record.review_status = ReviewStatus.ADJUDICATED.value
        annotation_mode_tag = AnnotationMode.ADJUDICATION.value
    elif is_blind_holdout_sample:
        record.annotator_id = "human_01"
        annotation_mode_tag = "BLIND_HUMAN"
    else:
        record.annotator_id = "human_01"
        annotation_mode_tag = "AI_ASSISTED_HUMAN_VERIFIED"

    # Infer provenance actions if in AI-assisted mode and not explicitly clicked
    if annotation_mode_tag == "AI_ASSISTED_HUMAN_VERIFIED" and ai_candidate:
        if query_action is None:
            ai_q = (ai_candidate.ai_proposed_query or ai_candidate.original_query).strip().lower()
            if record.query.strip().lower() == ai_q:
                query_action = QueryAction.ACCEPTED_AI.value
            else:
                query_action = QueryAction.EDITED_AI.value
        if window_action is None:
            if windows == ai_candidate.get_windows():
                window_action = WindowAction.ACCEPTED_AI.value
            else:
                window_action = WindowAction.EDITED_AI.value
    else:
        query_action = None
        window_action = None

    if query_status == QueryStatus.INVALID_VIDEO.value:
        record.gt_start_seconds = None
        record.gt_end_seconds = None
        record.multi_windows = []
    else:
        record.gt_start_seconds = windows[0][0]
        record.gt_end_seconds = windows[0][1]
        if len(windows) > 1:
            record.multi_windows = [TemporalWindow.from_list(w) for w in windows]
            save_multi_window(record.sample_id, windows)

    # 4. Save to storage
    if mode in (AnnotationMode.PRIMARY.value, AnnotationMode.AI_ASSISTED_PRIMARY.value, AnnotationMode.BLIND_PRIMARY.value):
        save_primary_records(records)
    elif mode == AnnotationMode.SECONDARY.value:
        save_secondary_records(records)
    else:
        save_adjudicated_record(record)
        save_primary_records(records)

    # 5. Append audit log
    log_entry = ReviewLogEntry(
        sample_id=record.sample_id,
        video_filename=record.video_filename,
        original_query=record.original_query or record.query,
        final_query=record.query,
        query_status=query_status,
        windows=windows,
        annotator_id=record.annotator_id,
        notes=notes,
        reviewed_at_utc=datetime.now(timezone.utc).isoformat(),
        previous_review_status=prev_status,
        annotation_mode=annotation_mode_tag,
        query_action=query_action,
        window_action=window_action,
        ai_query_status=ai_candidate.ai_query_status if ai_candidate else None,
        ai_proposed_query=ai_candidate.ai_proposed_query if ai_candidate else None,
        ai_candidate_windows=ai_candidate.get_windows() if ai_candidate else None,
    )
    append_review_audit_log(log_entry)

    if advance:
        st.session_state["current_index"] += 1

    return True


def handle_exclude(
    record: AnnotationRecord,
    records: list[AnnotationRecord],
    notes: str,
    mode: str,
) -> None:
    """Mark video sample as excluded."""
    prev_status = record.review_status
    record.review_status = ReviewStatus.EXCLUDED.value
    record.gt_start_seconds = None
    record.gt_end_seconds = None
    record.multi_windows = []
    record.annotation_notes = f"[EXCLUDED] {notes}".strip()

    if mode in (AnnotationMode.PRIMARY.value, AnnotationMode.AI_ASSISTED_PRIMARY.value):
        save_primary_records(records)
    elif mode == AnnotationMode.SECONDARY.value:
        save_secondary_records(records)

    log_entry = ReviewLogEntry(
        sample_id=record.sample_id,
        video_filename=record.video_filename,
        original_query=record.original_query or record.query,
        final_query=record.query,
        query_status=QueryStatus.INVALID_VIDEO.value,
        windows=[],
        annotator_id=record.annotator_id,
        notes=f"Sample marked EXCLUDED: {notes}",
        reviewed_at_utc=datetime.now(timezone.utc).isoformat(),
        previous_review_status=prev_status,
        annotation_mode="EXCLUDED",
    )
    append_review_audit_log(log_entry)


if __name__ == "__main__":
    main()
