"""One-screen human query -> frozen-v3 advice -> human temporal review."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from data_process.annotation import assisted_deployment as deployment
from data_process.annotation.fastlane import (
    current_lock, ensure_prediction, progress, revise_query_and_predict,
    submit_query_and_predict, submit_temporal,
)
from data_process.annotation.fastlane_freeze import freeze_if_complete
from data_process.annotation.human_query_review import decision_of, worklist
from data_process.annotation.query_sanitation import load_query_reviews
from data_process.annotation.storage import get_videos_dir


def _next_unresolved(root: Path, ids: list[str]) -> str:
    statuses = progress(root)["statuses"]
    return next((sid for sid in ids if statuses[sid] not in {"TEMPORAL_REVIEWED", "EXCLUDED"}), ids[-1])


def _move_to(root: Path, ids: list[str], *, stay: bool = False, current: str = "") -> None:
    st.session_state["a26r_sample"] = current if stay else _next_unresolved(root, ids)
    st.session_state["a26r_nav_override"] = True
    st.rerun()


def render_fastlane(records: list, root: Path) -> None:
    state = progress(root)
    cohort = worklist(root)
    ids = sorted((r["sample_id"] for r in cohort),
                 key=lambda sid: (state["statuses"][sid] in {"TEMPORAL_REVIEWED", "EXCLUDED"}, sid))
    if not ids:
        st.warning("No fast-lane samples found.")
        return
    by_id = {r.sample_id: r for r in records}
    target = {r["sample_id"]: r for r in cohort}
    if st.session_state.get("a26r_sample") not in ids:
        st.session_state["a26r_sample"] = _next_unresolved(root, ids)
    sid = st.session_state["a26r_sample"]
    if st.session_state.pop("a26r_nav_override", False) or st.session_state.get("a26r_jump_widget") not in ids:
        st.session_state["a26r_jump_widget"] = sid
    row = target[sid]
    record = by_id[sid]
    query = load_query_reviews(root / "local_data/annotations/lumae_ads/query_reviews.csv").get(sid)
    qdecision = decision_of(query)
    status = state["statuses"][sid]

    st.header("LUMAE Fast-Lane: Human Query and Temporal Review")
    st.caption("One video, two explicit human decisions. The frozen semantic_v3 candidate is advisory until a human confirms its interval.")
    metrics = st.columns(6)
    for col, label, value in zip(metrics,
        ("Pilot", "Pre-fast-lane reviewed", "Query approved", "AI preannotated", "Temporal reviewed", "Remaining"),
        (state["total_pilot"], state["reviewed_before_fastlane"],
         f"{state['query_approved']} / {state['fastlane_total']}",
         f"{state['preannotated']} / {state['fastlane_total']}",
         f"{state['temporal_reviewed']} / {state['fastlane_total']}", state["remaining"])):
        col.metric(label, value)
    st.progress((state["fastlane_total"] - state["remaining"]) / state["fastlane_total"])
    st.caption(f"Excluded: {state['excluded']} · AI windows accepted: {state['temporal_decisions']['ACCEPT_AI']} "
               f"· edited: {state['temporal_decisions']['EDIT_AI']} "
               f"· rejected/manual: {state['temporal_decisions']['REJECT_AI']}")
    if st.session_state.pop("fastlane_saved", False):
        st.success("Human decision saved.")

    st.selectbox("Jump to sample", ids,
                 format_func=lambda item: f"{item} · {state['statuses'][item]}",
                 key="a26r_jump_widget",
                 on_change=lambda: st.session_state.update(a26r_sample=st.session_state["a26r_jump_widget"]))
    index = ids.index(sid)
    prev, nxt = st.columns(2)
    if prev.button("Previous", disabled=index == 0, use_container_width=True):
        st.session_state["a26r_sample"] = ids[index - 1]
        st.session_state["a26r_nav_override"] = True
        st.rerun()
    if nxt.button("Next", disabled=index == len(ids) - 1, use_container_width=True):
        st.session_state["a26r_sample"] = ids[index + 1]
        st.session_state["a26r_nav_override"] = True
        st.rerun()

    st.subheader(f"{sid} · {status}")
    st.caption(f"Duration: {record.duration_seconds:.2f}s · Category: {record.product_category or 'Unknown'} · File: {record.video_filename}")
    video = get_videos_dir() / record.video_filename
    left, right = st.columns([1.1, 1], gap="large")
    with left:
        if video.is_file():
            st.video(str(video))
        else:
            st.error(f"Video unavailable: {video}")
    with right:
        st.markdown("**Original LUMAE query**")
        st.write(row["query"])
        st.caption(f"Original quality (AI advisory): {row['suggestion_original_query_quality']}")
        st.markdown("**AI suggested query — advisory**")
        st.info(row["suggestion_ai_suggested_query"])
        st.caption(f"Change type: {row['suggestion_query_change_type']}")
        st.markdown("**Whole-video visual evidence (AI advisory)**")
        st.write(row["suggestion_visual_evidence"])

        if status in {"TEMPORAL_REVIEWED", "EXCLUDED"}:
            st.success(f"Human workflow complete: {status}")
            st.write(f"Final query: {query.human_final_query if query and query.human_final_query else '(excluded)'}")
        elif video.is_file() and not qdecision:
            st.markdown("#### Human query decision")
            reviewer = st.text_input("Your human reviewer ID", value="", placeholder="human_01",
                                     key="a26r_reviewer")
            seen = st.checkbox("I watched this video", value=False, key=f"a26r_seen_{sid}")
            observable = st.checkbox("The final action is visibly observable", value=False,
                                     key=f"a26r_observable_{sid}")
            localizable = st.checkbox("The final action has a localizable start and end", value=False,
                                      key=f"a26r_localizable_{sid}")
            edited = st.text_area("Edited query (for Edit Query)",
                                  value=row["suggestion_ai_suggested_query"], key=f"a26r_edit_{sid}")
            note = st.text_area("Human review note (required for Exclude)", value="",
                                key=f"a26r_note_{sid}")
            confirm = st.checkbox("Confirm exclusion from the temporal dataset", value=False,
                                  key=f"a26r_exclude_confirm_{sid}")
            actions = (
                ("ACCEPT_SUGGESTION", "Accept Suggestion · Lock & Predict"),
                ("EDIT_QUERY", "Edit Query · Lock & Predict"),
                ("KEEP_ORIGINAL", "Keep Original · Lock & Predict"),
                ("EXCLUDE", "Exclude Sample"),
            )
            for decision, label in actions:
                if st.button(label, key=f"fastlane_query_{decision}_{sid}", use_container_width=True):
                    try:
                        submit_query_and_predict(root, sample_id=sid, decision=decision,
                                                 reviewer_id=reviewer, edited_query=edited, note=note,
                                                 seen_video=seen, observable=observable,
                                                 localizable=localizable, confirm_exclude=confirm)
                    except (ValueError, OSError) as exc:
                        st.error(str(exc))
                    else:
                        st.session_state["fastlane_saved"] = True
                        _move_to(root, ids, stay=(decision != "EXCLUDE"), current=sid)
        elif qdecision == "EXCLUDE":
            st.warning("Human exclusion was saved; completing its primary-record commit.")
            try:
                submit_temporal(root, sid, "EXCLUDE", query.reviewer_id,
                                note=query.review_notes, confirm_exclude=True)
            except (ValueError, OSError) as exc:
                st.error(str(exc))
            else:
                _move_to(root, ids)
        elif video.is_file():
            st.success(f"Human-approved query: {query.human_final_query}")
            st.caption(f"Query reviewer: {query.reviewer_id} · Version: {query.query_version}")
            try:
                prediction = ensure_prediction(root, sid)
            except (ValueError, OSError) as exc:
                st.error(f"Prediction is pending: {exc}")
                if st.button("Retry query lock and semantic_v3", key=f"retry_{sid}"):
                    st.rerun()
                prediction = None
            if prediction:
                lock = current_lock(root, sid)
                st.caption(f"Query SHA256: {lock['query_sha256']}")
                start, end = prediction["pred_start_seconds"], prediction["pred_end_seconds"]
                st.markdown(f"#### AI temporal candidate: {start:.1f}s–{end:.1f}s")
                st.video(str(video), start_time=max(0, int(start) - 2),
                         end_time=min(int(record.duration_seconds), int(end) + 3))
                st.caption("Frame-change proposals ranked by the unchanged semantic_v3 algorithm. Scores are uncalibrated; verify boundaries in the video.")
                with st.expander("Candidate windows and ranking details"):
                    st.json({"candidate_windows": prediction["candidate_windows"],
                             "candidate_scores": prediction["candidate_scores"],
                             "ranking_margin": prediction["ranking_margin"],
                             "semantic_phase_information": prediction["semantic_phase_information"]})
                temporal_reviewer = st.text_input("Your human temporal reviewer ID", value="",
                                                  key="fastlane_temporal_reviewer")
                human_start = st.number_input("Manual start (seconds)", min_value=0.0,
                                              max_value=float(record.duration_seconds),
                                              value=float(start), step=0.1, format="%.1f",
                                              key=f"fastlane_start_{sid}_v{query.query_version}")
                human_end = st.number_input("Manual end (seconds)", min_value=0.0,
                                            max_value=float(record.duration_seconds),
                                            value=float(end), step=0.1, format="%.1f",
                                            key=f"fastlane_end_{sid}_v{query.query_version}")
                temporal_note = st.text_area("Temporal review note / exclusion reason", value="",
                                             key=f"fastlane_temporal_note_{sid}")
                temporal_confirm = st.checkbox("Confirm temporal exclusion", value=False,
                                               key=f"fastlane_temporal_exclude_{sid}")
                temporal_actions = (
                    ("ACCEPT_AI", "Accept AI Window"), ("EDIT_AI", "Edit Window"),
                    ("REJECT_AI", "Reject AI Window · Save Manual Interval"),
                    ("EXCLUDE", "Exclude Sample from Temporal Dataset"),
                )
                for decision, label in temporal_actions:
                    if st.button(label, key=f"fastlane_temporal_{decision}_{sid}",
                                 use_container_width=True):
                        try:
                            submit_temporal(root, sid, decision, temporal_reviewer,
                                            start=None if decision in {"ACCEPT_AI", "EXCLUDE"} else human_start,
                                            end=None if decision in {"ACCEPT_AI", "EXCLUDE"} else human_end,
                                            note=temporal_note, confirm_exclude=temporal_confirm)
                        except (ValueError, OSError) as exc:
                            st.error(str(exc))
                        else:
                            st.session_state["fastlane_saved"] = True
                            _move_to(root, ids)
            else:
                st.markdown("#### No usable AI interval")
                st.caption("You may retry prediction or explicitly exclude this sample with a human reason.")
                fallback_reviewer = st.text_input("Your reviewer ID for exclusion", value="",
                                                  key=f"fastlane_fallback_reviewer_{sid}")
                fallback_reason = st.text_area("Exclusion reason", value="",
                                               key=f"fastlane_fallback_reason_{sid}")
                fallback_confirm = st.checkbox("Confirm sample exclusion", value=False,
                                               key=f"fastlane_fallback_confirm_{sid}")
                if st.button("Exclude Sample · No Usable Interval", key=f"fastlane_fallback_exclude_{sid}"):
                    try:
                        submit_temporal(root, sid, "EXCLUDE", fallback_reviewer,
                                        note=fallback_reason, confirm_exclude=fallback_confirm)
                    except (ValueError, OSError) as exc:
                        st.error(str(exc))
                    else:
                        st.session_state["fastlane_saved"] = True
                        _move_to(root, ids)
            with st.expander("Revise approved query before temporal review"):
                st.warning("Revising creates a new query version and invalidates the current prediction. Old versions remain in the audit history.")
                revision_choice = st.selectbox("Revision decision", (
                    "EDIT_QUERY", "ACCEPT_SUGGESTION", "KEEP_ORIGINAL"),
                    key=f"fastlane_revision_choice_{sid}")
                revised = st.text_area("Revised query", value=query.human_final_query,
                                       key=f"fastlane_revision_text_{sid}_v{query.query_version}")
                revision_seen = st.checkbox("I rechecked this video", value=False,
                                            key=f"fastlane_revision_seen_{sid}_v{query.query_version}")
                revision_observable = st.checkbox("Revised action is visible", value=False,
                                                  key=f"fastlane_revision_obs_{sid}_v{query.query_version}")
                revision_localizable = st.checkbox("Revised action is localizable", value=False,
                                                   key=f"fastlane_revision_loc_{sid}_v{query.query_version}")
                revision_reviewer = st.text_input("Your reviewer ID for revision", value="",
                                                  key=f"fastlane_revision_reviewer_{sid}_v{query.query_version}")
                revision_note = st.text_area("Revision reason", value="",
                                             key=f"fastlane_revision_note_{sid}_v{query.query_version}")
                if st.button("Revise Query · Regenerate Prediction", key=f"fastlane_revise_{sid}"):
                    try:
                        revise_query_and_predict(root, sample_id=sid, decision=revision_choice,
                                                 reviewer_id=revision_reviewer,
                                                 edited_query=revised, note=revision_note,
                                                 seen_video=revision_seen,
                                                 observable=revision_observable,
                                                 localizable=revision_localizable)
                    except (ValueError, OSError) as exc:
                        st.error(str(exc))
                    else:
                        st.rerun()

    if state["remaining"] == 0 and all(
        row["review_status"] in {"REVIEWED", "EXCLUDED"}
        for row in deployment._read_csv(root / "local_data/annotations/lumae_ads/human_primary.csv")
    ):
        try:
            frozen = freeze_if_complete(root)
        except (ValueError, OSError) as exc:
            st.error(f"Dataset freeze integrity gate failed: {exc}")
        else:
            if frozen:
                st.success(f"Dataset frozen: {frozen['dataset_dir']} · SHA256 {frozen['dataset_sha256']}")
