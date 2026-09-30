"""Streamlit controls for explicit human review of semantic_v3 deployment output."""

from __future__ import annotations

import json

import streamlit as st

from data_process.annotation.assisted_deployment import (
    ROOT, _paths, _read_csv, approved_query, load_state, precompute,
    submit_temporal_decision,
)


def render_assisted_temporal_review(record, position: int, total: int) -> None:
    state = load_state()
    sid = record.sample_id
    query_review = state["queries"].get(sid)
    st.header(f"Assisted temporal review — {sid}")
    st.caption(f"Deployment sample {position + 1} of {total}. Query approval precedes semantic_v3 inference and temporal review.")
    video = _paths(ROOT)[2] / record.video_filename
    left, right = st.columns([1.1, 1.0], gap="large")
    with left:
        st.video(str(video))
        st.caption(f"Duration: {record.duration_seconds:.2f}s")
    with right:
        if not approved_query(query_review):
            st.warning("Human query approval is required. Use QUERY_SANITATION first.")
            st.write(f"Original query: {record.query}")
            return
        st.markdown("**Human-verified query**")
        st.info(query_review["human_final_query"])
        st.caption(f"Query reviewer: {query_review['reviewer_id']} • Version: {query_review['query_version']}")
        pre_path = _paths(ROOT)[0] / "semantic_v3_deployment_preannotations.csv"
        pre = next((r for r in _read_csv(pre_path) if r["sample_id"] == sid), None)
        if pre is None:
            st.warning("No semantic_v3 candidate is ready. Video-inspected candidate events must be provided before precomputation.")
            if st.button("Precompute approved queries", key=f"precompute_{sid}"):
                try:
                    result = precompute()
                    st.info(result)
                except Exception as exc:
                    st.error(str(exc))
            return
        if pre["verified_query"] != query_review["human_final_query"] or pre["query_version"] != query_review["query_version"]:
            st.error("Preannotation is stale because the verified query changed. Recompute it before review.")
            return
        selected = json.loads(pre["selected_candidate_window"])
        st.markdown(f"**Selected AI candidate:** {selected[0]:.1f}s – {selected[1]:.1f}s")
        st.caption(f"Confidence: {pre['semantic_confidence']} • Ranking margin: {pre['ranking_margin']} • Rank: {pre['selected_candidate_rank']}")
        with st.expander("Candidate windows and semantic reason"):
            st.json(json.loads(pre["candidate_windows_json"]))
            st.write(pre["semantic_reason"])
        reviewer = st.text_input("Human temporal reviewer ID", value="human_01", key=f"temporal_reviewer_{sid}")
        start = st.number_input("Human start (seconds)", min_value=0.0, max_value=float(record.duration_seconds), value=float(selected[0]), step=0.1, format="%.1f", key=f"assist_start_{sid}")
        end = st.number_input("Human end (seconds)", min_value=0.0, max_value=float(record.duration_seconds), value=float(selected[1]), step=0.1, format="%.1f", key=f"assist_end_{sid}")
        notes = st.text_area("Review notes", key=f"assist_notes_{sid}")
        st.caption("Accept keeps the AI boundaries. Edit changes the boundaries. Reject records a manually supplied interval for the correct event.")
        buttons = st.columns(4)
        choices = ("ACCEPT_AI", "EDIT_AI", "REJECT_AI", "EXCLUDE")
        for column, choice in zip(buttons, choices):
            with column:
                if st.button(choice.replace("_", " "), key=f"assist_{choice}_{sid}", use_container_width=True):
                    try:
                        submit_temporal_decision(sid, choice, reviewer.strip(),
                                                 None if choice == "ACCEPT_AI" else start,
                                                 None if choice == "ACCEPT_AI" else end,
                                                 notes, root=ROOT)
                    except Exception as exc:
                        st.error(str(exc))
                    else:
                        st.success(f"Human decision saved: {choice}")
                        st.rerun()
