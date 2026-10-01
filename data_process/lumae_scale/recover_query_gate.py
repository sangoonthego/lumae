"""Audited recovery of a source rejected only by the former subject-word gate."""
from .models import WORK, atomic_json, read_json
from .freeze import ACCEPTED
from .source_pool import SELECTION, selection
from .pipeline import _ledger
from .quality_gate import query_errors


def main():
    vid = "29d9a376d250e42cb236f6c42fb6e3ba"
    state = selection()
    row = next(r for r in state["candidates"] if r["source_video_id"] == vid)
    assert row["status"] == "rejected" and row["rejection_reason"] == "all query candidates failed"
    assert not (ACCEPTED / (vid + ".json")).exists()
    import json
    history = [json.loads(line) for line in (WORK / "build_ledger.jsonl").read_text().splitlines()]
    rejection = [r for r in history if r.get("source_video_id") == vid and r["event"] == "rejected"][-1]
    assert len(rejection["failures"]) == 1
    assert rejection["failures"][0]["errors"] == ["missing concrete subject phrase"]
    path = WORK / "visual_index" / vid / "visual_events.json"
    events = read_json(path)
    assert events["inspection_method"] == "agent_viewed_overview"
    events["events"][0]["query"] = "Two workers in white protective suits spray rows of blue stadium seats."
    assert not query_errors(events["events"][0]["query"])
    atomic_json(WORK / "query_gate_recovery_snapshot.json", {"candidate": row.copy(), "rejection": rejection})
    atomic_json(path, events)
    row.update(status="annotation_candidate_ready", rejection_reason=None)
    atomic_json(SELECTION, state)
    _ledger("administrative_query_gate_recovery", vid, reason="False rejection from determiner-only subject check; original rejection retained", prior_status="rejected", restored_status="annotation_candidate_ready")


if __name__ == "__main__":
    main()
