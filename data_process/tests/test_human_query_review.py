"""A.2.6R decisions operate on disposable artifact copies only."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import shutil

import pytest

from data_process.annotation.human_query_review import (
    _paths, decision_of, freeze_if_complete, progress, submit_decision, worklist,
)
from data_process.annotation.query_sanitation import (
    CANONICAL_QUERY_REVIEW_FIELDS, load_query_reviews, validate_query_review_schema,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def review_root(tmp_path):
    source = _paths(REPO_ROOT)
    target = _paths(tmp_path)
    for src, dst in zip(source[:3], target[:3]):
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    return tmp_path


def _click(root, sid, decision, **kwargs):
    return submit_decision(root, sample_id=sid, decision=decision,
                           reviewer_id="human_tester", seen_video=True,
                           observable=True, localizable=True, **kwargs)


def test_no_approval_without_click_and_exclude_requires_confirmation(review_root):
    root = review_root
    _, _, reviews_path, manifest = _paths(root)
    before = reviews_path.read_bytes()
    assert progress(root)["reviewed"] == 0
    assert freeze_if_complete(root) is None
    assert not manifest.exists()
    assert reviews_path.read_bytes() == before
    sid = worklist(root)[0]["sample_id"]
    with pytest.raises(ValueError, match="watched"):
        submit_decision(root, sample_id=sid, decision="ACCEPT_SUGGESTION",
                        reviewer_id="human_tester", observable=True, localizable=True)
    with pytest.raises(ValueError, match="confirmation"):
        _click(root, sid, "EXCLUDE", note="Video irrelevant")
    assert reviews_path.read_bytes() == before


def test_explicit_decisions_resume_and_preserve_other_rows_and_gt(review_root):
    root = review_root
    primary_path, _, reviews_path, manifest = _paths(root)
    primary_sha = hashlib.sha256(primary_path.read_bytes()).hexdigest()
    original = reviews_path.read_text(encoding="utf-8")
    reviewed_line = next(line for line in original.splitlines(keepends=True)
                         if '"lumae_ads_pilot_0028"' in line)
    ids = [r["sample_id"] for r in worklist(root)]
    accepted = _click(root, ids[0], "ACCEPT_SUGGESTION")
    assert accepted.query_review_status == "HUMAN_EDITED"
    assert accepted.human_final_query == worklist(root)[0]["suggestion_ai_suggested_query"]
    assert accepted.temporal_annotation_locked is False
    assert accepted.query_version == "2"
    edited = _click(root, ids[1], "EDIT_QUERY", edited_query="A person lifts a visible bottle from a table.")
    assert edited.human_final_query == "A person lifts a visible bottle from a table."
    kept = _click(root, ids[2], "KEEP_ORIGINAL")
    assert kept.query_review_status == "VALID_AS_IS"
    assert kept.human_final_query == kept.original_query
    excluded = _click(root, ids[3], "EXCLUDE", note="No useful event", confirm_exclude=True)
    assert excluded.query_review_status == "EXCLUDE" and excluded.temporal_annotation_locked is True
    state = progress(root)
    assert state["reviewed"] == 4 and state["remaining"] == 26
    assert state["items"][0]["sample_id"] == ids[4]
    assert state["counts"] == {"ACCEPT_SUGGESTION": 1, "EDIT_QUERY": 1,
                               "KEEP_ORIGINAL": 1, "EXCLUDE": 1}
    assert reviewed_line in reviews_path.read_text(encoding="utf-8")
    assert hashlib.sha256(primary_path.read_bytes()).hexdigest() == primary_sha
    assert not manifest.exists()
    validate_query_review_schema(reviews_path)
    with reviews_path.open("r", encoding="utf-8-sig", newline="") as f:
        assert list(csv.DictReader(f))[0].keys() == dict.fromkeys(CANONICAL_QUERY_REVIEW_FIELDS).keys()
    with pytest.raises(ValueError, match="already"):
        _click(root, ids[0], "ACCEPT_SUGGESTION")


@pytest.mark.parametrize("bad_query", ["", "00:12 A person lifts a bottle.",
                                           "A patented bottle guarantees better health.", "bottle"])
def test_invalid_final_query_does_not_save(review_root, bad_query):
    root = review_root
    path = _paths(root)[2]
    before = path.read_bytes()
    sid = worklist(root)[0]["sample_id"]
    with pytest.raises(ValueError):
        _click(root, sid, "EDIT_QUERY", edited_query=bad_query)
    assert path.read_bytes() == before


def test_freeze_only_after_every_explicit_decision(review_root):
    root = review_root
    items = worklist(root)
    excluded_id = items[-1]["sample_id"]
    for item in items[:-1]:
        _click(root, item["sample_id"], "ACCEPT_SUGGESTION")
    assert freeze_if_complete(root) is None
    _click(root, excluded_id, "EXCLUDE", note="Human excluded this sample", confirm_exclude=True)
    path, sha = freeze_if_complete(root)
    assert sha == hashlib.sha256(path.read_bytes()).hexdigest()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["total_reviewed"] == 30
    assert data["decision_counts"]["EXCLUDE"] == 1
    assert len(data["approved_queries"]) == 29
    assert excluded_id not in {r["sample_id"] for r in data["approved_queries"]}
    assert all(r["human_final_query"] and r["reviewer_id"].startswith("human_")
               and r["reviewed_at_utc"] for r in data["approved_queries"])
    assert all(decision_of(r) for sid, r in load_query_reviews(_paths(root)[2]).items()
               if sid in {item["sample_id"] for item in items})
