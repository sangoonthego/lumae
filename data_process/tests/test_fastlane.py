"""Fast-lane tests use synthetic human inputs and a disposable local video."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import shutil

import cv2
import numpy as np
import pytest

from data_process.annotation import assisted_deployment as deployment
from data_process.annotation import fastlane as fastlane_module
from data_process.annotation.fastlane import (
    current_lock, current_prediction, ensure_prediction, progress,
    recover_incomplete_commits, revise_query_and_predict,
    submit_query_and_predict, submit_temporal,
)
from data_process.annotation.fastlane_freeze import (
    DEVELOPMENT_EXPOSED, FORCED_TRAIN, _split, freeze_dataset,
)
from data_process.annotation.models import QueryReviewRecord
from data_process.annotation.human_query_review import submit_decision
from data_process.annotation.query_sanitation import atomic_write_query_reviews


REPO_ROOT = Path(__file__).resolve().parents[2]
SID = "lumae_ads_pilot_0001"


def _csv(path: Path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def fastlane_root(tmp_path):
    root = tmp_path
    ann = root / "local_data/annotations/lumae_ads"
    videos = root / "local_data/raw/adsqa/videos"
    videos.mkdir(parents=True)
    root_manifest = root / "local_data/manifests/semantic_v3_algorithm_freeze.json"
    root_manifest.parent.mkdir(parents=True)
    shutil.copy2(REPO_ROOT / "local_data/manifests/semantic_v3_algorithm_freeze.json", root_manifest)
    frozen = json.loads(root_manifest.read_text(encoding="utf-8"))
    for name in frozen["source_file_hashes"]:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO_ROOT / name, path)
    with (REPO_ROOT / "local_data/annotations/lumae_ads/human_primary.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as stream:
        fields = next(csv.reader(stream))
    rows = []
    for i in range(1, 49):
        sid = f"lumae_ads_pilot_{i:04d}"
        stem = f"vid_{i:04d}"
        rows.append({
            "sample_id": sid, "video_filename": stem + ".mp4",
            "vid": "lumae_ads_" + stem, "source_dataset": "AdsQA",
            "source_video_id": stem, "source_split": "test", "source_url": "",
            "product_category": "Home", "query": "A person lifts a bottle from a table.",
            "duration_seconds": "8.0", "gt_start_seconds": "" if i == 1 else "1.0",
            "gt_end_seconds": "" if i == 1 else "3.0",
            "annotator_id": "ann_lead" if i == 1 else "human_01",
            "annotation_notes": "" if i == 1 else "Human verified historical annotation",
            "review_status": "DRAFT" if i == 1 else "REVIEWED",
        })
        if i != 1:
            (videos / f"{stem}.mp4").write_bytes(b"historical-video-placeholder")
    _csv(ann / "human_primary.csv", fields, rows)
    _csv(ann / "ai_query_suggestions.csv", deployment.SUGGESTION_FIELDS, [{
        "sample_id": SID, "video_filename": "vid_0001.mp4",
        "original_query": "A person lifts a bottle from a table.",
        "original_query_quality": "VAGUE_BUT_RELEVANT",
        "ai_suggested_query": "A person lifts a visible bottle from a table.",
        "query_change_type": "NARROWED_INTENT", "query_localizable": "TRUE",
        "query_observable": "TRUE", "visual_evidence": "A moving object is visible.",
        "inspection_method": "test video inspection", "generated_at_utc": "2026-09-30T00:00:00Z",
        "status": "AI_SUGGESTED", "sampled_times_seconds": "[1,2,3]",
        "query_quality_suggestion": "VAGUE_BUT_RELEVANT",
        "observable_evidence": "A moving object is visible.",
    }])
    atomic_write_query_reviews(ann / "query_reviews.csv", [QueryReviewRecord(
        sample_id=SID, video_filename="vid_0001.mp4", source_dataset="AdsQA",
        source_video_id="vid_0001", original_query=rows[0]["query"],
        query_version="1", temporal_annotation_locked=True,
    )])
    _csv(ann / "assisted_annotation_audit.csv", deployment.AUDIT_FIELDS, [])
    _csv(ann / "semantic_v3_deployment_preannotations.csv", deployment.PREANNOTATION_FIELDS, [])
    out = cv2.VideoWriter(str(videos / "vid_0001.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (64, 48))
    assert out.isOpened()
    for frame_no in range(40):
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        cv2.rectangle(frame, (frame_no % 40, 8), (frame_no % 40 + 12, 30), (255, 255, 255), -1)
        out.write(frame)
    out.release()
    return root


def _approve(root):
    return submit_query_and_predict(root, sample_id=SID, decision="ACCEPT_SUGGESTION",
                                    reviewer_id="human_test", seen_video=True,
                                    observable=True, localizable=True)


def test_lock_precedes_prediction_and_ai_never_writes_gt(fastlane_root, monkeypatch):
    root = fastlane_root
    primary = root / "local_data/annotations/lumae_ads/human_primary.csv"
    before = primary.read_bytes()
    with pytest.raises(ValueError, match="human query decision"):
        ensure_prediction(root, SID)
    assert primary.read_bytes() == before
    ranker = fastlane_module.rank_and_select_candidates
    def assert_locked_before_ranker(decomposition, candidates):
        assert current_lock(root, SID) is not None
        return ranker(decomposition, candidates)
    monkeypatch.setattr(fastlane_module, "rank_and_select_candidates", assert_locked_before_ranker)
    pred = _approve(root)
    lock = current_lock(root, SID)
    assert lock["query_sha256"] == hashlib.sha256(lock["human_final_query"].encode()).hexdigest()
    assert pred["query_lock_sha256"]
    assert pred["query_sha256"] == lock["query_sha256"]
    assert pred["semantic_v3_algorithm_sha256"] == deployment.ALGORITHM_SHA256
    assert pred["status"] == "AI_PREANNOTATION"
    assert pred["candidate_windows"] and pred["candidate_scores"]
    assert pred["video_sha256"] == hashlib.sha256(
        (root / "local_data/raw/adsqa/videos/vid_0001.mp4").read_bytes()).hexdigest()
    assert primary.read_bytes() == before
    assert progress(root)["statuses"][SID] == "AI_PREANNOTATED"
    assert not (root / "local_data/datasets/lumae_ads_v1").exists()


def test_query_revision_invalidates_old_prediction_and_preserves_versions(fastlane_root):
    root = fastlane_root
    first = _approve(root)
    lock1 = current_lock(root, SID)
    second = revise_query_and_predict(root, sample_id=SID, decision="EDIT_QUERY",
        reviewer_id="human_test", edited_query="A person moves a visible bottle across the table.",
        note="Action occurs later", seen_video=True, observable=True, localizable=True)
    assert second["query_version"] == str(int(first["query_version"]) + 1)
    assert second["query_sha256"] != first["query_sha256"]
    assert current_prediction(root, SID)["query_sha256"] == second["query_sha256"]
    lock_dir = root / "local_data/manifests/a26_fastlane_query_locks"
    pre_dir = root / "local_data/manifests/a26_fastlane_preannotations"
    assert len(list(lock_dir.glob("*.json"))) == 2
    assert len(list(pre_dir.glob("*.preannotation.json"))) == 2
    tombstone = json.loads((pre_dir / f"{SID}.v{lock1['query_version']}.superseded.json").read_text())
    assert tombstone["old_prediction_valid_for_current_query"] is False
    assert json.loads((lock_dir / f"{SID}.v{lock1['query_version']}.query_lock.json").read_text()) == lock1
    pre_csv = root / "local_data/annotations/lumae_ads/semantic_v3_deployment_preannotations.csv"
    rows = deployment._read_csv(pre_csv)
    rows[0]["query_version"] = first["query_version"]
    deployment._atomic_csv(pre_csv, deployment.PREANNOTATION_FIELDS, rows)
    with pytest.raises(ValueError, match="stale"):
        submit_temporal(root, SID, "ACCEPT_AI", "human_test")


def test_invalid_intervals_exclusion_confirmation_and_freeze_gate(fastlane_root):
    root = fastlane_root
    _approve(root)
    with pytest.raises(ValueError, match="Manual interval"):
        submit_temporal(root, SID, "EDIT_AI", "human_test", start=-1, end=3)
    with pytest.raises(ValueError, match="confirmation"):
        submit_temporal(root, SID, "EXCLUDE", "human_test", note="No event")
    assert not (root / "local_data/datasets/lumae_ads_v1").exists()
    with pytest.raises(ValueError, match="unresolved"):
        freeze_dataset(root)


def test_explicit_query_exclusion_omitted_from_all_training_manifests(fastlane_root):
    root = fastlane_root
    submit_query_and_predict(root, sample_id=SID, decision="EXCLUDE",
        reviewer_id="human_test", seen_video=True, note="No localizable visible event",
        confirm_exclude=True)
    primary = deployment._read_csv(root / "local_data/annotations/lumae_ads/human_primary.csv")
    assert primary[0]["review_status"] == "EXCLUDED"
    assert primary[0]["gt_start_seconds"] == primary[0]["gt_end_seconds"] == ""
    directory = root / "local_data/datasets/lumae_ads_v1"
    assert directory.is_dir()
    for name in ("all", "train", "val", "test"):
        rows = [json.loads(line) for line in (directory / f"lumae_ads_v1_{name}.jsonl").read_text().splitlines()]
        assert SID not in {row["qid"] for row in rows}
    manifest = json.loads((directory / "dataset_manifest.json").read_text())
    assert manifest["included_count"] == 47 and manifest["excluded_count"] == 1


def test_human_can_exclude_when_no_preannotation_is_available(fastlane_root):
    root = fastlane_root
    submit_decision(root, sample_id=SID, decision="ACCEPT_SUGGESTION",
                    reviewer_id="human_test", seen_video=True,
                    observable=True, localizable=True, freeze_batch=False)
    assert current_prediction(root, SID) is None
    with pytest.raises(ValueError, match="confirmation"):
        submit_temporal(root, SID, "EXCLUDE", "human_test", note="Cannot define an interval")
    submit_temporal(root, SID, "EXCLUDE", "human_test",
                    note="Cannot define an interval", confirm_exclude=True)
    assert deployment._read_csv(root / "local_data/annotations/lumae_ads/human_primary.csv")[0]["review_status"] == "EXCLUDED"


def test_legacy_deployment_paths_cannot_bypass_per_sample_lock(fastlane_root):
    root = fastlane_root
    submit_decision(root, sample_id=SID, decision="ACCEPT_SUGGESTION",
                    reviewer_id="human_test", seen_video=True,
                    observable=True, localizable=True, freeze_batch=False)
    with pytest.raises(ValueError, match="per-sample video-derived inference"):
        deployment.precompute(root)
    with pytest.raises(ValueError, match="per-sample query lock"):
        deployment.submit_temporal_decision(SID, "ACCEPT_AI", "human_test", root=root)
    assert deployment._read_csv(root / "local_data/annotations/lumae_ads/human_primary.csv")[0]["review_status"] == "DRAFT"


def test_split_is_deterministic_and_refuses_exposed_validation_test():
    samples = [{"qid": f"lumae_ads_pilot_{i:04d}", "vid": f"video_{i:04d}"}
               for i in range(1, 49)]
    first, manifest1 = _split(samples)
    second, manifest2 = _split(samples)
    assert first == second and manifest1 == manifest2
    assert manifest1["seed"] == 20260930
    assert not FORCED_TRAIN.intersection(manifest1["val_ids"] + manifest1["test_ids"])
    duplicate_video = samples + [{"qid": "another_query_same_video", "vid": "video_0001"}]
    _, grouped = _split(duplicate_video)
    group_names = [name for name in ("train", "val", "test")
                   if "lumae_ads_pilot_0001" in grouped[f"{name}_ids"]]
    assert len(group_names) == 1
    assert "another_query_same_video" in grouped[f"{group_names[0]}_ids"]
    only_exposed = [sample for sample in samples if sample["qid"] in FORCED_TRAIN]
    with pytest.raises(ValueError, match="Insufficient clean videos"):
        _split(only_exposed)


def test_freeze_rejects_missing_human_provenance(fastlane_root):
    root = fastlane_root
    _approve(root)
    rows = deployment._read_csv(root / "local_data/annotations/lumae_ads/human_primary.csv")
    rows[0].update(review_status="REVIEWED", gt_start_seconds="1.0", gt_end_seconds="2.0",
                   annotator_id="ann_lead")
    deployment._atomic_csv(root / "local_data/annotations/lumae_ads/human_primary.csv",
                           list(rows[0]), rows)
    with pytest.raises(ValueError, match="human temporal reviewer"):
        freeze_dataset(root)


def test_human_temporal_click_auto_freezes_and_seals_split(fastlane_root):
    root = fastlane_root
    pred = _approve(root)
    submit_temporal(root, SID, "ACCEPT_AI", "human_test")
    primary = deployment._unique(deployment._read_csv(
        root / "local_data/annotations/lumae_ads/human_primary.csv"), "primary")
    assert primary[SID]["review_status"] == "REVIEWED"
    assert float(primary[SID]["gt_start_seconds"]) == pred["pred_start_seconds"]
    assert "Human verified AI temporal preannotation" in primary[SID]["annotation_notes"]
    directory = root / "local_data/datasets/lumae_ads_v1"
    assert directory.is_dir()
    rows = [json.loads(line) for line in (directory / "lumae_ads_v1_all.jsonl").read_text().splitlines()]
    assert len(rows) == 48
    assert all(row["relevant_clip_ids"] is None and row["saliency_scores"] is None for row in rows)
    split = json.loads((directory / "split_manifest.json").read_text())
    ids = {name: set(split[f"{name}_ids"]) for name in ("train", "val", "test")}
    vids = {name: set(split[f"{name}_vids"]) for name in ("train", "val", "test")}
    assert not (vids["train"] & vids["val"] or vids["train"] & vids["test"] or vids["val"] & vids["test"])
    assert not (FORCED_TRAIN & (ids["val"] | ids["test"]))
    assert DEVELOPMENT_EXPOSED & ids["train"]
    assert len(ids["train"] | ids["val"] | ids["test"]) == 48
    handoff = json.loads((root / "local_data/manifests/lumae_ads_v1_training_handoff.json").read_text())
    for name in ("train", "val", "test"):
        item = handoff[f"{name}_manifest"]
        assert hashlib.sha256((root / item["path"]).read_bytes()).hexdigest() == item["sha256"]
    assert freeze_dataset(root)["reused_existing"] is True
    primary["lumae_ads_pilot_0003"]["annotation_notes"] += " changed"
    deployment._atomic_csv(root / "local_data/annotations/lumae_ads/human_primary.csv",
                           list(next(iter(primary.values()))), list(primary.values()))
    with pytest.raises(ValueError, match="different source bytes"):
        freeze_dataset(root)


def test_frozen_source_video_hashes_detect_later_mutation(fastlane_root):
    root = fastlane_root
    _approve(root)
    submit_temporal(root, SID, "ACCEPT_AI", "human_test")
    video = root / "local_data/raw/adsqa/videos/vid_0002.mp4"
    video.write_bytes(b"changed source video")
    with pytest.raises(ValueError, match="source video changed"):
        freeze_dataset(root)


def test_interrupted_human_commit_recovers_from_durable_audit(fastlane_root, monkeypatch):
    root = fastlane_root
    _approve(root)
    original = deployment._atomic_csv
    primary_path = root / "local_data/annotations/lumae_ads/human_primary.csv"
    def fail_primary(path, fields, rows):
        if path == primary_path:
            raise OSError("simulated interruption")
        return original(path, fields, rows)
    monkeypatch.setattr(deployment, "_atomic_csv", fail_primary)
    with pytest.raises(OSError, match="simulated interruption"):
        submit_temporal(root, SID, "ACCEPT_AI", "human_test")
    monkeypatch.setattr(deployment, "_atomic_csv", original)
    assert deployment._read_csv(primary_path)[0]["review_status"] == "DRAFT"
    assert len(deployment._read_csv(root / "local_data/annotations/lumae_ads/assisted_annotation_audit.csv")) == 1
    assert recover_incomplete_commits(root) == 1
    assert deployment._read_csv(primary_path)[0]["review_status"] == "REVIEWED"
    assert recover_incomplete_commits(root) == 0
