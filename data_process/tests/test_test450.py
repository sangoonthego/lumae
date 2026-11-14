import json
from types import SimpleNamespace

import pytest

from data_process.test450.annotation import AnnotationStore, SOURCE, progress
from data_process.test450.evaluation import load_predictions, metrics
from data_process.test450.parent import Feature, digest, jsonl_bytes, save_bytes, save_json
from data_process.test450.release import WEIGHTS_HASH, publish, text_path, validate_release
from data_process.test450.resume import queue
from data_process.test450.training import SEEDS, matrix


@pytest.fixture
def case(tmp_path):
    original = {
        "qid": "original", "vid": "video", "query": "A runner ties a shoe.",
        "duration": 8, "relevant_windows": [[0, 1]], "metadata": {"source_video_sha256": "hash"},
    }
    parent = SimpleNamespace(test={"video": original}, qids={"original"})
    store = AnnotationStore(parent, tmp_path)
    path = store.case("video", 1)
    for role, step in (("generation", 0.5), ("review", 0.25)):
        packet = {
            "duration": 8, "timestamps": [i * step for i in range(int(8 / step))],
            "sampling_seconds": step, "sheets": {"sheet.jpg": "unused"},
        }
        sheet = path / role / "sheet.jpg"
        save_bytes(sheet, role.encode())
        packet["sheets"]["sheet.jpg"] = digest(sheet)
        if role == "review":
            packet["queries"] = ["The runner picks up a blue cup.", "A door swings open."]
        save_json(path / role / "evidence.json", {k: v for k, v in packet.items() if k != "queries"})
        if role == "generation":
            save_json(path / role / "task.json", packet)
    return store, path


def decision(path, role):
    return {
        "agent": "visual reviewer", "model": "fixture", "session": role,
        "timestamp_utc": "2026-11-14T00:00:00+00:00",
        "task_sha256": digest(path / role / "task.json"),
        "events": [
            {"query": text, "decision": "accept", "interval": [start, start + 1],
             "confidence": 0.9, "unique_occurrence": True, "note": "Fixture evidence.",
             "evidence": {"before": [start - 0.5], "inside": [start + 0.5], "after": [start + 1.5]}}
            for text, start in [("The runner picks up a blue cup.", 2), ("A door swings open.", 5)]
        ],
    }


def test_blind_review_and_resumable_state(case):
    store, path = case
    generation = decision(path, "generation")
    store.record("video", 1, "generation", generation)
    assert progress(*store.accepted())["unresolved_videos"] == 1
    review_packet = store.review_packet("video", 1)
    assert not {"interval", "original_window", "events"} & review_packet.keys()
    review = decision(path, "review")
    with pytest.raises(ValueError, match="separate blind"):
        store.record("video", 1, "review", review | {"session": "generation"})
    store.record("video", 1, "review", review)
    store.record("video", 1, "review", review)
    rows, provenance, states = store.accepted()
    assert len(rows) == 2 and len({r["qid"] for r in rows}) == 2
    assert all(r["vid"] == "video" for r in rows)
    assert all(r["reviewer_agreement_tiou"] == 1 and r["annotation_source"] == SOURCE for r in provenance)
    assert progress(rows, provenance, states)["unresolved_videos"] == 0
    with pytest.raises(ValueError, match="Immutable artifact differs"):
        store.record("video", 1, "generation", generation | {"agent": "other"})
    (path / "review/sheet.jpg").write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        store.accepted()


def test_disagreement_and_retirement(case):
    store, path = case
    generation = decision(path, "generation")
    store.record("video", 1, "generation", generation)
    store.review_packet("video", 1)
    review = decision(path, "review")
    review["events"][0] |= {"interval": [2.75, 3.5],
                            "evidence": {"before": [2.5], "inside": [3], "after": [3.75]}}
    store.record("video", 1, "review", review)
    rows, _, states = store.accepted()
    assert not rows
    assert states["video"]["attempts"][0]["status"] == "BOUNDARY_DISAGREEMENT"
    store.retire("video", 1, {"agent": "fixture", "reason": "Distinct boundaries disagreed"})
    assert store.accepted()[2]["video"]["attempts"][0]["status"] == "RETIRED"


def test_resume_excludes_accepted_and_routes_review_and_retry():
    ready = {"status": "ACCEPTED_VISUAL_REVIEW", "attempts": []}
    failed = {"attempt": 2, "status": "BOUNDARY_DISAGREEMENT", "generation_decisions": ["accept", "accept"]}
    reviewing = {"attempt": 2, "status": "WAITING_REVIEW", "generation_decisions": ["accept", "accept"]}
    states = {
        "completed": ready,
        "pending": {"status": "UNRESOLVED", "attempts": [{"attempt": 2, "status": "WAITING_GENERATION"}]},
        "review": {"status": "UNRESOLVED", "attempts": [reviewing]},
        "retry": {"status": "UNRESOLVED", "attempts": [failed]},
        "exhausted": {"status": "UNRESOLVED", "attempts": [failed | {"attempt": i} for i in (2, 3, 4)]},
    }
    result = queue(states)
    assert "completed" not in result
    assert result["pending"]["status"] == "PENDING" and result["pending"]["attempt"] == 2
    assert result["review"]["status"] == "WAITING_REVIEW"
    assert result["retry"]["status"] == "RETRY_REQUIRED" and result["retry"]["attempt"] == 3
    assert result["exhausted"]["status"] == "BLOCKED" and result["exhausted"]["attempt"] == 4


def test_canonical_predictions_and_video_bootstrap():
    rows = [
        {"qid": qid, "vid": vid, "duration": 5, "relevant_windows": [[1, 2]]}
        for qid, vid in [("a", "shared"), ("b", "shared"), ("c", "other")]
    ]
    predictions = [
        {"qid": "a", "vid": "shared", "pred_relevant_windows": [[1, 2, 0.9]]},
        {"qid": "b", "vid": "shared", "pred_relevant_windows": [[1, 2, 0.9]]},
        {"qid": "c", "vid": "other", "pred_relevant_windows": []},
    ]
    canonical = load_predictions(rows, predictions)
    result = metrics(rows, canonical, bootstrap=100, seed=42)
    assert result["video_count"] == 2 and result["query_count"] == 3
    assert result["R1@0.5"] == result["MR_mAP"] == 66.67
    assert result["video_bootstrap"]["ci95"]["R1@0.5"] == [0, 100]
    with pytest.raises(ValueError, match="exactly match"):
        load_predictions(rows, predictions[:-1])


def test_matrix_has_nine_distinct_runs(tmp_path):
    parent = SimpleNamespace(seed=20261110, hashes={"train.jsonl": "train", "val.jsonl": "val"})
    plan = matrix(parent, tmp_path)
    assert plan["training_seeds"] == SEEDS
    assert len({(run["model"], run["seed"]) for run in plan["runs"]}) == 9
    assert len({run["planned_checkpoint"] for run in plan["runs"]}) == 9
    assert all(run["selection_split"] == "val" and run["execution"] == "NOT_RUN" for run in plan["runs"])
    assert all("test.jsonl" not in json.dumps(run["training_config"]) for run in plan["runs"])


def test_full_release_preserves_sealed_test_and_video_identity(tmp_path):
    import numpy as np

    def row(qid, vid):
        return {"qid": qid, "vid": vid, "query": qid.replace("_", " "), "duration": 5,
                "relevant_windows": [[1, 2]], "relevant_clip_ids": None, "saliency_scores": None}

    originals = [row(f"original_{i}", f"v{i}") for i in range(150)]
    additions = [row(f"extra{slot}_{i}", f"v{i}") for i in range(150) for slot in range(2)]
    train = [row(f"train_{i}", f"train{i}") for i in range(1200)]
    val = [row(f"val_{i}", f"val{i}") for i in range(150)]
    sidecar = [{"qid": item["qid"], "vid": item["vid"], "annotation_source": SOURCE,
                "reviewer_agreement_tiou": 0.8, "generation": {"events": [{"confidence": 0.9}]},
                "review": {"events": [{"confidence": 0.9}]}} for item in additions]
    states = {item["vid"]: {"status": "ACCEPTED_VISUAL_REVIEW", "attempts": []} for item in originals}
    frozen = tmp_path / "parent"
    frozen.mkdir()
    frozen_test = jsonl_bytes(originals)
    (frozen / "test.jsonl").write_bytes(frozen_test)
    video = frozen / "shared_video.npz"
    np.savez(video, features=np.ones((1, 512), dtype=np.float32))
    all_rows = originals + train + val
    video_features = {item["vid"]: Feature("video", item["vid"], video, digest(video)) for item in all_rows}
    text_features = {}
    for index, item in enumerate(originals + additions):
        file = text_path(tmp_path, item) if item in additions else frozen / f"{item['qid']}.npz"
        file.parent.mkdir(parents=True, exist_ok=True)
        vector = np.zeros((3, 512), dtype=np.float32)
        vector[1, index] = 1
        np.savez(file, last_hidden_state=vector)
        if item in additions:
            save_json(file.with_suffix(".json"), {
                "query_sha256": file.stem, "weights_sha256": WEIGHTS_HASH, "sha256": digest(file), "token_count": 3,
            })
        else:
            text_features[item["qid"]] = Feature("text", item["qid"], file, digest(file))
    for item in train + val:
        feature = text_features["original_0"]
        text_features[item["qid"]] = Feature("text", item["qid"], feature.path, feature.sha256)
    parent = SimpleNamespace(
        frozen=frozen, hashes={"test.jsonl": digest(frozen / "test.jsonl")},
        rows={"train": train, "val": val, "test": originals},
        test={item["vid"]: item for item in originals},
        features=lambda _: {"video": video_features, "text": text_features},
    )
    store = SimpleNamespace(work=tmp_path, accepted=lambda _: (additions, sidecar, states))
    manifest = publish(parent, store)
    assert manifest["counts"]["video_features"] == 1500
    assert manifest["counts"]["text_features"] == 1800
    assert (tmp_path / "release/test450.jsonl").read_bytes().startswith(frozen_test)
    assert validate_release(parent, store)["test_queries"] == 450
    assert publish(parent, store) == manifest
