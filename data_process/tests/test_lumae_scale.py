"""D200 integrity gates, deterministic selection and crash-resume behavior."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

from data_process.lumae_scale import freeze as freezing
from data_process.lumae_scale import pipeline, source_pool
from data_process.lumae_scale import query_generator
from data_process.lumae_scale import foundation
from data_process.lumae_scale.models import digest
from data_process.lumae_scale.quality_gate import interval_errors, query_errors


def test_frozen_d48_fingerprint_and_deterministic_source_order(tmp_path, monkeypatch):
    d48 = tmp_path / "local_data/datasets/lumae_ads_v1"
    d48.mkdir(parents=True)
    rows = [{"qid": f"q{i}", "vid": f"lumae_ads_{i:032x}",
             "metadata": {"review_provenance": "HUMAN_VERIFIED" if i < 18
                          else "AI_PSEUDO_LABELED"}} for i in range(48)]
    (d48 / "lumae_ads_v1_all.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    (d48 / "dataset_manifest.json").write_text("{}\n", encoding="utf-8")
    for name in ("train", "val", "test"):
        (d48 / f"lumae_ads_v1_{name}.jsonl").write_text("{}\n", encoding="utf-8")
    handoff = tmp_path / "local_data/manifests/lumae_ads_v1_training_handoff.json"
    handoff.parent.mkdir(parents=True)
    data = {"dataset_manifest_sha256": digest(d48 / "dataset_manifest.json")}
    for name in ("train", "val", "test"):
        path = d48 / f"lumae_ads_v1_{name}.jsonl"
        data[f"{name}_manifest"] = {"path": path.relative_to(tmp_path).as_posix(),
                                     "sha256": digest(path)}
    handoff.write_text(json.dumps(data), encoding="utf-8")
    for name, value in (("D48", d48), ("ROOT", tmp_path), ("HANDOFF", handoff),
                        ("WORK", tmp_path / "work"),
                        ("FINGERPRINT", tmp_path / "work/fingerprint.json")):
        monkeypatch.setattr(foundation, name, value)
    record = foundation.verify_d48()
    assert foundation.verify_d48() == record
    assert record["count"] == 48
    assert record["provenance_counts"] == {"HUMAN_VERIFIED": 18, "AI_PSEUDO_LABELED": 30}
    (d48 / "dataset_manifest.json").write_text("{\"changed\":true}", encoding="utf-8")
    with pytest.raises(ValueError):
        foundation.verify_d48()
    source = tmp_path / "source_manifest.json"
    source.write_text(json.dumps({"videos": [
        {"source_video_id": f"{i:032x}", "target_name": f"{i:032x}.mp4"} for i in range(53)
    ]}), encoding="utf-8")
    monkeypatch.setattr(source_pool, "ROOT", tmp_path)
    monkeypatch.setattr(source_pool, "SOURCE", source)
    (d48 / "dataset_manifest.json").write_text("{}\n", encoding="utf-8")
    left = source_pool.ordered_sources()
    right = source_pool.ordered_sources()
    assert [r["source_video_id"] for r in left] == [r["source_video_id"] for r in right]
    assert len(left) == 5


def test_query_and_window_sanitation():
    assert query_errors("The creator demonstrates the product.")
    assert query_errors("Something happens.")
    assert not query_errors("A woman presses the red button on the small device.")
    assert interval_errors(float("nan"), 4, 10)
    assert interval_errors(0, 10, 10)
    assert not interval_errors(2, 5, 10)


def test_local_visual_provider_protocol_requires_agent_inspection(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    work = tmp_path / "work"
    video_id = "fixture"
    index = cache / video_id / "metadata.json"
    index.parent.mkdir(parents=True)
    index.write_text(json.dumps({"contact_sheets": ["contact_sheets/sheet.jpg"],
                                 "frames": [{"timestamp": 2.0}]}), encoding="utf-8")
    monkeypatch.setattr(query_generator, "VISUAL_CACHE", cache)
    monkeypatch.setattr(query_generator, "WORK", work)
    script = tmp_path / "provider.py"
    script.write_text("import json,sys\np=json.load(sys.stdin)\n"
                      "print(json.dumps({'video_id':p['video_id'],'events':[]}))\n", encoding="utf-8")
    provider = query_generator.LocalCommandProvider([sys.executable, str(script)],
                                                     model_id="local-fixture", revision="v1")
    assert provider.events(video_id) == []
    with pytest.raises(ValueError, match="requires agent inspection"):
        query_generator.require_agent_inspection(video_id)
    record = work / "visual_index" / video_id / "agent_inspection.json"
    record.parent.mkdir(parents=True)
    record.write_text(json.dumps({"inspection_method": "agent_viewed_contact_sheets",
                                  "inspected_contact_sheets": ["contact_sheets/sheet.jpg"]}),
                      encoding="utf-8")
    query_generator.require_agent_inspection(video_id)


def test_exact_200_freeze_and_train_only_d48(tmp_path: Path, monkeypatch):
    root = tmp_path
    d48 = root / "local_data/datasets/lumae_ads_v1"
    d200 = root / "local_data/datasets/lumae_ads_d200"
    work = root / "local_data/intermediate/lumae_ads_d200"
    visual = root / "local_data/cache/lumae_visual_index"
    d48.mkdir(parents=True)
    work.mkdir(parents=True)
    (work / "d48_fingerprint.json").write_text("{}\n", encoding="utf-8")
    (work / "source_selection.json").write_text("{}\n", encoding="utf-8")
    (work / "build_ledger.jsonl").write_text("{}\n", encoding="utf-8")
    old = []
    for i in range(48):
        filename = f"old_{i}.mp4"
        old_video = root / "local_data/raw/adsqa/videos" / filename
        old_video.parent.mkdir(parents=True, exist_ok=True)
        old_video.write_bytes(f"old video {i}".encode())
        old.append({"qid": f"old_{i}", "vid": f"lumae_ads_old_{i}",
                    "query": "A person lifts a book from the table.", "duration": 10.0,
                    "relevant_windows": [[2.0, 4.0]], "relevant_clip_ids": None,
                    "saliency_scores": None, "metadata": {
                        "review_provenance": "HUMAN_VERIFIED" if i < 18 else "AI_PSEUDO_LABELED",
                        "video_filename": filename}})
    original_bytes = b"".join((json.dumps(r) + "\n").encode() for r in old)
    (d48 / "lumae_ads_v1_all.jsonl").write_bytes(original_bytes)
    records = []
    for i in range(152):
        vid = f"{i:032x}"
        query = f"A woman presses the red button on a device at station {i}."
        video = root / "local_data/raw/adsqa/videos" / f"{vid}.mp4"
        video.parent.mkdir(exist_ok=True)
        video.write_bytes(f"fixture-{i}".encode())
        artifact = work / "predictions" / f"{vid}.json"
        artifact.parent.mkdir(exist_ok=True)
        artifact.write_text(json.dumps({"query": query,
                                        "start": 2.0, "end": 5.0,
                                        "semantic_v3_algorithm_sha256": freezing.ALGORITHM_SHA256}),
                            encoding="utf-8")
        index = visual / vid / "metadata.json"
        index.parent.mkdir(parents=True)
        index.write_text("{}", encoding="utf-8")
        sheet = index.parent / "contact_sheets" / "contact_sheet_000.jpg"
        sheet.parent.mkdir()
        sheet.write_bytes(b"sheet")
        verification = work / f"verification_{vid}.jpg"
        verification.write_bytes(b"verification")
        records.append({"source_video_id": vid, "source_video_path": str(video),
                        "source_video_sha256": digest(video), "query": query,
                        "duration": 10.0, "window": [2.0, 5.0],
                        "approximate_event_phase_seconds": [1.0, 6.0],
                        "query_sanitation": {"passed": True, "errors": []},
                        "visual_evidence": {"contact_sheets": ["contact_sheets/contact_sheet_000.jpg"],
                                            "timestamps": [2.0]},
                        "visual_index_sha256": digest(index),
                        "verification_image_path": str(verification),
                        "verification_image_sha256": digest(verification),
                        "verifier": {"pass": True, "notes": "visible"},
                        "semantic_artifact_path": str(artifact),
                        "semantic_artifact_sha256": digest(artifact),
                        "semantic_v3_prediction_sha256": digest(artifact),
                        "review_provenance": "AI_PSEUDO_LABELED",
                        "reviewer_id": "AI_PIPELINE",
                        "query_source": "AI_GENERATED",
                        "query_review_status": "AI_AUTO_ACCEPTED_FOR_TRAINING",
                        "temporal_label_source": "SEMANTIC_V3_PSEUDO_LABEL",
                        "semantic_v3_algorithm_sha256": freezing.ALGORITHM_SHA256,
                        "query_provider_config": {"model_id": "fixture", "revision": "v1"},
                        "clip_signals": {"inside_minus_outside_margin": 0.1}})
    for name, value in (("ROOT", root), ("D48", d48), ("D200", d200),
                        ("WORK", work), ("VISUAL_CACHE", visual)):
        monkeypatch.setattr(freezing, name, value)
    monkeypatch.setattr(freezing, "verify_d48", lambda: {})
    monkeypatch.setattr(freezing, "accepted_records", lambda: records)
    manifest = freezing.freeze()
    assert manifest["total_samples"] == 200
    assert (d200 / "lumae_ads_d200_all.jsonl").read_bytes().startswith(original_bytes)
    splits = {name: freezing.load_jsonl(d200 / f"lumae_ads_d200_{name}.jsonl")
              for name in ("train", "val", "test")}
    assert [len(splits[k]) for k in ("train", "val", "test")] == [140, 30, 30]
    assert {r["qid"] for r in old} <= {r["qid"] for r in splits["train"]}
    assert not ({r["vid"] for r in splits["val"]} & {r["vid"] for r in splits["test"]})
    assert len({r["qid"] for group in splits.values() for r in group}) == 200
    assert len({r["vid"] for group in splits.values() for r in group}) == 200
    assert all(r["source"] == "adsqa" for group in splits.values()
               for r in group if r["qid"].startswith("lumae_ads_d200_"))
    assert all(r["saliency_scores"] is None and r["relevant_clip_ids"] is None
               for group in splits.values() for r in group)
    from experiments.M1_lumae_ads.train_d200 import preflight
    _, _, checked_splits = preflight(root)
    assert len(checked_splits["train"]) == 140
    assert freezing.freeze() == manifest  # idempotent resume
    with pytest.raises(ValueError, match="152"):
        freezing.validate_complete(records[:-1])
    Path(records[0]["source_video_path"]).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="source video"):
        freezing.validate_complete(records)


def test_rejected_candidate_is_replaced_without_restarting(tmp_path: Path, monkeypatch):
    state = {"candidates": [
        {"source_video_id": "bad", "target_name": "bad.mp4", "status": "unattempted"},
        {"source_video_id": "next", "target_name": "next.mp4", "status": "unattempted"},
    ]}
    video = tmp_path / "next.mp4"
    video.write_bytes(b"video")
    monkeypatch.setattr(pipeline, "verify_d48", lambda: {})
    monkeypatch.setattr(pipeline, "verify_frozen_ranker", lambda _: None)
    monkeypatch.setattr(pipeline, "selection", lambda _: state)
    monkeypatch.setattr(pipeline, "accepted_records", lambda: [])
    monkeypatch.setattr(pipeline, "ACCEPTED", tmp_path / "accepted")
    monkeypatch.setattr(pipeline, "WORK", tmp_path)
    monkeypatch.setattr(pipeline, "LEDGER", tmp_path / "ledger.jsonl")
    monkeypatch.setattr(pipeline, "write_report", lambda: {})
    def fake_download(row):
        if row["source_video_id"] == "bad":
            raise ValueError("corrupt source")
        return video
    monkeypatch.setattr(pipeline, "download", fake_download)
    monkeypatch.setattr(pipeline, "probe", lambda _: 20.0)
    monkeypatch.setattr(pipeline, "build_visual_index", lambda *_, **__: {"contact_sheets": ["sheet.jpg"]})
    from data_process.lumae_scale import throughput
    monkeypatch.setattr(throughput, "TIMINGS", tmp_path / "timings")
    monkeypatch.setattr(pipeline, "write_profile", lambda: {})
    monkeypatch.setattr(pipeline, "save_candidate",
                        lambda _, video_id, **changes: next(r for r in state["candidates"]
                                                              if r["source_video_id"] == video_id).update(changes))
    class EmptyProvider:
        def events(self, video_id):
            return []
    result = pipeline.run(provider=EmptyProvider(), max_candidates=2)
    assert result["status"] == "AWAITING_VISUAL_INSPECTION"
    assert result["video_id"] == "next"
    assert state["candidates"][0]["status"] == "rejected"
    assert state["candidates"][1]["status"] == "awaiting_visual_inspection"
    result = pipeline.run(provider=EmptyProvider(), max_candidates=1)
    assert result["video_id"] == "next"  # the rejected candidate stays skipped


def test_router_never_treats_a_human_review_as_an_automatic_verifier():
    from data_process.lumae_scale.router import route_candidate
    from data_process.annotation.assisted_deployment import ALGORITHM_SHA256
    query = "A woman presses the red button on a small device."
    prediction = {"start": 2.0, "end": 5.0, "semantic_confidence": "MEDIUM",
                  "semantic_v3_algorithm_sha256": ALGORITHM_SHA256}
    evidence = {"timestamps": [2.0, 4.0], "contact_sheets": ["sheet.jpg"]}
    assert route_candidate(query=query, prediction=prediction, duration=10,
                           visual_evidence=evidence, automatic_verifier=None)["route"] == "MEDIUM"
    assert route_candidate(query=query, prediction={**prediction, "end": 11}, duration=10,
                           visual_evidence=evidence, automatic_verifier=None)["route"] == "LOW"
    verifier = {"pass": True, "model_id": "fixture-vlm", "frame_references": ["a.jpg"],
                "inside_event_visible": True, "correct_phase": True,
                "outside_event_absent": True, "specific_event": True, "boundary_valid": True}
    assert route_candidate(query=query, prediction=prediction, duration=10,
                           visual_evidence=evidence, automatic_verifier=verifier)["route"] == "HIGH"
    assert route_candidate(query=query, prediction=prediction, duration=10,
                           visual_evidence=evidence, automatic_verifier={**verifier, "correct_phase": False})["route"] == "MEDIUM"


def test_historical_calibration_is_fixed_to_first_eleven_candidates(tmp_path, monkeypatch):
    from data_process.lumae_scale import router
    work = tmp_path / "work"
    accepted = work / "accepted"
    accepted.mkdir(parents=True)
    baseline = [{"source_video_id": f"old{i}", "status": "accepted"} for i in range(10)]
    baseline.append({"source_video_id": "rejected", "status": "rejected"})
    baseline.append({"source_video_id": "new", "status": "accepted"})
    (work / "source_selection.json").write_text(json.dumps({"candidates": baseline}), encoding="utf-8")
    prediction = work / "prediction.json"
    prediction.write_text(json.dumps({"query": "A woman presses the red button on a small device.",
                                      "start": 2.0, "end": 5.0,
                                      "semantic_v3_algorithm_sha256": "frozen"}), encoding="utf-8")
    verification = work / "verification.jpg"
    verification.write_bytes(b"fixture")
    record = {"query": "A woman presses the red button on a small device.",
              "duration": 10.0, "window": [2.0, 5.0],
              "visual_evidence": {"timestamps": [2.0], "contact_sheets": ["sheet.jpg"]},
              "verification_image_path": str(verification),
              "verifier": {"evidence_timestamps": {"before": [0.0],
                                                    "inside": [2.0], "after": [6.0]}},
              "semantic_artifact_path": str(prediction)}
    for row in baseline:
        if row["status"] == "accepted":
            (accepted / f"{row['source_video_id']}.json").write_text(
                json.dumps({**record, "source_video_id": row["source_video_id"]}), encoding="utf-8")
    monkeypatch.setattr(router, "WORK", work)
    monkeypatch.setattr(router, "CALIBRATION", tmp_path / "calibration.json")
    result = router.calibrate()
    assert result["accepted_examples"] == 10
    assert result["rejected_examples"] == 1
    assert result["high_auto_accept_enabled"] is False
    assert "new" not in {r["source_video_id"] for r in result["examples"]}


def test_parallel_prefetch_serializes_manifest_writes(tmp_path, monkeypatch):
    import threading
    from data_process.lumae_scale import prefetch as preparing, throughput
    ids = [f"video{i}" for i in range(6)]
    state = {"candidates": [{"source_video_id": video_id, "status": "unattempted"}
                            for video_id in ids]}
    main_thread = threading.get_ident()
    writes = []
    monkeypatch.setattr(preparing, "selection", lambda _: state)
    monkeypatch.setattr(preparing, "_fetch", lambda row: (
        tmp_path / row["source_video_id"], 30.0, row["source_video_id"], False, 0.1, 0.1))
    monkeypatch.setattr(preparing, "build_visual_index", lambda path, duration, sha, timer: {
        "frames": [{"timestamp": 0}, {"timestamp": 2}]})
    def save(_, video_id, **changes):
        writes.append((threading.get_ident(), video_id, changes["status"]))
        next(row for row in state["candidates"] if row["source_video_id"] == video_id).update(changes)
    monkeypatch.setattr(preparing, "save_candidate", save)
    monkeypatch.setattr(preparing, "write_profile", lambda: {})
    monkeypatch.setattr(throughput, "TIMINGS", tmp_path / "timings")
    result = preparing.prefetch(count=6, download_workers=4, ffmpeg_workers=2)
    assert result["ready"] == 6 and result["failed"] == 0
    assert all(tid == main_thread for tid, _, _ in writes)
    assert {video_id for _, video_id, status in writes if status == "visual_index_ready"} == set(ids)


def test_high_random_audit_is_stable_and_covers_at_least_ten_percent():
    from data_process.lumae_scale.audit import select_audit_ids
    ids = [f"video_{i:03d}" for i in range(23)]
    first = select_audit_ids(ids)
    assert len(first) == 3
    assert first == select_audit_ids(list(reversed(ids)))
    assert select_audit_ids([]) == []
    with pytest.raises(ValueError, match="Duplicate"):
        select_audit_ids(["same", "same"])


# ============================================================
# FOCUSED CORE STABILIZATION TEST SUITE (Fixes #1 - #7 & Gates)
# ============================================================

def test_1_clip_singleton_and_persistent_loading():
    from data_process.lumae_scale.clip_runtime import get_clip_runtime, reset_clip_runtimes
    reset_clip_runtimes()
    r1 = get_clip_runtime("ViT-B/32", device="cpu")
    r2 = get_clip_runtime("ViT-B/32", device="cpu")
    assert r1 is r2


def test_2_repeated_refinement_does_not_reload_clip():
    from data_process.lumae_scale.clip_runtime import get_clip_runtime
    runtime = get_clip_runtime("ViT-B/32", device="cpu")
    init_before = runtime.init_count
    v1 = runtime.encode_text("A person pours tea.")
    v2 = runtime.encode_text("A person opens a door.")
    v3 = runtime.encode_text("A car turns around.")
    assert runtime.init_count == max(1, init_before)
    assert v1.shape == (1, 512)
    assert v2.shape == (1, 512)
    assert v3.shape == (1, 512)


def test_3_path_relative_to_absolute_resolution():
    from data_process.lumae_scale.repo_paths import PathResolver
    from data_process.lumae_scale.models import ROOT
    resolver = PathResolver(ROOT)
    resolved = resolver.resolve("local_data/raw/adsqa/videos/test.mp4")
    assert resolved == (ROOT / "local_data/raw/adsqa/videos/test.mp4").resolve()


def test_4_windows_to_relative_migration():
    from data_process.lumae_scale.repo_paths import PathResolver
    resolver = PathResolver("D:/DUT/Se07/CV/lumae")
    win_path = "D:\\DUT\\Se07\\CV\\lumae\\local_data\\raw\\adsqa\\videos\\0d1.mp4"
    rel = resolver.to_relative(win_path)
    assert rel == "local_data/raw/adsqa/videos/0d1.mp4"


def test_5_linux_colab_resolution_portability():
    from data_process.lumae_scale.repo_paths import PathResolver
    colab = PathResolver("/content/lumae")
    res_colab = colab.resolve("local_data/raw/adsqa/videos/0d1.mp4")
    assert res_colab.as_posix() == "/content/lumae/local_data/raw/adsqa/videos/0d1.mp4"

    linux = PathResolver("/home/research/lumae")
    res_linux = linux.resolve("local_data/intermediate/predictions/abc.json")
    assert res_linux.as_posix() == "/home/research/lumae/local_data/intermediate/predictions/abc.json"


def test_6_absolute_path_detection():
    from data_process.lumae_scale.repo_paths import is_canonical_relative
    assert is_canonical_relative("local_data/raw/adsqa/videos/0d1.mp4")
    assert not is_canonical_relative("D:\\DUT\\Se07\\CV\\lumae\\test.mp4")
    assert not is_canonical_relative("/home/user/test.mp4")

    # Verify zero machine-absolute paths exist in all accepted records
    from data_process.lumae_scale.models import ROOT, read_json
    accepted_files = list((ROOT / "local_data/intermediate/lumae_ads_d200/accepted").glob("*.json"))
    for f in accepted_files:
        rec = read_json(f)
        for key in ("source_video_path", "semantic_artifact_path", "verification_image_path"):
            val = rec.get(key)
            if val:
                assert is_canonical_relative(val), f"Non-canonical path in {f.name}: {key}={val}"


def test_7_legacy_router_backfill():
    from data_process.lumae_scale.models import ROOT, read_json
    report_p = ROOT / "local_data/reports/lumae_ads_d200/d200_legacy_router_migration.json"
    assert report_p.is_file()
    report = read_json(report_p)
    assert report["migrated_count"] == 10
    for entry in report["records"]:
        rec = read_json(ROOT / f"local_data/intermediate/lumae_ads_d200/accepted/{entry['source_video_id']}.json")
        assert rec["router"]["route"] == "PRE_ROUTER_LEGACY"
        assert rec["router"]["routing_version"] == "pre_router"


def test_8_no_fake_high_history_in_legacy_backfill():
    from data_process.lumae_scale.models import ROOT, read_json
    report = read_json(ROOT / "local_data/reports/lumae_ads_d200/d200_legacy_router_migration.json")
    for entry in report["records"]:
        rec = read_json(ROOT / f"local_data/intermediate/lumae_ads_d200/accepted/{entry['source_video_id']}.json")
        assert rec["router"]["route"] != "HIGH"
        assert rec["router"]["auto_accepted"] is False


def test_9_state_machine_valid_transitions_and_recovery():
    from data_process.lumae_scale.state_machine import can_transition, validate_transition
    assert can_transition("unattempted", "downloading")
    assert can_transition("downloading", "downloaded")
    assert can_transition("visual_index_ready", "awaiting_visual_inspection")
    assert not can_transition("accepted", "downloading")
    with pytest.raises(ValueError, match="Illegal state machine transition"):
        validate_transition("accepted", "unattempted", "fixture")


def test_10_network_error_does_not_content_reject():
    import requests
    from data_process.lumae_scale.state_machine import is_network_or_io_retryable
    assert is_network_or_io_retryable(requests.RequestException("connection closed"))
    assert is_network_or_io_retryable(ConnectionResetError("connection reset by peer"))
    assert is_network_or_io_retryable(TimeoutError("timed out"))
    assert not is_network_or_io_retryable(ValueError("invalid query syntax"))


def test_11_cache_stage_fingerprint_generation():
    from data_process.lumae_scale.cache_identity import VisualIndexIdentity, ClipCacheIdentity
    id1 = VisualIndexIdentity(source_sha256="abc123sha")
    id2 = VisualIndexIdentity(source_sha256="abc123sha")
    id3 = VisualIndexIdentity(source_sha256="different_sha")
    assert id1.fingerprint() == id2.fingerprint()
    assert id1.fingerprint() != id3.fingerprint()


def test_12_cache_invalidation_rules():
    from data_process.lumae_scale.cache_identity import does_invalidation_affect
    assert does_invalidation_affect("visual_index", "clip")
    assert does_invalidation_affect("semantic", "refinement")
    assert not does_invalidation_affect("verifier", "visual_index")
    assert not does_invalidation_affect("verifier", "clip")


def test_13_source_sha_mismatch_invalidates_dependent_cache():
    from data_process.lumae_scale.cache_identity import does_invalidation_affect
    assert does_invalidation_affect("source", "visual_index")
    assert does_invalidation_affect("source", "clip")
    assert does_invalidation_affect("source", "semantic")


def test_14_verifier_version_change_does_not_invalidate_video_download():
    from data_process.lumae_scale.cache_identity import does_invalidation_affect
    assert not does_invalidation_affect("verifier", "source")
    assert not does_invalidation_affect("verifier", "visual_index")


def test_15_semantic_v3_algorithm_hash_remains_frozen():
    from data_process.annotation.assisted_deployment import verify_frozen_ranker, ALGORITHM_SHA256
    from data_process.lumae_scale.models import ROOT
    verify_frozen_ranker(ROOT)
    assert ALGORITHM_SHA256 == "cc7097db06a1376819a193b09054c135f8e5fadcef67ba58a82c34b24ba5d5fd"


def test_16_d48_fingerprint_remains_frozen():
    from data_process.lumae_scale.foundation import verify_d48
    res = verify_d48()
    assert res["count"] == 48
    expected_dataset_sha = "f02c0c393d4d685ba8d9fa8ca02b0222362da8a13effbd0970ee5715b8bb37a4"
    actual = res.get("file_sha256", {}).get("local_data/datasets/lumae_ads_v1/lumae_ads_v1_all.jsonl")
    assert actual == expected_dataset_sha


def test_17_temporal_windows_unchanged_during_migration():
    from data_process.lumae_scale.models import ROOT, read_json
    mig_dir = next((ROOT / "local_data/intermediate/lumae_ads_d200/migrations").glob("pre_core_stabilization_*"))
    for backup_f in (mig_dir / "accepted").glob("*.json"):
        curr_f = ROOT / f"local_data/intermediate/lumae_ads_d200/accepted/{backup_f.name}"
        assert curr_f.is_file()
        before = read_json(backup_f)
        after = read_json(curr_f)
        assert after["window"] == before["window"], f"Window modified for {backup_f.name}!"
        assert after["duration"] == before["duration"]


def test_18_query_text_unchanged_during_migration():
    from data_process.lumae_scale.models import ROOT, read_json
    mig_dir = next((ROOT / "local_data/intermediate/lumae_ads_d200/migrations").glob("pre_core_stabilization_*"))
    for backup_f in (mig_dir / "accepted").glob("*.json"):
        curr_f = ROOT / f"local_data/intermediate/lumae_ads_d200/accepted/{backup_f.name}"
        before = read_json(backup_f)
        after = read_json(curr_f)
        assert after["query"] == before["query"], f"Query modified for {backup_f.name}!"


def test_19_accepted_sample_count_matches_final_freeze():
    from data_process.lumae_scale.models import ROOT
    files = list((ROOT / "local_data/intermediate/lumae_ads_d200/accepted").glob("*.json"))
    assert len(files) == 152


def test_20_negative_margin_audit_does_not_silently_relabel():
    from data_process.lumae_scale.models import ROOT, read_json
    audit_p = ROOT / "local_data/reports/lumae_ads_d200/d200_negative_clip_margin_audit.json"
    assert audit_p.is_file()
    audit = read_json(audit_p)
    assert audit["total_negative_margin_records"] == 20
    assert audit["visually_supported_count"] == 20
    assert audit["obvious_mismatch_count"] == 0
    # verify windows are identical to original accepted
    for item in audit["records"]:
        rec = read_json(ROOT / f"local_data/intermediate/lumae_ads_d200/accepted/{item['source_video_id']}.json")
        assert rec["window"] == item["window"]
        assert rec["query"] == item["query"]
        assert rec["quality_audit_status"] == "VISUALLY_SUPPORTED"


def test_final_d200_integrity_and_master_csv():
    import csv
    from data_process.lumae_scale.finalize_cpu import audit, MASTER_PATH
    result = audit()
    assert (result["total_samples"], result["new_accepted"]) == (200, 152)
    assert result["splits"] == {"train": 140, "val": 30, "test": 30}
    assert result["provenance"] == {"HUMAN_VERIFIED": 18, "AI_PSEUDO_LABELED": 182}
    with MASTER_PATH.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == len({r["qid"] for r in rows}) == 200
    assert sum(r["final_split"] == "train" for r in rows) == 140
    assert sum(bool(r["total_accepted_seconds"]) for r in rows) == 86
    assert all(r["relative_video_path"].startswith("local_data/") for r in rows)


def test_final_colab_zip_matches_frozen_files():
    import zipfile
    from data_process.lumae_scale.models import ROOT, D200
    bundle = ROOT / "local_data/exports/lumae_ads_d200_colab.zip"
    with zipfile.ZipFile(bundle) as archive:
        names = archive.namelist()
        assert len(names) == len(set(names))
        assert all(not n.startswith(("/", "\\")) and ":" not in n and ".." not in Path(n).parts for n in names)
        assert sum(n.startswith("local_data/raw/adsqa/videos/") for n in names) == 200
        for path in D200.iterdir():
            if path.is_file():
                assert hashlib.sha256(archive.read(path.relative_to(ROOT).as_posix())).hexdigest() == digest(path)
        assert "experiments/M1_lumae_ads/train_d200.py" in names
        assert "local_data/manifests/lumae_ads_d200_training_handoff.json" in names
