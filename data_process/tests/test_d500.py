"""Scaling integrity, engineering failures and resumable cache behavior."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import requests

from data_process.lumae_scale import incremental, source_pool
from data_process.lumae_scale.cache_validation import atomic_npz, clip_npz_valid, visual_index_valid
from data_process.lumae_scale.incremental_session import IncrementalSession
from data_process.lumae_scale.layout import BuildLayout
from data_process.lumae_scale.models import atomic_json, digest, read_json, load_jsonl
from data_process.lumae_scale.preparation import metadata_preflight, SourceFailure, FailureKind, classify_failure
from data_process.lumae_scale.telemetry import Telemetry, aggregate
from data_process.lumae_scale.state_machine import validate_transition


@pytest.fixture
def synthetic(tmp_path, monkeypatch):
    layout = BuildLayout(root=tmp_path)
    rows = [{"qid": f"lumae_ads_v1_{i}" if i < 48 else f"lumae_ads_d200_{i:04d}",
             "vid": f"lumae_ads_{i:032x}", "query": f"parent query {i}", "duration": 20,
             "relevant_windows": [[2, 6]], "relevant_clip_ids": None, "saliency_scores": None,
             "metadata": {"review_provenance": "HUMAN_VERIFIED" if i < 18 else "AI_PSEUDO_LABELED"}}
            for i in range(200)]
    layout.base.mkdir(parents=True)
    for name, items in {"all": rows, "train": rows[:140], "val": rows[140:170], "test": rows[170:]}.items():
        (layout.base / f"lumae_ads_d200_{name}.jsonl").write_bytes(incremental._jsonl_bytes(items))
    atomic_json(layout.base / "dataset_manifest.json", {
        "total_samples": 200, "dataset_sha256": digest(layout.base / "lumae_ads_d200_all.jsonl"),
        "source_video_sha256": {r["qid"]: {"sha256": f"{n:064x}", "video_filename": f"{n:032x}.mp4"} for n,r in enumerate(rows)}})
    (layout.base / "checksums.sha256").write_text("".join(f"{digest(p)}  {p.name}\n" for p in sorted(layout.base.iterdir())))
    atomic_json(tmp_path / "local_data/manifests/lumae_ads_d500_protected_inputs.json", {
        "sha256": {p.relative_to(tmp_path).as_posix(): digest(p) for p in layout.base.iterdir()}
    })
    records = [{"source_video_id": f"{i:032x}", "source_video_sha256": f"{i:064x}",
                "source_video_path": f"local_data/raw/adsqa/videos/{i:032x}.mp4", "source_url": "https://video.adsoftheworld.com/fixture.mp4",
                "query": f"A woman lifts the red package numbered {i}.", "duration": 20, "window": [2, 6],
                "review_provenance": "AI_PSEUDO_LABELED", "router": {"route": "MEDIUM"}, "verifier": {"pass": True}}
               for i in range(200, 500)]
    for record in records:
        atomic_json(layout.accepted / (record["source_video_id"] + ".json"), record)
    atomic_json(layout.selection, {"candidates": []})
    layout.ledger.write_text("{}\n")
    monkeypatch.setattr(incremental, "validate_annotation", lambda *a, **kw: None)
    monkeypatch.setattr(incremental, "verify_frozen_ranker", lambda *a: None)
    return layout, rows, records


def test_five_hundred_freeze_preserves_parent_and_fixed_benchmark(synthetic):
    layout, parent, _ = synthetic
    before = {p.name: p.read_bytes() for p in layout.base.iterdir()}
    manifest = incremental.freeze_incremental(layout)
    assert manifest["total_samples"] == 500
    assert manifest["provenance_counts"] == {"human_verified": 18, "ai_pseudo_labeled": 482}
    for s,n in (("train",440),("val",30),("test",30)):
        assert len(load_jsonl(layout.dataset / f"lumae_ads_d500_{s}.jsonl")) == n
    for s in ("val", "test"):
        assert (layout.dataset / f"lumae_ads_d500_{s}.jsonl").read_bytes() == before[f"lumae_ads_d200_{s}.jsonl"]
    assert load_jsonl(layout.dataset / "lumae_ads_d500_all.jsonl")[:200] == parent
    assert {p.name:p.read_bytes() for p in layout.base.iterdir()} == before
    assert incremental.freeze_incremental(layout) == manifest
    assert read_json(layout.handoff)["feature_manifest_status"] == "EXTERNAL_PHASE_B_NOT_EXTRACTED"
    master = layout.reports / "lumae_ads_d500_annotation_master.csv"
    master.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="Frozen D500 output would change"):
        incremental.freeze_incremental(layout)


@pytest.mark.parametrize("failure", ["count", "id_overlap", "duplicate_id", "content_overlap", "duplicate_query", "parent_query_overlap", "fake_human"])
def test_extension_rejects_integrity_failures(synthetic, failure):
    layout, parent, records = synthetic
    if failure == "count": records.pop()
    elif failure == "id_overlap": records[0]["source_video_id"] = f"{0:032x}"
    elif failure == "duplicate_id": records[0]["source_video_id"] = records[1]["source_video_id"]
    elif failure == "content_overlap": records[0]["source_video_sha256"] = f"{0:064x}"
    elif failure == "duplicate_query": records[0]["query"] = records[1]["query"]
    elif failure == "parent_query_overlap": records[0]["query"] = parent[0]["query"]
    elif failure == "fake_human": records[0]["review_provenance"] = "HUMAN_VERIFIED"
    with pytest.raises(ValueError): incremental.validate_extension(layout, records)
    assert not layout.dataset.exists()


def test_cache_validates_contents_and_atomic_replacement(tmp_path):
    path = tmp_path / "features.npz"
    record = {"frames": [{"timestamp": 0}, {"timestamp": 2}]}
    atomic_npz(path, features=np.ones((2,512), dtype=np.float32), timestamps=np.array([0,2]))
    record["clip_features_sha256"] = digest(path)
    assert clip_npz_valid(path, record)
    original = path.read_bytes()
    with pytest.raises(ValueError): atomic_npz(path, features=np.array([np.nan]))
    assert path.read_bytes() == original
    atomic_npz(path, features=np.ones((2,511)), timestamps=np.array([0,2]))
    record["clip_features_sha256"] = digest(path)
    assert not clip_npz_valid(path, record)
    path.write_bytes(b"invalid zip")
    record["clip_features_sha256"] = digest(path)
    assert not clip_npz_valid(path, record)
    assert not visual_index_valid({}, tmp_path, "hash")
    assert not list(tmp_path.glob("*.tmp"))


def test_source_duplicates_rejected_before_io_and_failure_classes():
    row = {"source_video_id": "a"*32, "target_name": "a"*32+".mp4", "source_url": "https://video.adsoftheworld.com/v.mp4"}
    with pytest.raises(SourceFailure) as exc: metadata_preflight(row, {row["source_video_id"]})
    assert exc.value.kind == FailureKind.DUPLICATE
    response = requests.Response(); response.status_code = 404
    assert classify_failure(requests.HTTPError(response=response)) == FailureKind.PERMANENT_SOURCE_FAILURE
    assert classify_failure(requests.Timeout()) == FailureKind.RETRYABLE_NETWORK_ERROR
    assert classify_failure(OSError()) == FailureKind.RETRYABLE_IO_ERROR
    assert classify_failure(ValueError()) == FailureKind.CONTENT_REJECTED
    for stage in ("downloaded", "visual_index_ready", "awaiting_visual_inspection", "awaiting_interval_verification"):
        validate_transition(stage, "download_deferred")
    with pytest.raises(ValueError): validate_transition("accepted", "rejected")


def test_telemetry_keeps_misses_and_unknown_tokens(tmp_path, monkeypatch):
    t = Telemetry(tmp_path, "vid")
    t.mark_cache("clip", False); t.mark_cache("clip", True)
    with t.stage("clip"): pass
    t.usage(visual_calls=2)
    t.finish("ACCEPT", visual_views_count=2)
    report = aggregate(tmp_path)
    assert report["cache"]["clip"] == {"hits":1,"checks":2,"hit_rate":0.5}
    assert report["input_tokens"] is None and report["visual_calls"] == 2
    from data_process.lumae_scale import telemetry
    monkeypatch.setattr(telemetry, "atomic_json", lambda *a: (_ for _ in ()).throw(OSError("telemetry disk failure")))
    assert t.finish("ACCEPT")["telemetry_write_errors"]


def test_prefetch_stays_bounded_and_prunes_finished_jobs(monkeypatch):
    session = IncrementalSession.__new__(IncrementalSession)
    session.closed = False; session.active = None; session.jobs = {}
    session.layout = SimpleNamespace(accepted=Path("nonexistent_fixture_cache"))
    sources = [{"source_video_id": str(i)} for i in range(100)]
    session.eligible = lambda: sources
    session._prepare = lambda row: row
    class Job:
        def done(self): return True
        def cancel(self): return True
    class Pool:
        def submit(self,*args): return Job()
    session.pool = Pool()
    session.fill_queue()
    assert len(session.jobs) == 16
    sources = sources[16:]
    session.fill_queue()
    assert len(session.jobs) == 16 and set(session.jobs) == {str(i) for i in range(16,32)}
    session.closed = True; sources = sources[16:]
    session.fill_queue()
    assert set(session.jobs) == {str(i) for i in range(16,32)}


def test_protected_input_tampering_detected(tmp_path):
    layout = BuildLayout(root=tmp_path)
    p=tmp_path/"frozen.json";p.write_bytes(b"original")
    atomic_json(tmp_path/"local_data/manifests/lumae_ads_d500_protected_inputs.json", {"sha256":{"frozen.json":digest(p)}})
    guard = incremental.ProtectedInputs(layout)
    assert guard.verify(full=True)["status"] == "PASS"
    p.write_bytes(b"modified")
    with pytest.raises(RuntimeError): guard.verify()
