"""Tests for LUMAE D1500 Test-378 freeze, exclusion guard, portability, multi-query loading, and packaging."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import pathlib
import zipfile

import numpy as np

from data_process.test450 import test378
from data_process.test450.parent import CANONICAL, OUTPUT, ROOT, SOURCE, Parent
from experiments.M1_lumae_ads import cached_runner as runner

OUT_DIR = ROOT / "research_outputs" / "d1500_test378_ai_v1"
TEST300_DIR = ROOT / "research_outputs" / "d1500_test300_ai_v1"


def _write_npz(path: pathlib.Path, **arrays: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_test378_cutoff_snapshot_and_excluded_preservation() -> None:
    assert not (OUTPUT / "release" / "manifest.json").exists()
    state = test378.collect_test378_state(Parent(), OUTPUT)
    assert len(state["new_rows"]) == 228
    assert len(state["combined_rows"]) == 378
    assert state["multiplicity_hist"] == {"1": 36, "3": 114}
    assert len(state["excluded_records"]) == 2
    assert {item["row"]["vid"] for item in state["excluded_records"]} == {test378.BLOCKED_CLIP_TOKEN_VID}
    assert state["quality"]["validated_text_features"] == 228
    assert state["quality"]["feature_errors"] == {}
    assert state["quality"]["diversity"]["exact_duplicates"] == []
    assert state["quality"]["diversity"]["flags"] == []
    assert state["quality"]["max_clip_content_tokens"] <= 30


def test_test378_sealed_release_and_protected_test300_intact() -> None:
    for folder in (OUT_DIR, TEST300_DIR):
        sha_file = folder / "SHA256SUMS.txt"
        assert sha_file.is_file()
        for line in sha_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            digest, rel = line.split("  ", 1)
            assert hashlib.sha256((folder / rel).read_bytes()).hexdigest() == digest

    manifest = json.loads((OUT_DIR / "release" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["dataset_version"] == test378.VERSION
    assert manifest["counts"]["combined_test_queries"] == 378
    assert manifest["counts"]["new_test_queries"] == 228
    assert len(manifest["text_features"]) == 228


def test_test378_zip_portability_isolation_and_multi_query_loader(tmp_path: pathlib.Path) -> None:
    pkg_a = OUT_DIR / "packages" / "lumae_test378_incremental.zip"
    pkg_b = OUT_DIR / "packages" / "lumae_colab_benchmark.zip"
    assert test378.verify_test378_zip_portable(pkg_a)["non_portable_refs"] == 0
    assert test378.verify_test378_zip_portable(pkg_b)["non_portable_refs"] == 0

    ws = tmp_path / "ws"
    with zipfile.ZipFile(pkg_b, "r") as zb:
        zb.extractall(ws)
    with zipfile.ZipFile(pkg_a, "r") as za:
        za.extractall(ws)

    assert len((ws / "dataset" / "train.jsonl").read_text(encoding="utf-8").splitlines()) == 1200
    assert len((ws / "dataset" / "val.jsonl").read_text(encoding="utf-8").splitlines()) == 150
    assert len((ws / "dataset" / "test378.jsonl").read_text(encoding="utf-8").splitlines()) == 378
    assert len((ws / "dataset" / "new_test228.jsonl").read_text(encoding="utf-8").splitlines()) == 228
    nb = json.loads((ws / "LUMAE_D1500_Benchmark_3Seeds_Ablation.ipynb").read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4 and len(nb["cells"]) >= 8

    # Synthetic multi-query shared-video test + extracted colab_eval_test378 slice verification.
    feature_root = tmp_path / "feat_ws" / "features"
    vid_a, vid_b = "lumae_ads_demo_multi", "lumae_ads_demo_single"
    _write_npz(feature_root / "clip_features" / f"{vid_a}.npz", features=np.ones((8, 512), dtype=np.float32))
    _write_npz(feature_root / "clip_features" / f"{vid_b}.npz", features=np.full((6, 512), 2.0, dtype=np.float32))

    rows = [
        {"qid": f"{vid_a}_orig", "vid": vid_a, "query": "orig query", "duration": 16.0, "relevant_windows": [[2.0, 6.0]]},
        {"qid": f"{vid_a}_new1", "vid": vid_a, "query": "new query one", "duration": 16.0, "relevant_windows": [[6.0, 10.0]]},
        {"qid": f"{vid_a}_new2", "vid": vid_a, "query": "new query two", "duration": 16.0, "relevant_windows": [[10.0, 14.0]]},
        {"qid": f"{vid_b}_orig", "vid": vid_b, "query": "other orig", "duration": 12.0, "relevant_windows": [[1.0, 5.0]]},
    ]
    text_manifest = {}
    for idx, row in enumerate(rows):
        rel = f"test378_text_features/{row['qid']}/vec.npz"
        arr = np.zeros((5, 512), dtype=np.float32)
        arr[:, idx] = 1.0
        _write_npz(feature_root / rel, last_hidden_state=arr)
        text_manifest[row["qid"]] = rel

    ann_path = tmp_path / "test378.jsonl"
    ann_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    feat_manifest = {
        "canonical_sha256": CANONICAL,
        "feature_contract": {"tef_in_cache": False},
        "video": {vid_a: f"clip_features/{vid_a}.npz", vid_b: f"clip_features/{vid_b}.npz"},
        "text": text_manifest,
    }
    ds = runner.dataset_type(SOURCE / "upstream")(
        ann_path, feature_root, feat_manifest, json.loads((ws / "configs" / "moment_detr_d1500_recipe.json").read_text(encoding="utf-8"))
    )
    items = [ds[i] for i in range(len(ds))]
    assert len(items) == 4
    assert items[0]["model_inputs"]["video_feat"].shape[1] == 514
    assert not np.allclose(items[0]["model_inputs"]["query_feat"], items[1]["model_inputs"]["query_feat"])

    eval_path = ws / "scripts" / "colab_eval_test378.py"
    spec = importlib.util.spec_from_file_location("colab_eval_test378", eval_path)
    assert spec and spec.loader
    eval_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(eval_mod)
    preds = {row["qid"]: [[row["relevant_windows"][0][0], row["relevant_windows"][0][1], 0.99]] for row in rows}
    orig_report = eval_mod.evaluate_slice([rows[0], rows[3]], preds, SOURCE / "upstream", bootstrap=10)
    new_report = eval_mod.evaluate_slice(rows[1:3], preds, SOURCE / "upstream", bootstrap=10)
    combined_report = eval_mod.evaluate_slice(rows, preds, SOURCE / "upstream", bootstrap=10)
    assert orig_report["query_count"] == 2 and orig_report["video_count"] == 2 and orig_report["R1@0.5"] == 100.0
    assert new_report["query_count"] == 2 and new_report["video_count"] == 1 and new_report["R1@0.5"] == 100.0
    assert combined_report["query_count"] == 4 and combined_report["video_count"] == 2 and combined_report["R1@0.5"] == 100.0
