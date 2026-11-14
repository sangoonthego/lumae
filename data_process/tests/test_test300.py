"""Tests for LUMAE D1500 Test-300 freezing, portability, multi-query loading, and packaging."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import pathlib
import zipfile

import numpy as np
import pytest

from data_process.test450 import test300
from data_process.test450.parent import CANONICAL, OUTPUT, ROOT, SOURCE, Parent
from experiments.M1_lumae_ads import cached_runner as runner

OUT_DIR = ROOT / "research_outputs" / "d1500_test300_ai_v1"


def _write_npz(path: pathlib.Path, **arrays: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_base_feature_index_maps_3000_unique_portable_paths() -> None:
    index = test300.build_base_feature_index(Parent())
    assert index["counts"] == {"video": 1500, "text": 1500, "total": 3000}
    assert index["package_c_required"] is False
    all_paths = [rec["manifest_path"] for kind in ("video", "text") for rec in index[kind].values()]
    assert len(set(all_paths)) == 3000
    assert all("\\" not in p and ":" not in p and not p.startswith("/") for p in all_paths)
    assert sum(index["source_breakdown"].values()) == 3000


def test_test300_ledger_state_and_open_test450_guard() -> None:
    assert not (OUTPUT / "release" / "manifest.json").exists()
    state = test300.collect_test300_state(Parent(), OUTPUT)
    assert len(state["new_rows"]) == 150
    assert len(state["combined_rows"]) == 300
    assert state["multiplicity_hist"] == {"1": 75, "3": 75}
    assert state["quality"]["validated_text_features"] == 150
    assert state["quality"]["feature_errors"] == {}
    assert state["quality"]["diversity"]["exact_duplicates"] == []
    assert state["quality"]["diversity"]["flags"] == []


def test_multi_query_shared_video_loader_and_eval_subsets(tmp_path: pathlib.Path) -> None:
    feature_root = tmp_path / "features"
    vid_a = "lumae_ads_demo_multi"
    vid_b = "lumae_ads_demo_single"
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
        rel = f"test300_text_features/{row['qid']}/vec.npz"
        arr = np.zeros((5, 512), dtype=np.float32)
        arr[:, idx] = 1.0
        _write_npz(feature_root / rel, last_hidden_state=arr)
        text_manifest[row["qid"]] = rel

    ann_path = tmp_path / "test300.jsonl"
    ann_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    manifest = {
        "canonical_sha256": CANONICAL,
        "feature_contract": {"tef_in_cache": False},
        "video": {vid_a: f"clip_features/{vid_a}.npz", vid_b: f"clip_features/{vid_b}.npz"},
        "text": text_manifest,
    }
    ds = runner.dataset_type(SOURCE / "upstream")(
        ann_path, feature_root, manifest, test300._portable_recipe(Parent(), 42)
    )
    assert len(ds) == 4
    items = [ds[i] for i in range(len(ds))]
    # Video tensor gains 2-col TEF on load -> 514 dims; multiple queries share the same vid tensor cleanly.
    assert items[0]["model_inputs"]["video_feat"].shape[1] == 514
    assert items[0]["model_inputs"]["video_feat"].shape == items[1]["model_inputs"]["video_feat"].shape
    assert items[1]["model_inputs"]["video_feat"].shape == items[2]["model_inputs"]["video_feat"].shape
    assert items[0]["model_inputs"]["query_feat"].shape == (5, 512)
    assert not np.allclose(items[0]["model_inputs"]["query_feat"], items[1]["model_inputs"]["query_feat"])

    # Verify standalone evaluator handles independent slices + shared-video multi-query combined slice cleanly.
    eval_mod_path = tmp_path / "colab_eval_test300.py"
    eval_mod_path.write_text(test300._colab_eval_script(), encoding="utf-8")
    spec = importlib.util.spec_from_file_location("colab_eval_test300", eval_mod_path)
    assert spec and spec.loader
    eval_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(eval_mod)

    runner.bind_upstream(SOURCE / "upstream")
    preds = {
        row["qid"]: [[row["relevant_windows"][0][0], row["relevant_windows"][0][1], 0.99]]
        for row in rows
    }
    orig_report = eval_mod.evaluate_slice([rows[0], rows[3]], preds, SOURCE / "upstream", bootstrap=10)
    new_report = eval_mod.evaluate_slice(rows[1:3], preds, SOURCE / "upstream", bootstrap=10)
    combined_report = eval_mod.evaluate_slice(rows, preds, SOURCE / "upstream", bootstrap=10)
    assert orig_report["query_count"] == 2 and orig_report["video_count"] == 2 and orig_report["R1@0.5"] == 100.0
    assert new_report["query_count"] == 2 and new_report["video_count"] == 1 and new_report["R1@0.5"] == 100.0
    assert combined_report["query_count"] == 4 and combined_report["video_count"] == 2 and combined_report["R1@0.5"] == 100.0


def test_packaged_archives_extract_and_verify_sha256sums() -> None:
    sha_file = OUT_DIR / "SHA256SUMS.txt"
    if not sha_file.exists():
        pytest.skip("Test-300 packages not yet materialized")
    for line in sha_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, rel = line.split("  ", 1)
        assert hashlib.sha256((OUT_DIR / rel).read_bytes()).hexdigest() == digest

    for zip_name in ("lumae_test300_incremental.zip", "lumae_colab_benchmark.zip"):
        zp = OUT_DIR / "packages" / zip_name
        check = test300.verify_zip_portable(zp)
        assert check["testzip"] == "OK" and check["non_portable_refs"] == 0
        with zipfile.ZipFile(zp, "r") as zf:
            names = set(zf.namelist())
            if zip_name == "lumae_test300_incremental.zip":
                assert "dataset/test300.jsonl" in names
                assert "dataset/new_test150.jsonl" in names
                assert "manifests/test300_feature_manifest.json" in names
                assert sum(n.startswith("features/test300_text_features/") and n.endswith(".npz") for n in names) == 150
            else:
                assert "upstream/SOURCE_MANIFEST.json" in names
                assert "experiments/M1_lumae_ads/cached_runner.py" in names
                assert "scripts/colab_prepare_workspace.py" in names
                assert "scripts/colab_eval_test300.py" in names
