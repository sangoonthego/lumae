"""Exact-count D200 freeze, leakage checks, handoff and reproducible packaging."""

from __future__ import annotations

import csv
import hashlib
import json
import random
import re
import statistics
import zipfile
from collections import Counter
from pathlib import Path

from data_process.annotation.assisted_deployment import ALGORITHM_SHA256

from .foundation import verify_d48
from .models import D48, D200, ROOT, SEED, VISUAL_CACHE, WORK, atomic_json, digest, load_jsonl, read_json
from .quality_gate import interval_errors, query_errors
from .repo_paths import resolve_path, to_repo_relative

ACCEPTED = WORK / "accepted"


def portable_copy(value):
    """Normalize export copies only; historical accepted/cache bytes stay intact."""
    if isinstance(value, dict):
        return {k: portable_copy(v) for k, v in value.items()}
    if isinstance(value, list):
        return [portable_copy(v) for v in value]
    if isinstance(value, str) and (re.match(r"^[A-Za-z]:[\\/]", value)
                                   or value.startswith(("/content/", "/home/", "\\\\"))):
        return to_repo_relative(value, root=ROOT)
    return value


def accepted_records() -> list[dict]:
    return [read_json(p) for p in sorted(ACCEPTED.glob("*.json"))]


def validate_complete(records: list[dict], target: int = 200) -> tuple[list[dict], list[dict]]:
    verify_d48()
    if target != 200:
        raise ValueError("This task freezes only D200")
    old = load_jsonl(D48 / "lumae_ads_v1_all.jsonl")
    if len(records) != 152:
        raise ValueError(f"Need 152 accepted new samples, have {len(records)}")
    old_hashes = []
    for row in old:
        path = ROOT / "local_data/raw/adsqa/videos" / row["metadata"]["video_filename"]
        if not path.is_file():
            raise ValueError("Missing frozen D48 source video")
        old_hashes.append(digest(path))
    if len(set(old_hashes + [r["source_video_sha256"] for r in records])) != 200:
        raise ValueError("D200 contains duplicate video content")
    if len({r["source_video_id"] for r in records}) != 152:
        raise ValueError("Duplicate new source videos")
    if len({r["source_video_sha256"] for r in records}) != 152:
        raise ValueError("Duplicate new video content")
    if len({r["query"].strip().casefold() for r in records}) != 152:
        raise ValueError("Duplicate new temporal queries")
    for record in records:
        source = resolve_path(record["source_video_path"])
        if not source.is_file() or digest(source) != record["source_video_sha256"]:
            raise ValueError("Missing or changed source video")
        if query_errors(record["query"]) or interval_errors(*record["window"], record["duration"]):
            raise ValueError("Invalid accepted query or interval")
        if record.get("query_sanitation") != {"passed": True, "errors": []}:
            raise ValueError("Accepted query lacks sanitation result")
        phase = record.get("approximate_event_phase_seconds", [])
        if len(phase) != 2 or interval_errors(*phase, record["duration"]):
            raise ValueError("Invalid visually inspected coarse event phase")
        if not record.get("visual_evidence") or not record.get("verifier", {}).get("pass"):
            raise ValueError("Missing visual inspection/verification")
        index_path = VISUAL_CACHE / record["source_video_id"] / "metadata.json"
        if not index_path.is_file() or digest(index_path) != record["visual_index_sha256"]:
            raise ValueError("Missing or changed visual index")
        for sheet in record["visual_evidence"]["contact_sheets"]:
            if not (index_path.parent / sheet).is_file():
                raise ValueError("Missing inspected contact sheet")
        verification_image = resolve_path(record["verification_image_path"])
        if not verification_image.is_file() or digest(verification_image) != record["verification_image_sha256"]:
            raise ValueError("Missing or changed before/inside/after inspection image")
        semantic_path = resolve_path(record["semantic_artifact_path"])
        if not semantic_path.is_file() or digest(semantic_path) != record["semantic_artifact_sha256"]:
            raise ValueError("Missing or changed frozen semantic_v3 prediction")
        if record["semantic_v3_prediction_sha256"] != record["semantic_artifact_sha256"]:
            raise ValueError("Semantic prediction checksum mismatch")
        prediction = read_json(semantic_path)
        if (prediction.get("query") != record["query"]
                or [prediction.get("start"), prediction.get("end")] != record["window"]
                or prediction.get("semantic_v3_algorithm_sha256") != ALGORITHM_SHA256):
            raise ValueError("Semantic prediction does not match accepted sample")
        expected_provenance = {
            "query_source": "AI_GENERATED",
            "query_review_status": "AI_AUTO_ACCEPTED_FOR_TRAINING",
            "temporal_label_source": "SEMANTIC_V3_PSEUDO_LABEL",
            "review_provenance": "AI_PSEUDO_LABELED",
            "reviewer_id": "AI_PIPELINE",
            "semantic_v3_algorithm_sha256": ALGORITHM_SHA256,
        }
        if any(record.get(key) != value for key, value in expected_provenance.items()):
            raise ValueError("Fake human provenance in new sample")
        if not record.get("query_provider_config"):
            raise ValueError("Query provider/model provenance missing")
    rows = []
    for index, record in enumerate(sorted(records, key=lambda r: r["source_video_id"]), 49):
        qid = f"lumae_ads_d200_{index:04d}"
        rows.append({"qid": qid, "vid": f"lumae_ads_{record['source_video_id']}",
                     "query": record["query"], "duration": record["duration"],
                     "relevant_windows": [record["window"]], "relevant_clip_ids": None,
                     "saliency_scores": None, "source": "adsqa", "intent": "product_demo",
                     "original_id": qid,
                     "metadata": portable_copy({**{k: v for k, v in record.items() if k not in
                                      {"query", "duration", "window"}},
                                  "video_filename": Path(record["source_video_path"]).name})})
    all_rows = old + rows
    if len(all_rows) != target or len({r["qid"] for r in all_rows}) != target or len({r["vid"] for r in all_rows}) != target:
        raise ValueError("D200 count/uniqueness mismatch")
    counts = Counter(r["metadata"]["review_provenance"] for r in all_rows)
    if counts != {"HUMAN_VERIFIED": 18, "AI_PSEUDO_LABELED": 182}:
        raise ValueError("D200 provenance mismatch")
    return old, rows


def _jsonl_bytes(rows: list[dict]) -> bytes:
    return b"".join((json.dumps(r, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8") for r in rows)


def _write_immutable(path: Path, data: bytes) -> None:
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f"Frozen output would change: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


def freeze(target: int = 200, seed: int = SEED) -> dict:
    records = accepted_records()
    old, new = validate_complete(records, target)
    if seed != SEED:
        raise ValueError("D200 split seed must be 20260930")
    # Preserve exact original D48 canonical lines in the all/train JSONL files.
    old_bytes = (D48 / "lumae_ads_v1_all.jsonl").read_bytes()
    if not old_bytes.endswith(b"\n"):
        raise ValueError("D48 JSONL has no terminal newline")
    new_by_id = {r["vid"]: r for r in new}
    shuffled = sorted(new_by_id)
    random.Random(seed).shuffle(shuffled)
    train_new = [new_by_id[i] for i in shuffled[:92]]
    val = [new_by_id[i] for i in shuffled[92:122]]
    test = [new_by_id[i] for i in shuffled[122:]]
    if (len(train_new), len(val), len(test)) != (92, 30, 30):
        raise ValueError("Invalid split size")
    splits = {"train": old + train_new, "val": val, "test": test}
    vid_sets = {k: {r["vid"] for r in v} for k, v in splits.items()}
    if any(vid_sets[a] & vid_sets[b] for a, b in
           (("train", "val"), ("train", "test"), ("val", "test"))):
        raise ValueError("Video leakage")
    if not {r["vid"] for r in old} <= vid_sets["train"]:
        raise ValueError("D48 leaked outside train")
    output = {"all": old_bytes + _jsonl_bytes(new),
              "train": old_bytes + _jsonl_bytes(train_new),
              "val": _jsonl_bytes(val), "test": _jsonl_bytes(test)}
    for name, data in output.items():
        _write_immutable(D200 / f"lumae_ads_d200_{name}.jsonl", data)
    split_manifest = {"seed": seed, "counts": {k: len(v) for k, v in splits.items()},
                      "d48_train_only": True,
                      "video_ids": {k: sorted(v) for k, v in vid_sets.items()},
                      "provenance": {k: dict(Counter(r["metadata"]["review_provenance"] for r in v))
                                     for k, v in splits.items()}}
    atomic_json(D200 / "split_manifest.json", split_manifest)
    selection_path = WORK / "source_selection.json"
    selection_data = portable_copy(read_json(selection_path))
    _write_immutable(D200 / "selection_manifest.json",
                     (json.dumps(selection_data, sort_keys=True, indent=2) + "\n").encode())
    ledger_path = WORK / "build_ledger.jsonl"
    if not ledger_path.is_file():
        raise ValueError("D200 build ledger is missing")
    _write_immutable(D200 / "build_ledger.jsonl",
                     _jsonl_bytes(portable_copy(load_jsonl(ledger_path))))
    audit = [{"qid": row["qid"], "vid": row["vid"], "query": row["query"],
              "window": row["relevant_windows"][0], "visual_evidence": rec["visual_evidence"],
              "semantic_artifact_path": rec["semantic_artifact_path"],
              "verifier": rec["verifier"], "clip_signals": rec["clip_signals"]}
             for row, rec in zip(new, sorted(records, key=lambda r: r["source_video_id"]), strict=True)]
    atomic_json(D200 / "annotation_audit.json", portable_copy(audit))
    manifest = {"dataset_version": "lumae_ads_d200", "total_samples": 200,
                "new_samples": 152, "reused_d48": 48, "seed": seed,
                "d48_fingerprint_sha256": digest(WORK / "d48_fingerprint.json"),
                "source_selection_sha256": digest(selection_path),
                "portable_selection_sha256": digest(D200 / "selection_manifest.json"),
                "original_ledger_sha256": digest(ledger_path),
                "accepted_record_sha256": {p.stem: digest(p) for p in sorted((WORK / "accepted").glob("*.json"))},
                "path_export_policy": "Only derived export copies normalize historical paths; accepted and D48 source bytes are unchanged.",
                "split_manifest_sha256": digest(D200 / "split_manifest.json"),
                "dataset_sha256": digest(D200 / "lumae_ads_d200_all.jsonl"),
                "source_video_sha256": {
                    row["qid"]: {
                        "video_filename": row["metadata"]["video_filename"],
                        "sha256": digest(ROOT / "local_data/raw/adsqa/videos" /
                                         row["metadata"]["video_filename"])
                    } for row in old + new
                },
                "provenance_counts": {"human_verified": 18, "ai_pseudo_labeled": 182},
                "warning": "Validation and test annotations are AI pseudo labels, not human ground truth."}
    atomic_json(D200 / "dataset_manifest.json", manifest)
    checksum_files = sorted(p for p in D200.iterdir() if p.is_file() and p.name != "checksums.sha256")
    checksums = "".join(f"{digest(p)}  {p.name}\n" for p in checksum_files)
    _write_immutable(D200 / "checksums.sha256", checksums.encode("utf-8"))
    handoff_path = ROOT / "local_data/manifests/lumae_ads_d200_training_handoff.json"
    handoff = {"dataset_version": "lumae_ads_d200", "total_samples": 200,
               "train": 140, "val": 30, "test": 30, "human_verified": 18,
               "ai_pseudo_labeled": 182, "feature_extractor": "CLIP ViT-B/32",
               "feature_mode": "CLIP-only", "clip_length": 2.0, "max_video_length": 75,
               "max_query_length": 32, "model": "Moment-DETR",
               "initialization": "QVHighlights pretrained CLIP-only M0",
               "has_saliency_gt": False, "saliency_loss_weight": 0.0, "seed": seed,
               "recommended_saliency_loss_weight": 0.0,
               "dataset_manifest_path": "local_data/datasets/lumae_ads_d200/dataset_manifest.json",
               "dataset_manifest_sha256": digest(D200 / "dataset_manifest.json"),
               "train_manifest": {"path": "local_data/datasets/lumae_ads_d200/lumae_ads_d200_train.jsonl",
                                  "sha256": digest(D200 / "lumae_ads_d200_train.jsonl")},
               "val_manifest": {"path": "local_data/datasets/lumae_ads_d200/lumae_ads_d200_val.jsonl",
                                "sha256": digest(D200 / "lumae_ads_d200_val.jsonl")},
               "test_manifest": {"path": "local_data/datasets/lumae_ads_d200/lumae_ads_d200_test.jsonl",
                                 "sha256": digest(D200 / "lumae_ads_d200_test.jsonl")},
               "paths_sha256": {p.name: digest(p) for p in checksum_files},
               "dataset_sha256": manifest["dataset_sha256"],
               "annotation_warning": manifest["warning"]}
    atomic_json(handoff_path, handoff)
    return manifest


def package_colab(destination: Path | None = None) -> dict:
    manifest = read_json(D200 / "dataset_manifest.json")
    if manifest["total_samples"] != 200:
        raise ValueError("D200 freeze incomplete")
    out = destination or (ROOT / "local_data/exports/lumae_ads_d200_colab.zip")
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = load_jsonl(D200 / "lumae_ads_d200_all.jsonl")
    videos = [ROOT / "local_data/raw/adsqa/videos" / r["metadata"].get(
        "video_filename", r["vid"].removeprefix("lumae_ads_") + ".mp4")
        for r in rows]
    if len(videos) != 200 or any(not p.is_file() for p in videos):
        raise ValueError("Colab package requires all 200 source videos")
    support = [ROOT / "experiments/M1_lumae_ads" / name for name in
               ("train_d200.py", "train_frozen.py", "config_d200.json", "README_D200.md")]
    if any(not p.is_file() for p in support):
        raise ValueError("D200 Colab runner files are missing")
    files = (sorted(D200.glob("*")) +
             [ROOT / "local_data/manifests/lumae_ads_d200_training_handoff.json"] +
             sorted(videos) +
             support)
    master = ROOT / "local_data/reports/lumae_ads_d200/lumae_ads_d200_annotation_master.csv"
    if master.is_file():
        files.append(master)
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        for p in files:
            archive.write(p, p.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(out) as archive:
        if archive.testzip() is not None:
            raise ValueError("Colab ZIP integrity failure")
        names = archive.namelist()
        if len(set(names)) != len(files) or any(
                n.startswith(("/", "\\")) or ":" in n or ".." in Path(n).parts for n in names):
            raise ValueError("Colab ZIP contains nonportable or duplicate entries")
    return {"path": to_repo_relative(out), "file_count": len(files), "bytes": out.stat().st_size,
            "sha256": digest(out), "archive_integrity": "PASS", "portable_entry_paths": "PASS"}
