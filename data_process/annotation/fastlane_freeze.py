"""A.2.7 integrity gate, sealed video split, and notebook-ready dataset freeze."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import statistics
import tempfile

from data_process.annotation import assisted_deployment as deployment
from data_process.annotation.fastlane import _file_sha, current_lock, current_prediction
from data_process.annotation.human_query_review import decision_of, worklist
from data_process.annotation.query_sanitation import load_query_reviews
from data_process.validation.schema import validate_canonical_dict


ROOT = Path(__file__).resolve().parents[2]
SEED = 20260930
VERSION = "lumae_ads_v1"
DEVELOPMENT_EXPOSED = frozenset({
    "lumae_ads_pilot_0004", "lumae_ads_pilot_0006", "lumae_ads_pilot_0007",
    "lumae_ads_pilot_0008", "lumae_ads_pilot_0010", "lumae_ads_pilot_0011",
    "lumae_ads_pilot_0012", "lumae_ads_pilot_0013", "lumae_ads_pilot_0017",
    "lumae_ads_pilot_0018", "lumae_ads_pilot_0020", "lumae_ads_pilot_0037",
    "lumae_ads_pilot_0043", "lumae_ads_pilot_0028", "lumae_ads_pilot_0033",
    "lumae_ads_pilot_0040", "lumae_ads_pilot_0045", "lumae_ads_pilot_0048",
})
LEGACY_RISK = frozenset({"lumae_ads_pilot_0002", "lumae_ads_pilot_0009"})
FORCED_TRAIN = DEVELOPMENT_EXPOSED | LEGACY_RISK


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def _source_hashes(root: Path) -> dict[str, str]:
    names = (
        "local_data/annotations/lumae_ads/human_primary.csv",
        "local_data/annotations/lumae_ads/query_reviews.csv",
        "local_data/annotations/lumae_ads/assisted_annotation_audit.csv",
    )
    return {name: _file_sha(root / name) for name in names}


def _read_inputs(root: Path) -> tuple[dict, dict, dict, set[str]]:
    ann = deployment._paths(root)[0]
    primary = deployment._unique(deployment._read_csv(ann / "human_primary.csv"), "human_primary.csv")
    queries = load_query_reviews(ann / "query_reviews.csv")
    audit = deployment._unique(deployment._read_csv(ann / "assisted_annotation_audit.csv"),
                               "assisted_annotation_audit.csv")
    cohort = {item["sample_id"] for item in worklist(root)}
    return primary, queries, audit, cohort


def _gate(root: Path, expected_total: int) -> tuple[list[dict], dict, dict]:
    deployment.verify_frozen_ranker(root)
    errors = deployment.frozen_artifact_errors(root)
    if errors:
        raise ValueError("Historical frozen artifact integrity failed: " + "; ".join(errors))
    primary, queries, audit, cohort = _read_inputs(root)
    if len(primary) != expected_total:
        raise ValueError(f"Pilot count {len(primary)} differs from expected {expected_total}")
    if any(row["review_status"] not in {"REVIEWED", "EXCLUDED"} for row in primary.values()):
        pending = sorted(sid for sid, row in primary.items() if row["review_status"] == "DRAFT")
        raise ValueError(f"Dataset freeze blocked by {len(pending)} unresolved samples")
    unknown_audit = set(audit) - set(primary)
    if unknown_audit:
        raise ValueError(f"Temporal audit contains unknown samples: {sorted(unknown_audit)}")
    videos = deployment._paths(root)[2]
    samples = []
    inspection = []
    video_hashes = {}
    for sid, row in sorted(primary.items()):
        status = row["review_status"]
        reviewer = row["annotator_id"].strip()
        if not reviewer.startswith("human_"):
            raise ValueError(f"Missing human temporal reviewer provenance: {sid}")
        if status == "EXCLUDED":
            if not row["annotation_notes"].strip() or row["annotation_notes"].strip() == "EXCLUDED":
                raise ValueError(f"Excluded sample lacks human reason: {sid}")
            if row["gt_start_seconds"].strip() or row["gt_end_seconds"].strip():
                raise ValueError(f"Excluded sample contains temporal GT: {sid}")
            if sid in cohort and (sid not in audit or audit[sid]["temporal_decision"] != "EXCLUDE"):
                raise ValueError(f"Fast-lane exclusion lacks human audit: {sid}")
            inspection.append({"sample_id": sid, "status": "EXCLUDED",
                               "reviewer_id": reviewer, "reason": row["annotation_notes"]})
            continue
        try:
            start, end, duration = map(float, (
                row["gt_start_seconds"], row["gt_end_seconds"], row["duration_seconds"]))
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Missing temporal GT: {sid}") from exc
        if not (all(map(math.isfinite, (start, end, duration))) and 0 <= start < end <= duration):
            raise ValueError(f"Invalid temporal GT: {sid}")
        if not row["query"].strip() or not row["vid"].strip() or not row["source_video_id"].strip():
            raise ValueError(f"Reviewed sample lacks query or video ID: {sid}")
        if row["source_video_id"] != Path(row["video_filename"]).stem:
            raise ValueError(f"Video ID and filename mismatch: {sid}")
        if not (videos / row["video_filename"]).is_file():
            raise ValueError(f"Reviewed sample video is missing: {sid}")
        video_hash = _file_sha(videos / row["video_filename"])
        video_hashes[sid] = {"video_filename": row["video_filename"], "sha256": video_hash}
        if sid in cohort:
            q = queries.get(sid)
            if not q or decision_of(q) not in {"ACCEPT_SUGGESTION", "EDIT_QUERY", "KEEP_ORIGINAL"}:
                raise ValueError(f"Fast-lane sample lacks human query approval: {sid}")
            lock = current_lock(root, sid)
            pred = current_prediction(root, sid)
            if lock is None or pred is None or sid not in audit:
                raise ValueError(f"Fast-lane sample lacks lock, prediction, or human temporal audit: {sid}")
            if (q.human_final_query != row["query"]
                    or pred["query_sha256"] != lock["query_sha256"]
                    or pred["video_sha256"] != video_hash
                    or audit[sid]["temporal_decision"] not in {"ACCEPT_AI", "EDIT_AI", "REJECT_AI"}
                    or audit[sid]["reviewer_id"] != reviewer):
                raise ValueError(f"Fast-lane provenance mismatch: {sid}")
            if [float(audit[sid]["ai_start"]), float(audit[sid]["ai_end"])] != [
                pred["pred_start_seconds"], pred["pred_end_seconds"]
            ]:
                raise ValueError(f"Fast-lane AI interval audit mismatch: {sid}")
            query_provenance = {"type": "human_approved_fastlane", "reviewer_id": q.reviewer_id,
                                "reviewed_at_utc": q.reviewed_at_utc, "query_version": q.query_version,
                                "query_sha256": lock["query_sha256"]}
            temporal_provenance = {"type": "human_verified_semantic_v3_advice",
                                   "decision": audit[sid]["temporal_decision"],
                                   "reviewed_at_utc": audit[sid]["reviewed_at_utc"]}
        else:
            query_provenance = {"type": "historical_human_primary",
                                "reviewer_id": reviewer}
            temporal_provenance = {"type": "historical_human_primary"}
        sample = {
            "qid": sid, "vid": row["vid"], "query": row["query"].strip(),
            "duration": duration, "relevant_windows": [[start, end]],
            "relevant_clip_ids": None, "saliency_scores": None,
            "source": "Lumae Ads", "intent": "product_demo", "original_id": sid,
            "metadata": {
                "video_filename": row["video_filename"],
                "product_category": row.get("product_category", ""),
                "source_video_dataset": "AdsQA",
                "query_creator": "LUMAE",
                "temporal_gt_creator": "LUMAE human reviewer",
                "annotation_reviewer_id": reviewer,
                "query_review_provenance": query_provenance,
                "temporal_review_provenance": temporal_provenance,
            },
        }
        valid, validation_errors = validate_canonical_dict(sample)
        if not valid:
            raise ValueError(f"Canonical sample invalid {sid}: {validation_errors}")
        samples.append(sample)
        inspection.append({"sample_id": sid, "status": "REVIEWED", "reviewer_id": reviewer,
                           "query_sha256": hashlib.sha256(row["query"].strip().encode()).hexdigest(),
                           "window": [start, end], "query_provenance": query_provenance,
                           "temporal_provenance": temporal_provenance})
    if len({r["qid"] for r in samples}) != len(samples):
        raise ValueError("Duplicate canonical qid")
    return samples, {"sample_checks": inspection, "source_hashes": _source_hashes(root),
                     "source_video_sha256": video_hashes,
                     "historical_frozen_artifacts_verified": True,
                     "semantic_v3_algorithm_sha256": deployment.ALGORITHM_SHA256}, primary


def _split(samples: list[dict]) -> tuple[dict[str, list[dict]], dict]:
    by_vid: dict[str, list[dict]] = defaultdict(list)
    for sample in samples:
        by_vid[sample["vid"]].append(sample)
    n = len(by_vid)
    val_target = round(n * 0.15)
    test_target = round(n * 0.15)
    forced_vids = {vid for vid, group in by_vid.items()
                   if any(item["qid"] in FORCED_TRAIN for item in group)}
    clean_vids = sorted(set(by_vid) - forced_vids)
    if len(clean_vids) < val_target + test_target:
        raise ValueError("Insufficient clean videos for requested validation/test split")
    random.Random(SEED).shuffle(clean_vids)
    val_vids = set(clean_vids[:val_target])
    test_vids = set(clean_vids[val_target:val_target + test_target])
    train_vids = set(by_vid) - val_vids - test_vids
    if train_vids & val_vids or train_vids & test_vids or val_vids & test_vids:
        raise ValueError("Video-level split leakage")
    splits = {"train": [], "val": [], "test": []}
    for sample in sorted(samples, key=lambda item: item["qid"]):
        vid = sample["vid"]
        name = "val" if vid in val_vids else "test" if vid in test_vids else "train"
        splits[name].append(sample)
    seen = [item["qid"] for rows in splits.values() for item in rows]
    if len(seen) != len(set(seen)) or set(seen) != {sample["qid"] for sample in samples}:
        raise ValueError("Included samples do not appear exactly once")
    if any(item["qid"] in FORCED_TRAIN for name in ("val", "test") for item in splits[name]):
        raise ValueError("Development-exposed sample entered validation/test")
    forced_reasons = {item["qid"]: (
        "LEGACY_TEMPORAL_EXPOSURE_RISK" if item["qid"] in LEGACY_RISK
        else "TEMPORAL_PREANNOTATOR_DEVELOPMENT_EXPOSURE")
        for item in splits["train"] if item["qid"] in FORCED_TRAIN}
    manifest = {
        "seed": SEED, "split_policy": "deterministic_video_level_forced_exposure_train",
        "target_ratios": {"train": 0.70, "val": 0.15, "test": 0.15},
        "forced_train_ids": sorted(forced_reasons), "forced_train_reasons": forced_reasons,
        **{f"{name}_ids": [item["qid"] for item in splits[name]] for name in splits},
        **{f"{name}_vids": sorted({item["vid"] for item in splits[name]}) for name in splits},
    }
    return splits, manifest


def _stats(rows: list[dict]) -> dict:
    if not rows:
        return {"sample_count": 0, "unique_video_count": 0,
                "duration_seconds": None, "product_category_distribution": {},
                "query_length_words": None, "query_length_characters": None}
    def summary(values: list[float]) -> dict:
        return {"mean": round(statistics.mean(values), 4),
                "median": round(statistics.median(values), 4),
                "min": min(values), "max": max(values)}
    return {
        "sample_count": len(rows), "unique_video_count": len({r["vid"] for r in rows}),
        "duration_seconds": summary([r["duration"] for r in rows]),
        "product_category_distribution": dict(sorted(Counter(
            r["metadata"]["product_category"] for r in rows).items())),
        "query_length_words": summary([len(r["query"].split()) for r in rows]),
        "query_length_characters": summary([len(r["query"]) for r in rows]),
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _verify_existing(root: Path, directory: Path, source_hashes: dict[str, str]) -> dict:
    manifest_path = directory / "dataset_manifest.json"
    if not manifest_path.is_file():
        raise ValueError("Frozen v1 directory exists without dataset manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("source_artifact_sha256") != source_hashes:
        raise ValueError("Frozen v1 exists with different source bytes; choose a new version")
    if len(manifest.get("source_video_sha256", {})) != manifest.get("included_count"):
        raise ValueError("Frozen v1 lacks complete source video hashes")
    for item in manifest["source_video_sha256"].values():
        video = deployment._paths(root)[2] / item["video_filename"]
        if not video.is_file() or _file_sha(video) != item["sha256"]:
            raise ValueError(f"Frozen v1 source video changed: {item['video_filename']}")
    checks = (directory / "checksums.sha256").read_text(encoding="utf-8").splitlines()
    for line in checks:
        digest, name = line.split("  ", 1)
        path = directory / name
        if not path.is_file() or _file_sha(path) != digest:
            raise ValueError(f"Frozen v1 checksum mismatch: {name}")
    return manifest


def _handoff(root: Path, directory: Path, manifest: dict) -> Path:
    rel_dir = directory.relative_to(root).as_posix()
    files = {name: {"path": f"{rel_dir}/{VERSION}_{name}.jsonl",
                    "sha256": _file_sha(directory / f"{VERSION}_{name}.jsonl")}
             for name in ("train", "val", "test")}
    payload = {
        "dataset_version": VERSION, "dataset_manifest_path": f"{rel_dir}/dataset_manifest.json",
        "dataset_manifest_sha256": _file_sha(directory / "dataset_manifest.json"),
        "all_manifest": {"path": f"{rel_dir}/{VERSION}_all.jsonl",
                         "sha256": manifest["dataset_sha256"]},
        "train_manifest": files["train"], "val_manifest": files["val"],
        "test_manifest": files["test"], "split_seed": SEED,
        "clip_length_seconds": 2.0, "max_video_length_clips": 75,
        "max_query_length": 32, "has_saliency_gt": False,
        "recommended_saliency_loss_weight": 0.0,
        "baseline_model": "Moment-DETR pretrained on QVHighlights",
        "feature_mode": "CLIP-only",
        "test_policy": "Sealed; final M0/M1 evaluation only. Train for optimization; Val for selection.",
    }
    target = root / "local_data" / "manifests" / f"{VERSION}_training_handoff.json"
    encoded = (json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
    if target.exists():
        if target.read_bytes() != encoded:
            raise ValueError("Existing training handoff differs; frozen v1 cannot be overwritten")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    return target


def freeze_dataset(root: Path = ROOT, *, expected_total: int = 48) -> dict:
    """Freeze only a fully resolved, provenance-verified pilot."""
    root = root.resolve()
    samples, audit, primary = _gate(root, expected_total)
    dataset_root = root / "local_data" / "datasets"
    final = dataset_root / VERSION
    if final.exists():
        manifest = _verify_existing(root, final, audit["source_hashes"])
        handoff = _handoff(root, final, manifest)
        return {"dataset_dir": str(final), "dataset_sha256": manifest["dataset_sha256"],
                "handoff_path": str(handoff), "reused_existing": True}
    splits, split_manifest = _split(samples)
    split_manifest["diagnostics"] = {name: _stats(rows) for name, rows in splits.items()}
    dataset_root.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f"{VERSION}.staging.", dir=dataset_root))
    try:
        _write_jsonl(stage / f"{VERSION}_all.jsonl", sorted(samples, key=lambda row: row["qid"]))
        for name, rows in splits.items():
            _write_jsonl(stage / f"{VERSION}_{name}.jsonl", rows)
        split_manifest["file_sha256"] = {
            name: _file_sha(stage / f"{VERSION}_{name}.jsonl") for name in ("train", "val", "test")}
        _write_json(stage / "split_manifest.json", split_manifest)
        audit.update({
            "total_pilot_count": len(primary), "included_count": len(samples),
            "excluded_count": len(primary) - len(samples),
            "split_video_overlap": {"train_val": [], "train_test": [], "val_test": []},
            "no_excluded_in_split": True, "included_appear_exactly_once": True,
            "saliency_gt_fabricated": False,
        })
        _write_json(stage / "annotation_audit.json", audit)
        freeze_time = datetime.now(timezone.utc).isoformat()
        dataset_sha = _file_sha(stage / f"{VERSION}_all.jsonl")
        manifest = {
            "dataset_name": "LUMAE Ads", "dataset_version": VERSION,
            "freeze_time_utc": freeze_time, "total_pilot_count": len(primary),
            "included_count": len(samples), "excluded_count": len(primary) - len(samples),
            "train_count": len(splits["train"]), "val_count": len(splits["val"]),
            "test_count": len(splits["test"]), "canonical_schema_version": "lumae_temporal_v1",
            "annotation_source": "LUMAE human temporal annotations",
            "query_review_provenance": "LUMAE human query review; historical human primary for pre-fast-lane samples",
            "temporal_review_provenance": "Explicit human primary or fast-lane decision",
            "semantic_v3_assistance_disclosure": "Frozen semantic_v3 ranking of video-derived advisory candidates; human verified all included GT",
            "semantic_v3_algorithm_sha256": deployment.ALGORITHM_SHA256,
            "source_video_dataset": "AdsQA", "query_creator": "LUMAE",
            "temporal_gt_creator": "LUMAE human reviewers",
            "has_saliency_gt": False, "dataset_sha256": dataset_sha,
            "source_artifact_sha256": audit["source_hashes"],
            "source_video_sha256": audit["source_video_sha256"],
            "split_seed": SEED,
        }
        _write_json(stage / "dataset_manifest.json", manifest)
        names = [f"{VERSION}_{name}.jsonl" for name in ("all", "train", "val", "test")]
        names += ["dataset_manifest.json", "split_manifest.json", "annotation_audit.json"]
        (stage / "checksums.sha256").write_text(
            "".join(f"{_file_sha(stage / name)}  {name}\n" for name in names), encoding="utf-8")
        if final.exists():
            raise ValueError("Frozen v1 appeared during export; refusing overwrite")
        os.rename(stage, final)
    finally:
        if stage.exists():
            if not stage.resolve().is_relative_to(dataset_root.resolve()):
                raise RuntimeError("Unsafe staging cleanup path")
            shutil.rmtree(stage)
    handoff = _handoff(root, final, manifest)
    return {"dataset_dir": str(final), "dataset_sha256": dataset_sha,
            "handoff_path": str(handoff), "reused_existing": False,
            "included": len(samples), "excluded": len(primary) - len(samples),
            "train": len(splits["train"]), "val": len(splits["val"]),
            "test": len(splits["test"])}


def freeze_if_complete(root: Path = ROOT, *, expected_total: int = 48) -> dict | None:
    primary = deployment._unique(deployment._read_csv(
        deployment._paths(root)[0] / "human_primary.csv"), "human_primary.csv")
    if len(primary) != expected_total or any(
        row["review_status"] not in {"REVIEWED", "EXCLUDED"} for row in primary.values()
    ):
        return None
    return freeze_dataset(root, expected_total=expected_total)
