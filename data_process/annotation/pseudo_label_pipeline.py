"""Explicit AI-only fast track from A.2.6 to a mixed-provenance v1 freeze.

Human primary and query review CSVs are read-only inputs. Motion-derived candidate
windows and frozen semantic_v3 ranking produce unverified training pseudo labels.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from data_process.adsqa.semantic_preannotator_v3 import (
    SemanticCandidateEvent, decompose_query, rank_and_select_candidates,
)
from data_process.annotation import assisted_deployment as deployment
from data_process.annotation.fastlane import _create_immutable_json, _file_sha
from data_process.annotation.fastlane_freeze import (
    FORCED_TRAIN, ROOT, SEED, VERSION, _split, _stats, _write_json, _write_jsonl,
)
from data_process.annotation.human_query_review import (
    INVISIBLE_CLAIM, NATURAL_LANGUAGE, TIMESTAMP,
)
from data_process.validation.schema import validate_canonical_dict


SOURCE_NAMES = (
    "local_data/annotations/lumae_ads/human_primary.csv",
    "local_data/annotations/lumae_ads/ai_query_suggestions.csv",
    "local_data/annotations/lumae_ads/query_reviews.csv",
)


def _sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _inputs(root: Path) -> tuple[dict, dict, dict]:
    deployment.verify_frozen_ranker(root)
    errors = deployment.frozen_artifact_errors(root)
    if errors:
        raise ValueError("Frozen artifact integrity failed: " + "; ".join(errors))
    ann = deployment._paths(root)[0]
    primary = deployment._unique(deployment._read_csv(ann / "human_primary.csv"), "human_primary.csv")
    suggestions = deployment._unique(deployment._read_csv(ann / "ai_query_suggestions.csv"), "ai_query_suggestions.csv")
    reviewed = {sid for sid, row in primary.items() if row["review_status"] == "REVIEWED"}
    draft = {sid for sid, row in primary.items() if row["review_status"] == "DRAFT"}
    if len(primary) != 48 or len(reviewed) != 18 or len(draft) != 30 or set(suggestions) != draft:
        raise ValueError("Expected 48 primary rows: 18 reviewed and exactly 30 suggested DRAFT rows")
    if not FORCED_TRAIN <= set(primary):
        raise ValueError("A forced-train sample is missing")
    hashes = {name: _file_sha(root / name) for name in SOURCE_NAMES}
    return primary, suggestions, hashes


def _pseudo_path(root: Path, sid: str) -> Path:
    return root / "local_data" / "manifests" / "a26_auto_pseudo_labels" / f"{sid}.json"


def _check_prediction(record: dict, row: dict, suggestion: dict, video_hash: str) -> None:
    query = suggestion["ai_suggested_query"].strip()
    duration = float(row["duration_seconds"])
    required = {
        "sample_id": row["sample_id"], "video_filename": row["video_filename"],
        "original_query": row["query"], "ai_suggested_query": query,
        "original_query_quality": suggestion["original_query_quality"],
        "query_change_type": suggestion["query_change_type"],
        "query_source": "AI_GENERATED", "query_review_status": "AI_AUTO_ACCEPTED_FOR_TRAINING",
        "reviewer_id": "AI_PIPELINE", "temporal_label_source": "SEMANTIC_V3_PSEUDO_LABEL",
        "review_provenance": "AI_PSEUDO_LABELED",
        "semantic_v3_algorithm_sha256": deployment.ALGORITHM_SHA256,
        "query_sha256": _sha_text(query), "video_sha256": video_hash,
    }
    if any(record.get(key) != value for key, value in required.items()):
        raise ValueError(f"Pseudo-label provenance changed: {row['sample_id']}")
    start, end = record.get("pred_start_seconds"), record.get("pred_end_seconds")
    if not (isinstance(start, (int, float)) and isinstance(end, (int, float))
            and math.isfinite(start) and math.isfinite(end) and 0 <= start < end <= duration):
        raise ValueError(f"Invalid pseudo interval: {row['sample_id']}")
    if not record.get("generated_at_utc") or record.get("status") != "AI_PSEUDO_LABEL_FOR_TRAINING":
        raise ValueError(f"Incomplete pseudo-label record: {row['sample_id']}")


def _low_memory_video_candidates(video: Path, duration: float, evidence: str) -> tuple[list, dict]:
    """Use the same fixed frame-change proposal rule on scaled decoded video frames."""
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-threads", "1",
               "-filter_threads", "1", "-i", str(video), "-vf", "fps=1,scale=64:36,format=gray",
               "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1"]
    completed = subprocess.run(command, capture_output=True, timeout=180, check=False)
    if completed.returncode:
        raise ValueError(f"FFmpeg decode failed for {video.name}: {completed.stderr.decode(errors='replace')[-500:]}")
    frame_size = 64 * 36
    data = completed.stdout
    if len(data) % frame_size or len(data) < 2 * frame_size:
        raise ValueError(f"Insufficient scaled frames for {video.name}")
    frames = [data[i:i + frame_size] for i in range(0, len(data), frame_size)]
    observations = [(float(i), sum(abs(a - b) for a, b in zip(frames[i], frames[i - 1])) / frame_size)
                    for i in range(1, min(len(frames), math.ceil(duration)))]
    selected: list[tuple[float, float]] = []
    for center, score in sorted(observations, key=lambda item: (-item[1], item[0])):
        if all(abs(center - old) >= 3.0 for old, _ in selected):
            selected.append((center, score))
            if len(selected) == 5:
                break
    candidates = []
    for idx, (center, motion) in enumerate(selected):
        # The frozen candidate type rounds effective windows to tenths. Floor
        # the source duration first so that rounding cannot cross its boundary.
        start = max(0.0, round(center - 2.5, 1))
        end = min(math.floor(duration * 10) / 10, round(center + 2.5, 1))
        if end - start < 0.1:
            continue
        candidates.append(SemanticCandidateEvent(
            candidate_id=chr(ord("A") + idx), start_coarse=start, end_coarse=end,
            visible_subject="video scene", visible_action="visible frame change",
            visible_object="objects in the local video",
            supporting_visual_evidence=(
                f"Local video frame-change magnitude {motion:.3f} at {center:.1f}s. "
                f"Whole-video AI visual evidence (not localized): {evidence}"),
        ))
    if not candidates:
        raise ValueError(f"No video-derived candidate intervals for {video.name}")
    return candidates, {"method": "ffmpeg_scaled_frame_difference_v1", "frame_step_seconds": 1.0,
                        "sampled_frame_count": len(frames),
                        "motion_peak_times_seconds": [center for center, _ in selected]}


def generate_one(root: Path, row: dict, suggestion: dict) -> dict:
    sid = row["sample_id"]
    if (row["review_status"] != "DRAFT" or suggestion["sample_id"] != sid
            or suggestion["status"] != "AI_SUGGESTED"):
        raise ValueError(f"Invalid DRAFT/suggestion pairing: {sid}")
    if row["video_filename"] != suggestion["video_filename"] or row["query"] != suggestion["original_query"]:
        raise ValueError(f"Suggestion differs from primary metadata: {sid}")
    query = suggestion["ai_suggested_query"].strip()
    if (not query or not NATURAL_LANGUAGE.search(query) or TIMESTAMP.search(query)
            or INVISIBLE_CLAIM.search(query)
            or suggestion["query_localizable"].upper() != "TRUE"
            or suggestion["query_observable"].upper() != "TRUE"):
        raise ValueError(f"AI suggestion is not eligible for automatic training acceptance: {sid}")
    video = deployment._paths(root)[2] / row["video_filename"]
    if (not video.is_file() or row["source_video_id"] != video.stem
            or row["vid"] != f"lumae_ads_{video.stem}"):
        raise ValueError(f"Missing or mismatched video: {sid}")
    video_hash = _file_sha(video)
    path = _pseudo_path(root, sid)
    if path.exists():
        record = json.loads(path.read_text(encoding="utf-8"))
        _check_prediction(record, row, suggestion, video_hash)
        return record
    duration = float(row["duration_seconds"])
    candidates, proposal = _low_memory_video_candidates(video, duration, suggestion["visual_evidence"])
    top, margin, confidence, reason = rank_and_select_candidates(decompose_query(query), candidates)
    start, end = top.get_effective_window()
    record = {
        "sample_id": sid, "video_filename": row["video_filename"],
        "original_query": row["query"], "ai_suggested_query": query,
        "original_query_quality": suggestion["original_query_quality"],
        "query_change_type": suggestion["query_change_type"],
        "query_source": "AI_GENERATED", "query_review_status": "AI_AUTO_ACCEPTED_FOR_TRAINING",
        "reviewer_id": "AI_PIPELINE", "temporal_label_source": "SEMANTIC_V3_PSEUDO_LABEL",
        "review_provenance": "AI_PSEUDO_LABELED",
        "semantic_v3_algorithm_sha256": deployment.ALGORITHM_SHA256,
        "query_sha256": _sha_text(query), "video_sha256": video_hash,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "AI_PSEUDO_LABEL_FOR_TRAINING",
        "pred_start_seconds": start, "pred_end_seconds": end,
        "candidate_windows": [list(c.get_effective_window()) for c in candidates],
        "candidate_scores": [{"candidate_id": c.candidate_id, "score": c.query_match_score}
                             for c in candidates],
        "selected_candidate_id": top.candidate_id, "ranking_margin": margin,
        "semantic_confidence": confidence, "semantic_reason": reason,
        "proposal": proposal,
        "score_notice": "Frame-change candidates and semantic_v3 ranking are unverified, uncalibrated pseudo labels.",
    }
    _check_prediction(record, row, suggestion, video_hash)
    _create_immutable_json(path, record)
    return record


def _canonical(row: dict, prediction: dict | None) -> dict:
    sid = row["sample_id"]
    duration = float(row["duration_seconds"])
    if prediction is None:
        if row["review_status"] != "REVIEWED" or not row["annotator_id"].startswith("human_"):
            raise ValueError(f"Invalid human provenance: {sid}")
        query = row["query"].strip()
        start, end = float(row["gt_start_seconds"]), float(row["gt_end_seconds"])
        provenance = {"query_source": "HUMAN_VERIFIED", "query_review_status": "HUMAN_REVIEWED",
                      "reviewer_id": row["annotator_id"], "temporal_label_source": "HUMAN_GT",
                      "review_provenance": "HUMAN_VERIFIED"}
    else:
        query = prediction["ai_suggested_query"]
        start, end = prediction["pred_start_seconds"], prediction["pred_end_seconds"]
        provenance = {key: prediction[key] for key in (
            "query_source", "query_review_status", "reviewer_id", "temporal_label_source",
            "review_provenance", "original_query", "ai_suggested_query", "original_query_quality",
            "query_change_type", "semantic_v3_algorithm_sha256", "query_sha256", "generated_at_utc")}
        provenance["pseudo_label_artifact"] = f"local_data/manifests/a26_auto_pseudo_labels/{sid}.json"
        provenance["pseudo_label_artifact_sha256"] = prediction["artifact_sha256"]
    if not (query and all(map(math.isfinite, (start, end, duration))) and 0 <= start < end <= duration):
        raise ValueError(f"Invalid canonical interval/query: {sid}")
    sample = {
        "qid": sid, "vid": row["vid"], "query": query, "duration": duration,
        "relevant_windows": [[start, end]], "relevant_clip_ids": None,
        "saliency_scores": None, "source": "Lumae Ads", "intent": "product_demo",
        "original_id": sid,
        "metadata": {**provenance, "video_filename": row["video_filename"],
                     "product_category": row.get("product_category", ""),
                     "source_video_dataset": "AdsQA"},
    }
    valid, errors = validate_canonical_dict(sample)
    if not valid:
        raise ValueError(f"Canonical validation failed for {sid}: {errors}")
    return sample


def _verify_existing(root: Path, final: Path, source_hashes: dict, video_hashes: dict) -> dict:
    manifest = json.loads((final / "dataset_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("source_artifact_sha256") != source_hashes or manifest.get("source_video_sha256") != video_hashes:
        raise ValueError("Frozen v1 inputs changed; refusing overwrite")
    if (manifest.get("total_samples"), manifest.get("human_verified_samples"),
            manifest.get("ai_pseudo_labeled_samples")) != (48, 18, 30):
        raise ValueError("Frozen v1 provenance counts changed")
    checks = (final / "checksums.sha256").read_text(encoding="utf-8").splitlines()
    if len(checks) != 7:
        raise ValueError("Frozen v1 checksum coverage is incomplete")
    for line in checks:
        digest, name = line.split("  ", 1)
        if not (final / name).is_file() or _file_sha(final / name) != digest:
            raise ValueError(f"Frozen v1 checksum mismatch: {name}")
    if _file_sha(final / f"{VERSION}_all.jsonl") != manifest.get("dataset_sha256"):
        raise ValueError("Frozen v1 dataset hash mismatch")
    audit = json.loads((final / "annotation_audit.json").read_text(encoding="utf-8"))
    if len(audit.get("pseudo_label_artifacts", {})) != 30:
        raise ValueError("Frozen v1 pseudo-label audit is incomplete")
    for item in audit["pseudo_label_artifacts"].values():
        path = root / item["path"]
        if not path.is_file() or _file_sha(path) != item["sha256"]:
            raise ValueError(f"Frozen v1 pseudo-label artifact changed: {item['path']}")
    return manifest


def _handoff(root: Path, final: Path, manifest: dict) -> Path:
    rel = final.relative_to(root).as_posix()
    files = {name: {"path": f"{rel}/{VERSION}_{name}.jsonl",
                    "sha256": _file_sha(final / f"{VERSION}_{name}.jsonl")}
             for name in ("train", "val", "test")}
    payload = {
        "dataset_version": VERSION,
        "dataset_manifest_path": f"{rel}/dataset_manifest.json",
        "dataset_manifest_sha256": _file_sha(final / "dataset_manifest.json"),
        "train_manifest": files["train"], "val_manifest": files["val"], "test_manifest": files["test"],
        "clip_length_seconds": 2.0, "max_video_length_clips": 75,
        "max_query_length": 32, "has_saliency_gt": False,
        "recommended_saliency_loss_weight": 0.0,
        "baseline_model": "Moment-DETR pretrained on QVHighlights", "feature_mode": "CLIP-only",
        "label_provenance_counts": {"human_verified": manifest["human_verified_samples"],
                                    "ai_pseudo_labeled": manifest["ai_pseudo_labeled_samples"]},
        "test_policy": "Sealed; evaluate only after model selection on Val.",
    }
    path = root / "local_data" / "manifests" / f"{VERSION}_training_handoff.json"
    encoded = (json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
    if path.exists():
        if path.read_bytes() != encoded:
            raise ValueError("Existing v1 handoff differs; refusing overwrite")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    return path


def run(root: Path = ROOT) -> dict:
    """Resolve all 30 AI rows, then atomically freeze all 48 or report failures."""
    root = root.resolve()
    primary, suggestions, source_hashes = _inputs(root)
    predictions: dict[str, dict] = {}
    failures: dict[str, str] = {}
    for sid in sorted(suggestions):
        try:
            record = generate_one(root, primary[sid], suggestions[sid])
            record = {**record, "artifact_sha256": _file_sha(_pseudo_path(root, sid))}
            predictions[sid] = record
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
            failures[sid] = str(exc)
    if failures:
        return {"status": "PARTIAL", "success": len(predictions), "failures": failures}
    if any(_file_sha(root / name) != digest for name, digest in source_hashes.items()):
        raise ValueError("Source CSV changed during pseudo-label generation")
    samples = [_canonical(row, predictions.get(sid)) for sid, row in sorted(primary.items())]
    if len(samples) != 48 or len({r["qid"] for r in samples}) != 48:
        raise ValueError("Canonical sample count or uniqueness failure")
    video_hashes = {sid: {"video_filename": row["video_filename"],
                          "sha256": predictions[sid]["video_sha256"] if sid in predictions
                          else _file_sha(deployment._paths(root)[2] / row["video_filename"])}
                    for sid, row in sorted(primary.items())}
    dataset_root = root / "local_data" / "datasets"
    final = dataset_root / VERSION
    if final.exists():
        manifest = _verify_existing(root, final, source_hashes, video_hashes)
        handoff = _handoff(root, final, manifest)
        return {"status": "COMPLETE", "dataset_dir": str(final), "handoff_path": str(handoff),
                "manifest": manifest, "reused_existing": True}
    splits, split_manifest = _split(samples)
    if any(s["qid"] in FORCED_TRAIN for name in ("val", "test") for s in splits[name]):
        raise ValueError("Forced train sample leaked")
    split_manifest["diagnostics"] = {name: _stats(rows) for name, rows in splits.items()}
    split_manifest["counts"] = {name: len(rows) for name, rows in splits.items()}
    dataset_root.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f"{VERSION}.staging.", dir=dataset_root))
    try:
        _write_jsonl(stage / f"{VERSION}_all.jsonl", samples)
        for name, rows in splits.items():
            _write_jsonl(stage / f"{VERSION}_{name}.jsonl", rows)
        split_manifest["file_sha256"] = {
            name: _file_sha(stage / f"{VERSION}_{name}.jsonl") for name in splits}
        _write_json(stage / "split_manifest.json", split_manifest)
        audit = {
            "total_samples": 48, "human_verified_samples": 18, "ai_pseudo_labeled_samples": 30,
            "human_primary_sha256_unchanged": _file_sha(root / SOURCE_NAMES[0]) == source_hashes[SOURCE_NAMES[0]],
            "query_reviews_sha256_unchanged": _file_sha(root / SOURCE_NAMES[2]) == source_hashes[SOURCE_NAMES[2]],
            "all_windows_valid": True, "video_split_overlap": False,
            "forced_train_ids": sorted(FORCED_TRAIN),
            "pseudo_label_artifacts": {sid: {"path": f"local_data/manifests/a26_auto_pseudo_labels/{sid}.json",
                                              "sha256": pred["artifact_sha256"]}
                                       for sid, pred in sorted(predictions.items())},
        }
        _write_json(stage / "annotation_audit.json", audit)
        manifest = {
            "dataset_name": "LUMAE Ads", "dataset_version": VERSION,
            "freeze_time_utc": datetime.now(timezone.utc).isoformat(),
            "total_samples": 48, "human_verified_samples": 18, "ai_pseudo_labeled_samples": 30,
            "train_count": len(splits["train"]), "val_count": len(splits["val"]),
            "test_count": len(splits["test"]), "dataset_sha256": _file_sha(stage / f"{VERSION}_all.jsonl"),
            "split_seed": SEED, "canonical_schema_version": "lumae_temporal_v1",
            "annotation_source": "Mixed: 18 human verified GT; 30 unverified AI pseudo labels",
            "pseudo_label_disclosure": "The 30 AI intervals use frame-change video candidates ranked by frozen semantic_v3. Scores are uncalibrated; intervals have no human temporal review and are training pseudo labels only.",
            "semantic_v3_algorithm_sha256": deployment.ALGORITHM_SHA256,
            "has_saliency_gt": False, "source_artifact_sha256": source_hashes,
            "source_video_sha256": video_hashes,
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
    return {"status": "COMPLETE", "dataset_dir": str(final), "handoff_path": str(handoff),
            "manifest": manifest, "reused_existing": False}


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, ensure_ascii=False))
