"""Decode each source once and cache timestamped inspection evidence and CLIP features."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from contextlib import nullcontext
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .models import ROOT, VISUAL_CACHE, WORK, atomic_json, digest, read_json
from .repo_paths import to_repo_relative
from .cache_validation import visual_index_valid, clip_npz_valid, atomic_npz

STEP = 2.0
SHEET_SIZE = 10


def build_visual_index(video: Path, duration: float, video_sha256: str,
                       timer=None, *, work: Path | None = None, strict: bool = False) -> dict:
    video_id = video.stem
    out = VISUAL_CACHE / video_id
    metadata_path = out / "metadata.json"
    if metadata_path.exists():
        try:
            old = read_json(metadata_path)
        except (ValueError, OSError):
            if not strict: raise
            old = {"frames": [], "contact_sheets": [], "video_sha256": None}
        valid = visual_index_valid(old, out, video_sha256) if strict else (
            old["video_sha256"] == video_sha256 and
            all((out / item["path"]).is_file() for item in old["frames"]) and
            all((out / item).is_file() for item in old["contact_sheets"]))
        if strict and valid:
            from .stage_cache import frame_fingerprint
            valid = old.get("visual_fingerprint") == frame_fingerprint(old)
        if valid:
            if timer:
                timer.mark_cache("visual_index", True)
            return old
    if timer:
        timer.mark_cache("visual_index", False)
    frames_dir = out / "sampled_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    if strict:
        # Only a new-source cache reaches this path. Remove stale extra frames
        # before re-decoding with the unchanged 2-second sampling command.
        for p in frames_dir.glob("frame_*.jpg"):
            p.unlink()
    # fps=1/2 preserves the same coarse temporal grid used in CLIP-only training.
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(video),
               "-vf", "fps=1/2,scale=320:-2", "-q:v", "4",
               str(frames_dir / "frame_%04d.jpg")]
    with timer.stage("frame_extraction") if timer else nullcontext():
        result = subprocess.run(command, capture_output=True, text=True, timeout=300)
    if result.returncode:
        raise ValueError(f"Frame decode failed: {result.stderr[-300:]}")
    frames = sorted(frames_dir.glob("frame_*.jpg"))
    if len(frames) < 2 or len(frames) > math.ceil(duration / STEP) + 2:
        raise ValueError("Invalid number of decoded frames")
    records = [{"timestamp": min(duration, i * STEP),
                "path": p.relative_to(out).as_posix(), "sha256": digest(p)}
               for i, p in enumerate(frames)]
    with timer.stage("contact_sheet") if timer else nullcontext():
        sheets_dir = out / "contact_sheets"
        sheets_dir.mkdir(exist_ok=True)
        sheets = []
        for batch_idx, start in enumerate(range(0, len(records), SHEET_SIZE)):
            batch = records[start:start + SHEET_SIZE]
            sheet = Image.new("RGB", (640, 5 * 208), "#171717")
            draw = ImageDraw.Draw(sheet)
            for j, item in enumerate(batch):
                with Image.open(out / item["path"]) as source:
                    thumbnail = source.convert("RGB")
                    thumbnail.thumbnail((310, 175))
                    x, y = (j % 2) * 320 + 5, (j // 2) * 208 + 5
                    sheet.paste(thumbnail, (x, y))
                    draw.text((x, y + 178), f"{item['timestamp']:.1f}s", fill="white")
            sheet_path = sheets_dir / f"contact_sheet_{batch_idx:03d}.jpg"
            sheet.save(sheet_path, quality=86)
            sheets.append(sheet_path.relative_to(out).as_posix())
    record = {"video_id": video_id, "video_path": to_repo_relative(video), "video_sha256": video_sha256,
              "duration": duration, "sampling_step_seconds": STEP, "frames": records,
              "contact_sheets": sheets, "feature_extractor": "CLIP ViT-B/32",
              "clip_features_status": "pending"}
    if strict:
        from .stage_cache import frame_fingerprint
        record["visual_fingerprint"] = frame_fingerprint(record)
    atomic_json(metadata_path, record)
    debug = (work or WORK) / "visual_index" / video_id
    atomic_json(debug / "metadata.json", record)
    atomic_json(debug / "sampled_frames.json", records)
    return record


def extract_clip_features(video_id: str, *, model_path: str = "ViT-B/32",
                          device: str = "cpu", batch_size: int = 16,
                          timer=None, clip_runtime=None, strict: bool = False) -> dict:
    """Cache normalized official OpenAI CLIP embeddings; load weights explicitly."""
    out = VISUAL_CACHE / video_id
    metadata_path = out / "metadata.json"
    record = read_json(metadata_path)
    feature_path = out / "clip_features.npz"
    identity_valid = True
    if strict:
        from .stage_cache import clip_fingerprint
        identity_valid = record.get("clip_fingerprint") == clip_fingerprint(record)
    if (feature_path.exists() and record.get("clip_features_sha256") == digest(feature_path)
            and identity_valid and (not strict or clip_npz_valid(feature_path,record))):
        if timer:
            timer.mark_cache("clip", True)
        return record
    if timer:
        timer.mark_cache("clip", False)
    with timer.stage("clip") if timer else nullcontext():
        return _extract_clip_features(record, out, metadata_path, feature_path,
                                      model_path, device, batch_size,
                                      clip_runtime=clip_runtime)


def _extract_clip_features(record: dict, out: Path, metadata_path: Path,
                           feature_path: Path, model_path: str, device: str,
                           batch_size: int, clip_runtime=None) -> dict:
    from .clip_runtime import get_clip_runtime

    runtime = clip_runtime or get_clip_runtime(model_path, device=device)
    paths = [out / item["path"] for item in record["frames"]]
    features = runtime.encode_images(paths, batch_size=batch_size, normalize=True)
    if features.shape != (len(paths), 512):
        raise ValueError("Unexpected CLIP feature dimensions")
    atomic_npz(feature_path, features=features,
               timestamps=np.array([x["timestamp"] for x in record["frames"]]))
    record.update(clip_features_status="complete", clip_features_sha256=digest(feature_path),
                  clip_model=model_path, clip_device=device)
    if record.get("visual_fingerprint"):
        from .stage_cache import clip_fingerprint
        record["clip_fingerprint"] = clip_fingerprint(record)
    atomic_json(metadata_path, record)
    return record


def clip_consistency(video_id: str, query: str, start: float, end: float,
                     *, model_path: str = "ViT-B/32", device: str = "cpu",
                     clip_runtime=None, query_vector: np.ndarray | None = None) -> dict:
    from .clip_runtime import get_clip_runtime

    record = read_json(VISUAL_CACHE / video_id / "metadata.json")
    if record.get("clip_features_status") != "complete":
        raise ValueError("CLIP visual index is incomplete")
    array = np.load(VISUAL_CACHE / video_id / "clip_features.npz")
    features = array["features"]
    times = array["timestamps"]

    if query_vector is not None:
        vector = query_vector
    else:
        runtime = clip_runtime or get_clip_runtime(model_path, device=device)
        vector = runtime.encode_text(query)[0]

    scores = features @ vector
    inside = (times >= start) & (times <= end)
    if not inside.any() or inside.all():
        raise ValueError("Interval lacks inside or outside CLIP frames")
    return {"best_inside_similarity": float(scores[inside].max()),
            "mean_inside_similarity": float(scores[inside].mean()),
            "best_outside_similarity": float(scores[~inside].max()),
            "inside_minus_outside_margin": float(scores[inside].max() - scores[~inside].max()),
            "query_frame_rank": int(np.argsort(-scores).tolist().index(int(np.argmax(np.where(inside, scores, -999)))) + 1),
            "notice": "CLIP similarities are uncalibrated audit signals"}
