"""Deterministic AdsQA source order and resumable, verified downloads."""

from __future__ import annotations

import json
import os
import random
import subprocess
from pathlib import Path
from urllib.parse import urlparse

import requests

from .models import ROOT, SEED, WORK, atomic_json, digest, load_jsonl, read_json
from .repo_paths import to_repo_relative

SOURCE = ROOT / "local_data/raw/adsqa/source_manifest.json"
VIDEO_DIR = ROOT / "local_data/raw/adsqa/videos"
SELECTION = WORK / "source_selection.json"


def ordered_sources(seed: int = SEED, *, layout=None) -> list[dict]:
    d48 = {r["vid"].removeprefix("lumae_ads_") for r in
           load_jsonl(layout.base / f"{layout.base_version}_all.jsonl" if layout else
                      ROOT / "local_data/datasets/lumae_ads_v1/lumae_ads_v1_all.jsonl")}
    entries = read_json(layout.root / "local_data/raw/adsqa/source_manifest.json" if layout else SOURCE)["videos"]
    ids = [row["source_video_id"] for row in entries]
    if len(ids) != len(set(ids)):
        raise ValueError("AdsQA source manifest contains duplicate video IDs")
    rows = sorted((r for r in entries if r["source_video_id"] not in d48),
                  key=lambda r: r["source_video_id"])
    random.Random(seed).shuffle(rows)
    return rows


def selection(seed: int = SEED, *, layout=None) -> dict:
    source_hash = digest(layout.root / "local_data/raw/adsqa/source_manifest.json" if layout else SOURCE)
    output = layout.selection if layout else SELECTION
    if output.exists():
        record = read_json(output)
        if record["seed"] != seed or record["source_manifest_sha256"] != source_hash:
            raise ValueError("Selection seed or AdsQA manifest changed")
        if layout and record.get("base_dataset_sha256") != digest(layout.base / f"{layout.base_version}_all.jsonl"):
            raise ValueError("Frozen parent dataset changed")
        return record
    record = {"seed": seed, "source_manifest_sha256": source_hash,
              "candidates": [{**r, "status": "unattempted", "reused_D48": False,
                              "local_path": str(VIDEO_DIR / r["target_name"]),
                              "duration": None, "sha256": None, "rejection_reason": None}
                             for r in ordered_sources(seed, layout=layout)]}
    if layout:
        previous_path = layout.root / "local_data/intermediate" / layout.base_version / "source_selection.json"
        previous = read_json(previous_path)["candidates"] if previous_path.is_file() else []
        prior_rejects = {r["source_video_id"]: r.get("rejection_reason") for r in previous if r["status"] == "rejected"}
        record["inherited_terminal_rejections"] = prior_rejects
        record["candidates"] = [r for r in record["candidates"] if r["source_video_id"] not in prior_rejects]
        for r in record["candidates"]:
            r["local_path"] = to_repo_relative(layout.videos / r["target_name"], root=layout.root)
            r["reused_D200"] = False
            r["network_retry_count"] = r["io_retry_count"] = r["annotation_retry_count"] = 0
            r["failure_kind"] = None
        record.update(dataset_version=layout.version, target_new_accepted=layout.new_count,
                      base_dataset_sha256=digest(layout.base / f"{layout.base_version}_all.jsonl"))
    atomic_json(output, record)
    return record


def save_candidate(record: dict, video_id: str, *, layout=None, **updates: object) -> None:
    for candidate in record["candidates"]:
        if candidate["source_video_id"] == video_id:
            if layout and "status" in updates:
                from .state_machine import validate_transition
                current, target = candidate["status"], updates["status"]
                stages = ["unattempted", "downloading", "downloaded", "visual_index_ready",
                          "awaiting_visual_inspection", "annotation_candidate_ready", "awaiting_interval_verification"]
                # Resume reuses validated earlier stages without rewinding state.
                if current in stages and target in stages and stages.index(target) < stages.index(current):
                    updates.pop("status")
                else:
                    validate_transition(current, target, video_id)
            candidate.update(updates)
            atomic_json(layout.selection if layout else SELECTION, record)
            return
    raise KeyError(video_id)


def download(row: dict, timeout: int = 60, *, video_dir: Path | None = None) -> Path:
    path = (video_dir or VIDEO_DIR) / row["target_name"]
    if path.is_file() and path.stat().st_size > 0:
        return path
    url = row.get("source_url", "")
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname != "video.adsoftheworld.com":
        raise ValueError("Source URL is outside the approved AdsQA video host")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    with requests.get(url, headers=headers, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        if offset and response.status_code != 206:
            offset = 0  # The server ignored Range; replace the incomplete file.
        if offset and response.status_code == 206 and not response.headers.get(
            "Content-Range", "").startswith(f"bytes {offset}-"):
            raise ValueError("Resumed download has an unexpected byte range")
        with partial.open("ab" if offset else "wb") as output:
            for chunk in response.iter_content(1024 * 1024):
                if chunk:
                    output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
    if not partial.stat().st_size:
        raise ValueError("Empty video download")
    os.replace(partial, path)
    return path


def probe(path: Path) -> float:
    result = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                             "-of", "json", str(path)], capture_output=True, text=True, timeout=30)
    if result.returncode:
        raise ValueError(f"Unreadable video: {result.stderr[-250:]}")
    duration = float(json.loads(result.stdout)["format"]["duration"])
    if not 0 < duration < float("inf"):
        raise ValueError("Invalid video duration")
    return duration
