"""Validate cache content before reuse; atomically persist new feature arrays."""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import zipfile
import numpy as np
from PIL import Image

from .models import digest


def visual_index_valid(record: dict, directory: Path, source_sha: str) -> bool:
    try:
        frames = record["frames"]
        if (record["video_sha256"] != source_sha or record["sampling_step_seconds"] != 2.0
                or len(frames) < 2 or not record["contact_sheets"]):
            return False
        for n, frame in enumerate(frames):
            p = directory / frame["path"]
            if frame["timestamp"] != min(record["duration"], n*2.0) or digest(p) != frame["sha256"]:
                return False
            with Image.open(p) as img: img.verify()
        for relative in record["contact_sheets"]:
            with Image.open(directory / relative) as img: img.verify()
        return True
    except (OSError, KeyError, ValueError, TypeError, json.JSONDecodeError):
        return False


def clip_npz_valid(path: Path, record: dict) -> bool:
    try:
        if not path.is_file() or path.stat().st_size == 0 or digest(path) != record.get("clip_features_sha256"):
            return False
        with np.load(path, allow_pickle=False) as data:
            features, times = data["features"], data["timestamps"]
            return (features.shape == (len(record["frames"]),512)
                    and np.issubdtype(features.dtype, np.floating)
                    and np.isfinite(features).all() and np.isfinite(times).all()
                    and np.array_equal(times, [f["timestamp"] for f in record["frames"]]))
    except (OSError, ValueError, KeyError, TypeError, EOFError, zipfile.BadZipFile):
        return False


def atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name+".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd,"wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush(); os.fsync(stream.fileno())
        with np.load(name,allow_pickle=False) as data:
            if set(data.files) != set(arrays) or any(not np.isfinite(data[k]).all() for k in data.files):
                raise ValueError("Invalid NPZ would be persisted")
        os.replace(name,path)
    finally:
        if os.path.exists(name): os.unlink(name)
