"""Read and validate the frozen D1500 corpus and feature bindings."""

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile

from data_process.lumae_scale.models import ROOT, digest, load_jsonl, read_json

VERSION = "LUMAE_D1500_TEST450_AI_V1"
OUTPUT = ROOT / "research_outputs/d1500_test450_ai_v1"
FROZEN = ROOT / "research_outputs/d1500_split_freeze_20261110_01"
UPLOAD = ROOT / "research_outputs/d1500_colab_audit_20261110_01/d1500_colab_upload_manifest.json"
SOURCE = ROOT / "research_outputs/p0_20261103_01"
CANONICAL = "f386b4e3e3a717dbd151f67abdce71e6366123c760753b7687f766ade7b38e6a"
SEAL = "ff6d288a5a1a393483ff0999bca232ce66b14107b1b882073bee7df9bea914f3"
INVENTORY = "cb0e83517d904e65203af7d349cb8ad329e803ce41fd20308ba0962f6187917e"
M0 = "ad5facec03008800536466c7d33f9232c3db2b71a1bd445073957bd6f4d0edbf"
COUNTS = {"train": 1200, "val": 150, "test": 150, "legacy_test30": 30, "new_test120": 120}
KEYS = {"video": "features", "text": "last_hidden_state"}


def storage_path(value):
    """Map the historical E:/LUMAE layout onto a staged storage root."""
    raw = str(value).replace("\\", "/")
    prefix = "E:/LUMAE/"
    if raw.casefold().startswith(prefix.casefold()):
        return Path(os.environ.get("LUMAE_D1500_STORAGE_ROOT", "E:/LUMAE")) / raw[len(prefix):]
    return ROOT / raw


def safe_id(value):
    value = str(value)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", value):
        raise ValueError(f"Invalid identifier: {value!r}")
    return value


def require_hash(path, expected):
    if digest(path) != expected:
        raise ValueError(f"SHA-256 mismatch: {path}")


def valid_window(window, duration):
    return (
        isinstance(window, (tuple, list)) and len(window) == 2
        and all(type(x) in (float, int) and math.isfinite(x) for x in (*window, duration))
        and 0 <= window[0] < window[1] <= duration
    )


def duration_distribution(rows):
    buckets = {"<=2": 0, "(2,4]": 0, "(4,6]": 0, ">6": 0}
    for row in rows:
        start, end = row["relevant_windows"][0]
        length = end - start
        key = next((label for limit, label in ((2, "<=2"), (4, "(2,4]"), (6, "(4,6]"))
                    if length <= limit), ">6")
        buckets[key] += 1
    return buckets


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def jsonl_bytes(rows):
    return b"".join(map(json_bytes, rows))


def save_bytes(path, data):
    """Publish an immutable file atomically. Identical retries are harmless."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()
            try:
                if os.name == "nt":
                    os.rename(temporary, path)  # Windows rename fails if the destination exists.
                else:
                    os.link(temporary, path)
            except FileExistsError:
                if path.read_bytes() != data:
                    raise ValueError(f"Immutable artifact differs: {path}") from None
        finally:
            handle.close()
            temporary.unlink(missing_ok=True)


def save_json(path, value):
    save_bytes(path, json_bytes(value))


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


@dataclass(frozen=True)
class Feature:
    kind: str
    ident: str
    path: Path
    sha256: str

    def array(self, verify=True):
        import numpy as np

        if verify:
            require_hash(self.path, self.sha256)
        key = KEYS[self.kind]
        with np.load(self.path, allow_pickle=False) as data:
            if data.files != [key]:
                raise ValueError(f"Wrong NPZ keys: {self.path}")
            array = data[key]
        if array.ndim != 2 or array.shape[1] != 512 or not len(array) or array.dtype.kind != "f":
            raise ValueError(f"Wrong tensor contract: {self.path}")
        if not np.isfinite(array).all() or (self.kind == "text" and not 2 <= len(array) <= 32):
            raise ValueError(f"Invalid values or text length: {self.path}")
        return array


class TrainingParent:
    """Validate only the frozen Train/Val files for model selection."""

    def __init__(self, frozen=FROZEN, feature_inventory=UPLOAD, *, all_splits=False):
        self.frozen, self.feature_inventory = frozen, feature_inventory
        self.seed = 20261110
        splits = COUNTS if all_splits else ("train", "val")
        require_hash(frozen / "SHA256SUMS.txt", SEAL)
        self.hashes = {}
        for line in (frozen / "SHA256SUMS.txt").read_text().splitlines():
            expected, filename = line.split()
            if all_splits or filename in ("train.jsonl", "val.jsonl"):
                require_hash(frozen / filename, expected)
                self.hashes[filename] = expected
        self.rows = {split: load_jsonl(frozen / f"{split}.jsonl") for split in splits}

    def features(self, splits=("train", "val")):
        rows = [row for split in splits for row in self.rows[split]]
        expected = {"video": {r["vid"] for r in rows}, "text": {str(r["qid"]) for r in rows}}
        require_hash(self.feature_inventory, INVENTORY)
        features = {"video": {}, "text": {}}
        for item in read_json(self.feature_inventory)["entries"]:
            if item["category"] != "shared_cached_features":
                continue
            path = storage_path(item["local_path"])
            kind = "text" if path.parent.name == "clip_text_features" else "video"
            ident = item["logical_artifact_name"]
            if ident in expected[kind]:
                if ident in features[kind]:
                    raise ValueError(f"Duplicate feature binding: {kind} {ident}")
                features[kind][ident] = Feature(kind, ident, path, item["sha256"])
        if any(set(features[kind]) != ids for kind, ids in expected.items()):
            raise ValueError("Frozen feature inventory does not cover the requested rows")
        return features


class Parent(TrainingParent):
    def __init__(self, frozen=FROZEN, feature_inventory=UPLOAD):
        super().__init__(frozen, feature_inventory, all_splits=True)
        manifest = read_json(frozen / "split_manifest.json")
        if self.seed != manifest["seed"] or manifest["dataset_source"]["sha256"] != CANONICAL:
            raise ValueError("Wrong parent dataset or split seed")
        canonical = ROOT / manifest["dataset_source"]["path"]
        require_hash(canonical, CANONICAL)
        all_rows = self.rows["train"] + self.rows["val"] + self.rows["test"]
        for key in ("vid", "qid"):
            if len({safe_id(row[key]) for row in all_rows}) != len(all_rows):
                raise ValueError(f"Duplicate original {key}")
        if {r["qid"]: r for r in load_jsonl(canonical)} != {r["qid"]: r for r in all_rows}:
            raise ValueError("Canonical annotations differ from frozen splits")
        for split, rows in self.rows.items():
            membership = {row["vid"]: row["qid"] for row in rows}
            if len(rows) != COUNTS[split] or membership != manifest["membership"][split]["qid_by_vid"]:
                raise ValueError(f"Wrong frozen membership: {split}")
            for row in rows:
                windows = row["relevant_windows"]
                if len(windows) != 1 or not valid_window(windows[0], row["duration"]) or not row["query"].strip():
                    raise ValueError(f"Invalid original annotation: {row['qid']}")
        subsets = self.rows["legacy_test30"] + self.rows["new_test120"]
        if sorted(subsets, key=lambda r: str(r["qid"])) != sorted(self.rows["test"], key=lambda r: str(r["qid"])):
            raise ValueError("Frozen test subset mismatch")
        self.test = {row["vid"]: row for row in self.rows["test"]}
        self.qids = {str(row["qid"]) for row in all_rows}
        self.vids = {row["vid"] for row in all_rows}

    def media(self, vid):
        meta = self.test[vid]["metadata"]
        path = storage_path(meta["source_video_path"])
        require_hash(path, meta["source_video_sha256"])
        return path

    def audit(self):
        def probe(vid):
            result = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                 "stream=codec_type", "-of", "csv=p=0", str(self.media(vid))],
                check=True, capture_output=True, text=True,
            )
            if result.stdout.strip() != "video":
                raise ValueError(f"No video stream: {vid}")
            return vid

        def inspect(feature):
            return f"{feature.kind}/{feature.array().dtype}"

        features = self.features(("train", "val", "test"))
        with ThreadPoolExecutor(max_workers=8) as pool:
            media = list(pool.map(probe, sorted(self.test)))
            dtypes = Counter(pool.map(inspect, (f for group in features.values() for f in group.values())))
        require_hash(SOURCE / "assets/M0_pretrained.ckpt", M0)
        return {
            "status": "PASS", "parent_canonical_sha256": CANONICAL, "split_seed": self.seed,
            "split_sha256": self.hashes, "counts": COUNTS, "video_ids": len(self.vids),
            "query_ids": len(self.qids), "test_media_verified": len(media),
            "feature_inventory_sha256": INVENTORY, "feature_dtypes": dict(dtypes),
            "m0_sha256": M0, "duration_buckets": {s: duration_distribution(r) for s, r in self.rows.items()},
        }
