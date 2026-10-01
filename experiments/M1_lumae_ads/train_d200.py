"""D200 adapter for the established CLIP-only Moment-DETR Colab runner.

This module only prepares/preflights data here. The training action is for a
later, explicit Colab experiment after the D200 freeze has completed.
"""

from __future__ import annotations

import math
import json
from collections import Counter
from pathlib import Path

from experiments.M1_lumae_ads import train_frozen as runner

VERSION = "lumae_ads_d200"
HANDOFF = Path("local_data/manifests/lumae_ads_d200_training_handoff.json")


def preflight(root: Path, *, check_videos: bool = True) -> tuple[dict, dict, dict]:
    root = root.resolve()
    handoff = runner._json(root / HANDOFF)
    if (handoff.get("dataset_version"), handoff.get("total_samples"),
            handoff.get("has_saliency_gt")) != (VERSION, 200, False):
        raise ValueError("D200 handoff is incomplete or inconsistent")
    manifest_path = root / handoff["dataset_manifest_path"]
    if runner.sha256(manifest_path) != handoff["dataset_manifest_sha256"]:
        raise ValueError("D200 dataset manifest changed")
    manifest = runner._json(manifest_path)
    if manifest.get("dataset_sha256") != handoff.get("dataset_sha256"):
        raise ValueError("D200 dataset hash mismatch")
    directory = manifest_path.parent
    checks = (directory / "checksums.sha256").read_text(encoding="utf-8").splitlines()
    if len(checks) < 7:
        raise ValueError("D200 checksum coverage incomplete")
    for line in checks:
        expected, name = line.split("  ", 1)
        if runner.sha256(directory / name) != expected:
            raise ValueError(f"D200 file changed: {name}")
    all_rows = [json.loads(line) for line in
        (directory / f"{VERSION}_all.jsonl").read_text(encoding="utf-8").splitlines() if line]
    if runner.sha256(directory / f"{VERSION}_all.jsonl") != manifest["dataset_sha256"]:
        raise ValueError("D200 canonical file changed")
    rows = {}
    for split in runner.SPLITS:
        spec = handoff[f"{split}_manifest"]
        path = root / spec["path"]
        if runner.sha256(path) != spec["sha256"]:
            raise ValueError(f"D200 {split} split changed")
        rows[split] = [json.loads(line) for line in
                       path.read_text(encoding="utf-8").splitlines() if line]
    if tuple(len(rows[x]) for x in runner.SPLITS) != (140, 30, 30):
        raise ValueError("D200 split must be 140/30/30")
    by_split = {name: {r["vid"] for r in group} for name, group in rows.items()}
    if any(by_split[a] & by_split[b] for a, b in
           (("train", "val"), ("train", "test"), ("val", "test"))):
        raise ValueError("D200 video leakage")
    all_qids = {r["qid"] for r in all_rows}
    if (len(all_rows), len(all_qids), len({r["vid"] for r in all_rows})) != (200, 200, 200):
        raise ValueError("D200 uniqueness/count mismatch")
    if all_qids != {r["qid"] for group in rows.values() for r in group}:
        raise ValueError("D200 split membership mismatch")
    d48 = {r["qid"] for r in all_rows if not r["qid"].startswith("lumae_ads_d200_")}
    if len(d48) != 48 or not d48 <= {r["qid"] for r in rows["train"]}:
        raise ValueError("D48 is not entirely train-only")
    expected = {"train": Counter({"HUMAN_VERIFIED": 18, "AI_PSEUDO_LABELED": 122}),
                "val": Counter({"AI_PSEUDO_LABELED": 30}),
                "test": Counter({"AI_PSEUDO_LABELED": 30})}
    for split, group in rows.items():
        if Counter(r["metadata"]["review_provenance"] for r in group) != expected[split]:
            raise ValueError("D200 split provenance mismatch")
    for row in all_rows:
        windows = row["relevant_windows"]
        if (not row["query"].strip() or len(windows) != 1 or len(windows[0]) != 2
                or not all(math.isfinite(float(v)) for v in (*windows[0], row["duration"]))
                or not 0 <= windows[0][0] < windows[0][1] <= row["duration"]
                or row["saliency_scores"] is not None or row["relevant_clip_ids"] is not None):
            raise ValueError(f"Invalid D200 temporal sample: {row['qid']}")
        if check_videos:
            video = root / runner.VIDEOS / row["metadata"]["video_filename"]
            if not video.is_file() or runner.sha256(video) != manifest["source_video_sha256"][row["qid"]]["sha256"]:
                raise ValueError(f"Missing or changed video: {row['qid']}")
    return handoff, manifest, rows


def _package(root: Path, destination: Path) -> dict:
    from data_process.lumae_scale.freeze import package_colab
    preflight(root)
    return package_colab(destination)


def main() -> None:
    runner.VERSION = VERSION
    runner.HANDOFF = HANDOFF
    runner.CONFIG = Path("experiments/M1_lumae_ads/config_d200.json")
    runner.preflight = preflight
    runner.package_colab = _package
    runner.main()


if __name__ == "__main__":
    main()
