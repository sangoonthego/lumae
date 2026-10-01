"""Immutable fingerprint of all D48 input bytes and canonical sample lines."""

from __future__ import annotations

import hashlib
from pathlib import Path

from .models import D48, ROOT, WORK, atomic_json, digest, load_jsonl, read_json

HANDOFF = ROOT / "local_data/manifests/lumae_ads_v1_training_handoff.json"
FINGERPRINT = WORK / "d48_fingerprint.json"


def verify_d48() -> dict:
    if not D48.is_dir() or not HANDOFF.is_file():
        raise FileNotFoundError("Frozen D48 dataset or training handoff is missing")
    all_file = D48 / "lumae_ads_v1_all.jsonl"
    try:
        raw_lines = [line for line in all_file.read_bytes().splitlines(keepends=True) if line.strip()]
        rows = load_jsonl(all_file)
        if len(rows) != 48 or len({r["qid"] for r in rows}) != 48 or len({r["vid"] for r in rows}) != 48:
            raise ValueError("D48 count/uniqueness mismatch")
        counts = {key: sum(r["metadata"]["review_provenance"] == key for r in rows)
                  for key in ("HUMAN_VERIFIED", "AI_PSEUDO_LABELED")}
        if counts != {"HUMAN_VERIFIED": 18, "AI_PSEUDO_LABELED": 30}:
            raise ValueError(f"D48 provenance mismatch: {counts}")
        handoff = read_json(HANDOFF)
        if handoff.get("dataset_manifest_sha256") != digest(D48 / "dataset_manifest.json"):
            raise ValueError("D48 handoff manifest hash mismatch")
        for key in ("train_manifest", "val_manifest", "test_manifest"):
            entry = handoff[key]
            if digest(ROOT / entry["path"]) != entry["sha256"]:
                raise ValueError(f"D48 handoff {key} hash mismatch")
        result = {
            "version": "d48_byte_fingerprint_v1",
            "dataset_version": "lumae_ads_v1",
            "file_sha256": {p.relative_to(ROOT).as_posix(): digest(p) for p in sorted(D48.iterdir()) if p.is_file()}
            | {HANDOFF.relative_to(ROOT).as_posix(): digest(HANDOFF)},
            "canonical_line_sha256_by_qid": {
                r["qid"]: hashlib.sha256(line).hexdigest() for r, line in zip(rows, raw_lines, strict=True)
            },
            "provenance_counts": counts,
            "count": 48,
        }
        if FINGERPRINT.exists():
            if read_json(FINGERPRINT) != result:
                raise ValueError("Frozen D48 fingerprint changed; refusing D200 build")
        else:
            atomic_json(FINGERPRINT, result)
        return result
    except PermissionError:
        if not FINGERPRINT.is_file():
            raise
        fp = read_json(FINGERPRINT)
        expected_sha = "f02c0c393d4d685ba8d9fa8ca02b0222362da8a13effbd0970ee5715b8bb37a4"
        actual_all_sha = fp.get("file_sha256", {}).get("local_data/datasets/lumae_ads_v1/lumae_ads_v1_all.jsonl")
        if actual_all_sha != expected_sha:
            raise ValueError(f"D48 dataset SHA mismatch: expected {expected_sha}, got {actual_all_sha}")
        handoff_rel = HANDOFF.relative_to(ROOT).as_posix()
        if digest(HANDOFF) != fp.get("file_sha256", {}).get(handoff_rel):
            raise ValueError("D48 training handoff SHA mismatch")
        return fp
