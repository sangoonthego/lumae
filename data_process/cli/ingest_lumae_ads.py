"""CLI tool for ingesting Lumae Product-Ads annotations from CSV or JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from ..adapters.lumae_ads import LumaeAdsAdapter
from ..io.jsonl import write_jsonl
from ..splitting.video_level import deterministic_video_split


def ingest_lumae_ads(
    input_path: Path | str,
    output_dir: Path | str,
    split: bool = False,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    video_root: Path | str | None = None,
) -> dict[str, Any]:
    src = Path(input_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    adapter = LumaeAdsAdapter(
        check_video_exists=bool(video_root),
        video_root=video_root,
    )

    if src.suffix.lower() == ".csv":
        samples, rejections = adapter.process_csv(src)
    else:
        samples, rejections = adapter.process_file(src)

    print(f"Ingested {len(samples):,} valid Lumae Ads samples ({len(rejections):,} rejected).")

    if rejections:
        rej_path = out / "rejected_records.jsonl"
        write_jsonl(rej_path, rejections)
        print(f"Saved rejections to: {rej_path}")

    stats: dict[str, Any] = {
        "total_valid_samples": len(samples),
        "total_rejected": len(rejections),
    }

    if split and len(samples) >= 3:
        train, val, test, split_stats = deterministic_video_split(
            samples,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
        )
        write_jsonl(out / "train.jsonl", train)
        write_jsonl(out / "val.jsonl", val)
        write_jsonl(out / "test.jsonl", test)
        stats["split_stats"] = split_stats
        print(f"Generated splits: {len(train)} train, {len(val)} val, {len(test)} test.")
    else:
        write_jsonl(out / "all.jsonl", samples)
        print(f"Saved canonical dataset to: {out / 'all.jsonl'}")

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest Lumae Product-Ads annotations.")
    parser.add_argument("--input", "-i", required=True, help="Input CSV or JSONL annotation file")
    parser.add_argument("--output-dir", "-o", default="local_data/canonical/lumae_ads", help="Output directory")
    parser.add_argument("--split", action="store_true", help="Split into train/val/test")
    parser.add_argument("--video-root", help="Optional directory to verify video existence")
    args = parser.parse_args()

    try:
        ingest_lumae_ads(
            input_path=args.input,
            output_dir=args.output_dir,
            split=args.split,
            video_root=args.video_root,
        )
    except Exception as exc:
        print(f"Ingestion failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
