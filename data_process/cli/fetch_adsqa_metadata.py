"""CLI utility to download and validate official AdsQA metadata and provenance."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any
import urllib.request


def compute_sha256(path: Path) -> str:
    """Compute SHA256 hexadecimal digest of a local file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def download_stream(url: str, dest_path: Path, timeout: int = 60) -> int:
    """Download file via streaming HTTP GET request with User-Agent header."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Lumae-AdsQA-Ingester/1.0"},
    )
    temp_path = dest_path.with_suffix(".tmp")
    bytes_downloaded = 0
    with urllib.request.urlopen(req, timeout=timeout) as response, temp_path.open("wb") as out_f:
        while chunk := response.read(65536):
            out_f.write(chunk)
            bytes_downloaded += len(chunk)
    temp_path.replace(dest_path)
    return bytes_downloaded


def validate_video_urls_structure(data: Any) -> list[dict[str, str]]:
    """Validate structure of video_urls.json without silent repair."""
    if not isinstance(data, list):
        raise ValueError(f"video_urls.json root must be a list, got {type(data).__name__}")
    if len(data) == 0:
        raise ValueError("video_urls.json is empty")

    validated: list[dict[str, str]] = []
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Item at index {idx} in video_urls.json is not an object: {type(item).__name__}")
        url = item.get("url")
        target_name = item.get("target_name")
        if not url or not isinstance(url, str) or not url.strip():
            raise ValueError(f"Item at index {idx} has invalid or missing 'url': {url}")
        if not target_name or not isinstance(target_name, str) or not target_name.strip():
            raise ValueError(f"Item at index {idx} has invalid or missing 'target_name': {target_name}")
        validated.append({
            "url": url.strip(),
            "target_name": target_name.strip(),
        })
    return validated


def fetch_adsqa_metadata(
    output_dir: Path | str,
    config_path: Path | str = "data_process/config/adsqa_source.json",
    overwrite: bool = False,
    timeout: int = 60,
) -> dict[str, Any]:
    """Fetch official AdsQA metadata files, validate schemas, and write source manifest."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    cfg_file = Path(config_path)
    if not cfg_file.is_file():
        raise FileNotFoundError(f"AdsQA source config not found at: {cfg_file}")

    with cfg_file.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    urls_cfg = cfg.get("urls", {})
    video_urls_source = urls_cfg.get("video_urls")
    train_annotations_source = urls_cfg.get("train_annotations")
    test_annotations_source = urls_cfg.get("test_annotations")

    if not video_urls_source:
        raise ValueError("Configuration missing 'urls.video_urls'")

    # 1. Download / Load video_urls.json
    video_urls_path = out / "video_urls.json"
    if video_urls_path.is_file() and not overwrite:
        print(f"Using existing video_urls.json at {video_urls_path}")
    else:
        print(f"Downloading video_urls.json from {video_urls_source}...")
        download_stream(video_urls_source, video_urls_path, timeout=timeout)

    video_urls_sha = compute_sha256(video_urls_path)
    video_urls_size = video_urls_path.stat().st_size
    with video_urls_path.open("r", encoding="utf-8") as f:
        raw_video_urls = json.load(f)

    validated_urls = validate_video_urls_structure(raw_video_urls)
    print(f"Validated {len(validated_urls):,} entries in video_urls.json.")

    # 2. Download / Load train.json to extract official train split
    train_video_ids: set[str] = set()
    train_file_record: dict[str, Any] | None = None
    if train_annotations_source:
        train_path = out / "train.json"
        if train_path.is_file() and not overwrite:
            print(f"Using existing train.json at {train_path}")
        else:
            print(f"Downloading train.json from {train_annotations_source}...")
            download_stream(train_annotations_source, train_path, timeout=timeout)

        train_sha = compute_sha256(train_path)
        train_size = train_path.stat().st_size
        with train_path.open("r", encoding="utf-8") as f:
            train_data = json.load(f)

        if not isinstance(train_data, list):
            raise ValueError(f"train.json root must be a list, got {type(train_data).__name__}")

        for item in train_data:
            vid = item.get("video")
            if vid and isinstance(vid, str):
                train_video_ids.add(vid.strip())

        train_file_record = {
            "filename": "train.json",
            "source_url": train_annotations_source,
            "sha256": train_sha,
            "size_bytes": train_size,
            "items_count": len(train_data),
            "unique_videos": len(train_video_ids),
        }
        print(f"Extracted {len(train_video_ids)} unique video IDs from AdsQA train split.")

    # 3. Download / Load testset_question.json to extract official test split
    test_video_ids: set[str] = set()
    test_file_record: dict[str, Any] | None = None
    if test_annotations_source:
        test_path = out / "testset_question.json"
        if test_path.is_file() and not overwrite:
            print(f"Using existing testset_question.json at {test_path}")
        else:
            print(f"Downloading testset_question.json from {test_annotations_source}...")
            download_stream(test_annotations_source, test_path, timeout=timeout)

        test_sha = compute_sha256(test_path)
        test_size = test_path.stat().st_size
        with test_path.open("r", encoding="utf-8") as f:
            test_data = json.load(f)

        if not isinstance(test_data, list):
            raise ValueError(f"testset_question.json root must be a list, got {type(test_data).__name__}")

        for item in test_data:
            vid = item.get("video")
            if vid and isinstance(vid, str):
                test_video_ids.add(vid.strip())

        test_file_record = {
            "filename": "testset_question.json",
            "source_url": test_annotations_source,
            "sha256": test_sha,
            "size_bytes": test_size,
            "items_count": len(test_data),
            "unique_videos": len(test_video_ids),
        }
        print(f"Extracted {len(test_video_ids)} unique video IDs from AdsQA test split.")

    # 4. Map video entries preserving original IDs and source splits
    unique_videos_map: dict[str, dict[str, Any]] = {}
    train_count = 0
    test_count = 0
    unassigned_count = 0

    for item in validated_urls:
        target_name = item["target_name"]
        url = item["url"]
        vid_id = target_name.replace(".mp4", "")

        if vid_id in unique_videos_map:
            # Duplicate entry in video_urls list, keep existing record
            continue

        if vid_id in train_video_ids:
            split = "train"
            train_count += 1
        elif vid_id in test_video_ids:
            split = "test"
            test_count += 1
        else:
            split = "unassigned"
            unassigned_count += 1

        unique_videos_map[vid_id] = {
            "source_video_id": vid_id,
            "target_name": target_name,
            "source_url": url,
            "source_split": split,
        }

    now_utc = datetime.now(timezone.utc).isoformat()
    source_files = [
        {
            "filename": "video_urls.json",
            "source_url": video_urls_source,
            "sha256": video_urls_sha,
            "size_bytes": video_urls_size,
            "entries_count": len(validated_urls),
        }
    ]
    if train_file_record:
        source_files.append(train_file_record)
    if test_file_record:
        source_files.append(test_file_record)

    source_manifest: dict[str, Any] = {
        "dataset": "AdsQA",
        "upstream_repository": cfg.get("upstream_repository", "https://github.com/TsinghuaC3I/AdsQA"),
        "huggingface_dataset": cfg.get("huggingface_dataset", "https://huggingface.co/datasets/TsinghuaC3I/AdsQA"),
        "retrieval_timestamp_utc": now_utc,
        "source_files": source_files,
        "splits_summary": {
            "total_unique_videos": len(unique_videos_map),
            "train_videos_count": train_count,
            "test_videos_count": test_count,
            "unassigned_videos_count": unassigned_count,
        },
        "videos": list(unique_videos_map.values()),
    }

    manifest_path = out / "source_manifest.json"
    manifest_path.write_text(json.dumps(source_manifest, indent=2), encoding="utf-8")
    print(f"Generated AdsQA source manifest: {manifest_path} ({len(unique_videos_map):,} unique videos).")

    return source_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch and validate official AdsQA metadata and provenance.")
    parser.add_argument(
        "--output-dir",
        "-o",
        default="local_data/raw/adsqa",
        help="Destination directory for raw metadata (default: local_data/raw/adsqa)",
    )
    parser.add_argument(
        "--config",
        "-c",
        default="data_process/config/adsqa_source.json",
        help="Path to adsqa_source.json configuration",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite files if they already exist locally",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Network timeout in seconds (default: 60)",
    )
    args = parser.parse_args()

    try:
        fetch_adsqa_metadata(
            output_dir=args.output_dir,
            config_path=args.config,
            overwrite=args.overwrite,
            timeout=args.timeout,
        )
    except Exception as exc:
        print(f"Error fetching AdsQA metadata: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
