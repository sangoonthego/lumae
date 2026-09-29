"""CLI utility to download official QVHighlights annotations from trusted repository."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import urllib.request
from typing import Any

OFFICIAL_SOURCES: dict[str, dict[str, str]] = {
    "train": {
        "filename": "highlight_train_release.jsonl",
        "url": "https://raw.githubusercontent.com/jayleicn/moment_detr/main/data/highlight_train_release.jsonl",
        "description": "Official QVHighlights training annotations",
    },
    "val": {
        "filename": "highlight_val_release.jsonl",
        "url": "https://raw.githubusercontent.com/jayleicn/moment_detr/main/data/highlight_val_release.jsonl",
        "description": "Official QVHighlights validation annotations",
    },
    "test": {
        "filename": "highlight_test_release.jsonl",
        "url": "https://raw.githubusercontent.com/jayleicn/moment_detr/main/data/highlight_test_release.jsonl",
        "description": "Official QVHighlights test queries without ground truth",
    },
}

OFFICIAL_LICENSE_URL = "https://github.com/jayleicn/moment_detr/blob/main/data/LICENSE"
OFFICIAL_REPO_URL = "https://github.com/jayleicn/moment_detr"


def compute_sha256(path: Path) -> str:
    """Compute SHA256 hexadecimal digest of a local file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def download_file(url: str, dest_path: Path, timeout: int = 60) -> int:
    """Download file via streaming HTTP GET request."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "Lumae-Research-Ingester/1.0"})

    temp_path = dest_path.with_suffix(".tmp")
    bytes_downloaded = 0
    with urllib.request.urlopen(req, timeout=timeout) as response, temp_path.open("wb") as out_f:
        while chunk := response.read(65536):
            out_f.write(chunk)
            bytes_downloaded += len(chunk)

    temp_path.replace(dest_path)
    return bytes_downloaded


def fetch_qvhighlights_annotations(
    target_dir: Path | str,
    include_test: bool = False,
    overwrite: bool = False,
    timeout: int = 60,
) -> dict[str, Any]:
    """Fetch official QVHighlights annotation files and generate provenance source manifest."""
    root = Path(target_dir)
    root.mkdir(parents=True, exist_ok=True)

    splits_to_fetch = ["train", "val"]
    if include_test:
        splits_to_fetch.append("test")

    manifest_records: list[dict[str, Any]] = []

    print(f"Fetching official QVHighlights annotations into: {root.resolve()}")
    for split_key in splits_to_fetch:
        info = OFFICIAL_SOURCES[split_key]
        filename = info["filename"]
        url = info["url"]
        dest = root / filename

        existed = dest.is_file()
        if existed and not overwrite:
            print(f"[{split_key.upper()}] File exists at {filename}; verifying checksum (use --overwrite to redownload)...")
            size = dest.stat().st_size
            sha = compute_sha256(dest)
            timestamp = datetime.fromtimestamp(dest.stat().st_mtime, tz=timezone.utc).isoformat()
        else:
            action = "Overwriting" if existed else "Downloading"
            print(f"[{split_key.upper()}] {action} from {url}...")
            size = download_file(url, dest, timeout=timeout)
            sha = compute_sha256(dest)
            timestamp = datetime.now(timezone.utc).isoformat()
            print(f"[{split_key.upper()}] Downloaded {size:,} bytes | SHA256: {sha[:12]}...")

        manifest_records.append({
            "split": split_key,
            "filename": filename,
            "source_url": url,
            "sha256": sha,
            "size_bytes": size,
            "retrieved_at_utc": timestamp,
            "description": info["description"],
        })

    source_manifest: dict[str, Any] = {
        "dataset": "QVHighlights",
        "upstream_repository": OFFICIAL_REPO_URL,
        "license_source": OFFICIAL_LICENSE_URL,
        "generator": "data_process.cli.fetch_qvhighlights",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": manifest_records,
    }

    manifest_path = root / "source_manifest.json"
    manifest_path.write_text(json.dumps(source_manifest, indent=2), encoding="utf-8")
    print(f"Source manifest generated: {manifest_path}")

    return source_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Download official QVHighlights annotation files.")
    parser.add_argument(
        "--output-dir",
        "-o",
        default="local_data/raw/qvhighlights",
        help="Destination directory for raw annotations (default: local_data/raw/qvhighlights)",
    )
    parser.add_argument(
        "--include-test",
        action="store_true",
        help="Also download official test release queries without GT",
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
        fetch_qvhighlights_annotations(
            target_dir=args.output_dir,
            include_test=args.include_test,
            overwrite=args.overwrite,
            timeout=args.timeout,
        )
    except Exception as exc:
        print(f"Fetch failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
