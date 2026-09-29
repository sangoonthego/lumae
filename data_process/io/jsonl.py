"""Streaming and robust JSONL reading and writing utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Generator, Iterable

from ..schemas.temporal_sample import CanonicalTemporalSample, RejectedSampleRecord


class JsonlParseError(Exception):
    """Raised when JSONL parsing fails on a specific line."""
    def __init__(self, line_num: int, raw_line: str, cause: Exception):
        snippet = raw_line[:80] + "..." if len(raw_line) > 80 else raw_line
        super().__init__(f"Malformed JSON on line {line_num}: '{snippet}' ({cause})")
        self.line_num = line_num
        self.raw_line = raw_line
        self.cause = cause


def read_jsonl(path: Path | str) -> Generator[dict[str, Any], None, None]:
    """Read a JSONL file line-by-line in a streaming manner.

    Yields:
        Parsed JSON dictionary for each non-empty line.

    Raises:
        FileNotFoundError: if the file does not exist.
        JsonlParseError: if any non-empty line contains invalid JSON.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"JSONL file not found at: {file_path}")

    with file_path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except Exception as exc:
                raise JsonlParseError(line_num, stripped, exc) from exc
            if not isinstance(record, dict):
                raise JsonlParseError(
                    line_num, stripped, ValueError(f"Line must parse to a JSON object, got {type(record)}")
                )
            yield record


def write_jsonl(
    path: Path | str,
    items: Iterable[dict[str, Any] | CanonicalTemporalSample | RejectedSampleRecord],
) -> int:
    """Write items to a JSONL file in UTF-8 format.

    Args:
        path: Destination file path. Parent directories are created if needed.
        items: Iterable of dicts, CanonicalTemporalSamples, or RejectedSampleRecords.

    Returns:
        Number of items written.
    """
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with dest.open("w", encoding="utf-8") as f:
        for item in items:
            if hasattr(item, "to_dict"):
                payload = item.to_dict()
            elif isinstance(item, dict):
                payload = item
            else:
                raise TypeError(f"Unsupported item type for JSONL serialization: {type(item)}")
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
            count += 1

    return count
