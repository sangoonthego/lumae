"""Semantic V2 Pre-annotation Module for Stage A.2.5D.

Upgrades from heuristic motion/scene proposals to query-conditioned visual inspection.
Generates coarse and dense timestamped contact sheets for multimodal inspection
and serializes standardized semantic_v2 candidate records.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
from typing import Any, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont


@dataclass
class SemanticPreannotationV2Record:
    """Standardized record for Stage A.2.5D semantic pre-annotation."""

    sample_id: str
    video_filename: str
    query: str
    semantic_status: str  # VALID, UNCERTAIN, NO_MATCH
    candidate_start_seconds: Optional[float]
    candidate_end_seconds: Optional[float]
    candidate_windows_json: str
    semantic_confidence: str  # HIGH, MEDIUM, LOW
    semantic_reason: str
    coarse_interval_seconds: float
    dense_interval_seconds: float
    analysis_method: str = "query_conditioned_multimodal_visual_inspection"
    generated_at_utc: str = ""
    preannotator_version: str = "semantic_v2"

    def get_windows(self) -> List[List[float]]:
        """Parse candidate windows JSON safely."""
        try:
            parsed = json.loads(self.candidate_windows_json)
            if isinstance(parsed, list):
                return [[round(float(w[0]), 1), round(float(w[1]), 1)] for w in parsed]
        except Exception:
            pass
        if self.candidate_start_seconds is not None and self.candidate_end_seconds is not None:
            return [[round(self.candidate_start_seconds, 1), round(self.candidate_end_seconds, 1)]]
        return []

    def to_csv_dict(self) -> dict[str, str]:
        start_str = f"{self.candidate_start_seconds:.1f}" if self.candidate_start_seconds is not None else ""
        end_str = f"{self.candidate_end_seconds:.1f}" if self.candidate_end_seconds is not None else ""
        return {
            "sample_id": self.sample_id,
            "video_filename": self.video_filename,
            "query": self.query,
            "semantic_status": self.semantic_status,
            "candidate_start_seconds": start_str,
            "candidate_end_seconds": end_str,
            "candidate_windows_json": self.candidate_windows_json,
            "semantic_confidence": self.semantic_confidence,
            "semantic_reason": self.semantic_reason,
            "coarse_interval_seconds": f"{self.coarse_interval_seconds:.1f}",
            "dense_interval_seconds": f"{self.dense_interval_seconds:.2f}",
            "analysis_method": self.analysis_method,
            "generated_at_utc": self.generated_at_utc or datetime.now(timezone.utc).isoformat(),
            "preannotator_version": self.preannotator_version,
        }

    @classmethod
    def from_csv_dict(cls, row: dict[str, str]) -> SemanticPreannotationV2Record:
        start_val = float(row["candidate_start_seconds"]) if row.get("candidate_start_seconds") else None
        end_val = float(row["candidate_end_seconds"]) if row.get("candidate_end_seconds") else None
        return cls(
            sample_id=row["sample_id"],
            video_filename=row["video_filename"],
            query=row["query"],
            semantic_status=row.get("semantic_status", "VALID"),
            candidate_start_seconds=start_val,
            candidate_end_seconds=end_val,
            candidate_windows_json=row.get("candidate_windows_json", "[]"),
            semantic_confidence=row.get("semantic_confidence", "MEDIUM"),
            semantic_reason=row.get("semantic_reason", ""),
            coarse_interval_seconds=float(row.get("coarse_interval_seconds", 2.0)),
            dense_interval_seconds=float(row.get("dense_interval_seconds", 0.5)),
            analysis_method=row.get("analysis_method", "query_conditioned_multimodal_visual_inspection"),
            generated_at_utc=row.get("generated_at_utc", ""),
            preannotator_version=row.get("preannotator_version", "semantic_v2"),
        )


CSV_FIELDS = [
    "sample_id",
    "video_filename",
    "query",
    "semantic_status",
    "candidate_start_seconds",
    "candidate_end_seconds",
    "candidate_windows_json",
    "semantic_confidence",
    "semantic_reason",
    "coarse_interval_seconds",
    "dense_interval_seconds",
    "analysis_method",
    "generated_at_utc",
    "preannotator_version",
]


def save_semantic_preannotations_v2(
    records: List[SemanticPreannotationV2Record],
    csv_path: Path,
) -> None:
    """Save records to semantic_preannotations_v2.csv."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for r in records:
            writer.writerow(r.to_csv_dict())


def load_semantic_preannotations_v2(csv_path: Path) -> List[SemanticPreannotationV2Record]:
    """Load records from semantic_preannotations_v2.csv."""
    if not csv_path.exists():
        return []
    records = []
    with open(csv_path, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(SemanticPreannotationV2Record.from_csv_dict(row))
    return records


def get_video_duration(video_path: Path) -> float:
    """Get video duration using ffprobe."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(video_path),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    data = json.loads(res.stdout)
    return float(data["format"]["duration"])


def extract_frame_at_timestamp(
    video_path: Path,
    timestamp: float,
    output_path: Path,
    width: int = 480,
) -> bool:
    """Extract a single frame at timestamp using ffmpeg."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{timestamp:.3f}",
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        "-vf",
        f"scale={width}:-1",
        "-q:v",
        "2",
        str(output_path),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    return res.returncode == 0 and output_path.exists()


def add_timestamp_banner(
    img: Image.Image,
    timestamp: float,
    label: str = "",
) -> Image.Image:
    """Draw a high-contrast banner with timestamp on the image."""
    img_copy = img.copy()
    draw = ImageDraw.Draw(img_copy)
    banner_height = 28
    draw.rectangle([(0, 0), (img_copy.width, banner_height)], fill=(0, 0, 0, 220))
    text = f"T = {timestamp:6.2f}s  {label}".strip()
    draw.text((10, 6), text, fill=(255, 255, 0))
    return img_copy


def build_contact_sheet(
    frames_with_times: List[Tuple[Image.Image, float, str]],
    columns: int = 5,
    title: str = "",
) -> Image.Image:
    """Combine frames into a grid contact sheet."""
    if not frames_with_times:
        raise ValueError("No frames provided for contact sheet")

    frame_w, frame_h = frames_with_times[0][0].size
    num_frames = len(frames_with_times)
    rows = math.ceil(num_frames / columns)

    title_height = 40 if title else 0
    margin = 4
    sheet_w = columns * frame_w + (columns + 1) * margin
    sheet_h = rows * frame_h + (rows + 1) * margin + title_height

    sheet = Image.new("RGB", (sheet_w, sheet_h), color=(20, 20, 20))
    draw = ImageDraw.Draw(sheet)

    if title:
        draw.text((margin + 10, 10), title, fill=(255, 255, 255))

    for idx, (img, t, label) in enumerate(frames_with_times):
        r = idx // columns
        c = idx % columns
        x = margin + c * (frame_w + margin)
        y = title_height + margin + r * (frame_h + margin)

        annotated = add_timestamp_banner(img, t, label)
        sheet.paste(annotated, (x, y))

    return sheet


def generate_coarse_contact_sheet(
    video_path: Path,
    query: str,
    output_sheet_path: Path,
    interval: Optional[float] = None,
    columns: int = 5,
    frame_width: int = 360,
) -> float:
    """Generate coarse contact sheet per Step 4 requirements."""
    duration = get_video_duration(video_path)
    if interval is None:
        interval = 2.0 if duration <= 60.0 else 3.0

    timestamps = []
    t = 0.0
    while t < duration:
        timestamps.append(round(t, 2))
        t += interval

    frames_with_times = []
    tmp_dir = output_sheet_path.parent / f"tmp_frames_{output_sheet_path.stem}"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    try:
        for t_val in timestamps:
            frame_file = tmp_dir / f"frame_{t_val:06.2f}.jpg"
            if extract_frame_at_timestamp(video_path, t_val, frame_file, width=frame_width):
                img = Image.open(frame_file)
                frames_with_times.append((img.copy(), t_val, ""))
                img.close()

        title = f"COARSE INSPECTION: {video_path.name} | Interval: {interval:.1f}s | Query: {query}"
        sheet = build_contact_sheet(frames_with_times, columns=columns, title=title)
        output_sheet_path.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(output_sheet_path, quality=90)
    finally:
        import shutil
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)

    return interval


def generate_dense_contact_sheet(
    video_path: Path,
    query: str,
    start_t: float,
    end_t: float,
    output_sheet_path: Path,
    interval: float = 0.5,
    columns: int = 5,
    frame_width: int = 360,
) -> None:
    """Generate dense contact sheet for candidate window boundary refinement."""
    duration = get_video_duration(video_path)
    s = max(0.0, start_t)
    e = min(duration, end_t)

    timestamps = []
    t = s
    while t <= e:
        timestamps.append(round(t, 2))
        t += interval

    frames_with_times = []
    tmp_dir = output_sheet_path.parent / f"tmp_frames_{output_sheet_path.stem}"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    try:
        for t_val in timestamps:
            frame_file = tmp_dir / f"frame_{t_val:06.2f}.jpg"
            if extract_frame_at_timestamp(video_path, t_val, frame_file, width=frame_width):
                img = Image.open(frame_file)
                frames_with_times.append((img.copy(), t_val, ""))
                img.close()

        title = f"DENSE REFINEMENT: {video_path.name} [{s:.1f}s - {e:.1f}s] | Interval: {interval:.2f}s | Query: {query}"
        sheet = build_contact_sheet(frames_with_times, columns=columns, title=title)
        output_sheet_path.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(output_sheet_path, quality=92)
    finally:
        import shutil
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
