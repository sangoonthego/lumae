"""Video probe and compatibility validation for Lumae / Moment-DETR pilot."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
from typing import Any

MAX_PILOT_DURATION_SECONDS = 150.0


def probe_video_file(video_path: Path | str) -> dict[str, Any]:
    """Inspect video file properties using ffprobe.

    Captures duration, resolution, codec, fps, file size, and model compatibility.
    """
    path = Path(video_path)
    if not path.is_file():
        return {
            "file_path": str(path),
            "file_size": 0,
            "duration": 0.0,
            "width": 0,
            "height": 0,
            "codec": None,
            "fps": None,
            "is_readable": False,
            "has_visual_stream": False,
            "is_model_compatible": False,
            "incompatibility_reason": f"File does not exist: {path}",
        }

    file_size = path.stat().st_size
    if file_size == 0:
        return {
            "file_path": str(path),
            "file_size": 0,
            "duration": 0.0,
            "width": 0,
            "height": 0,
            "codec": None,
            "fps": None,
            "is_readable": False,
            "has_visual_stream": False,
            "is_model_compatible": False,
            "incompatibility_reason": "File is empty (0 bytes)",
        }

    ffprobe_cmd = shutil.which("ffprobe")
    if not ffprobe_cmd:
        # Fallback if ffprobe is unexpectedly missing
        return {
            "file_path": str(path),
            "file_size": file_size,
            "duration": 0.0,
            "width": 0,
            "height": 0,
            "codec": "unknown",
            "fps": None,
            "is_readable": True,
            "has_visual_stream": True,
            "is_model_compatible": False,
            "incompatibility_reason": "ffprobe executable not found in PATH",
        }

    cmd = [
        ffprobe_cmd,
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]

    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
            check=False,
        )
        if proc.returncode != 0:
            return {
                "file_path": str(path),
                "file_size": file_size,
                "duration": 0.0,
                "width": 0,
                "height": 0,
                "codec": None,
                "fps": None,
                "is_readable": False,
                "has_visual_stream": False,
                "is_model_compatible": False,
                "incompatibility_reason": f"ffprobe failed: {proc.stderr[:100]}",
            }

        data = json.loads(proc.stdout)
    except Exception as exc:
        return {
            "file_path": str(path),
            "file_size": file_size,
            "duration": 0.0,
            "width": 0,
            "height": 0,
            "codec": None,
            "fps": None,
            "is_readable": False,
            "has_visual_stream": False,
            "is_model_compatible": False,
            "incompatibility_reason": f"Probe error: {exc}",
        }

    # Extract streams & format
    format_info = data.get("format", {})
    streams = data.get("streams", [])

    duration = 0.0
    if "duration" in format_info:
        try:
            duration = float(format_info["duration"])
        except (ValueError, TypeError):
            duration = 0.0

    video_stream: dict[str, Any] | None = None
    for st in streams:
        if st.get("codec_type") == "video":
            video_stream = st
            break

    has_visual = video_stream is not None
    width = 0
    height = 0
    codec = None
    fps = None

    if video_stream:
        width = int(video_stream.get("width") or 0)
        height = int(video_stream.get("height") or 0)
        codec = video_stream.get("codec_name")
        if duration <= 0 and "duration" in video_stream:
            try:
                duration = float(video_stream["duration"])
            except (ValueError, TypeError):
                pass

        # Parse fps
        r_fps = video_stream.get("r_frame_rate") or video_stream.get("avg_frame_rate")
        if r_fps and "/" in str(r_fps):
            num, den = str(r_fps).split("/", 1)
            try:
                den_f = float(den)
                if den_f > 0:
                    fps = round(float(num) / den_f, 2)
            except (ValueError, ZeroDivisionError):
                pass

    # Model compatibility checks
    is_compatible = True
    reason = None

    if not has_visual:
        is_compatible = False
        reason = "No visual video stream found"
    elif duration <= 0:
        is_compatible = False
        reason = "Invalid video duration (<= 0s)"
    elif duration > MAX_PILOT_DURATION_SECONDS:
        is_compatible = False
        reason = f"Duration exceeds {MAX_PILOT_DURATION_SECONDS:.0f}s limit (actual: {duration:.1f}s)"

    return {
        "file_path": str(path),
        "file_size": file_size,
        "duration": round(duration, 3),
        "width": width,
        "height": height,
        "codec": codec,
        "fps": fps,
        "is_readable": True,
        "has_visual_stream": has_visual,
        "is_model_compatible": is_compatible,
        "incompatibility_reason": reason,
    }
