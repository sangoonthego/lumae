"""Measured per-attempt timings; absent historical timings remain unknown."""

from __future__ import annotations

import statistics
import time
from contextlib import contextmanager
from pathlib import Path

from .models import ROOT, WORK, atomic_json, read_json

TIMINGS = WORK / "timings"
PROFILE = ROOT / "local_data/reports/lumae_ads_d200/d200_throughput_profile.json"
STAGES = ("download", "ffprobe", "frame_extraction", "contact_sheet", "clip",
          "query", "semantic", "refinement", "verification", "agent_review", "write")


class AttemptTimer:
    def __init__(self, video_id: str):
        self.video_id = video_id
        self.path = TIMINGS / f"{video_id}.json"
        self.data = read_json(self.path) if self.path.exists() else {
            "source_video_id": video_id, "stages_seconds": {name: None for name in STAGES},
            "cache_hits": {}, "total_seconds": 0.0, "status": "in_progress"}
        self._started = time.perf_counter()

    @contextmanager
    def stage(self, name: str):
        if name not in STAGES:
            raise ValueError(f"Unknown timing stage: {name}")
        started = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - started
            previous = self.data["stages_seconds"].get(name)
            self.data["stages_seconds"][name] = round((previous or 0.0) + elapsed, 4)

    def mark_cache(self, name: str, hit: bool) -> None:
        self.data["cache_hits"][name] = bool(hit)

    def save(self, *, status: str, duration: float | None = None,
             frame_count: int | None = None, route: str | None = None) -> None:
        self.data["total_seconds"] = round(
            self.data.get("total_seconds", 0.0) + time.perf_counter() - self._started, 4)
        self._started = time.perf_counter()
        self.data["status"] = status
        if duration is not None:
            self.data["video_duration_seconds"] = duration
        if frame_count is not None:
            self.data["frame_count"] = frame_count
        if route is not None:
            self.data["route"] = route
        atomic_json(self.path, self.data)


def hardware() -> dict:
    import torch
    available = torch.cuda.is_available()
    result = {"cuda_available": available, "cpu_only": not available,
              "gpu_name": None, "gpu_vram_bytes": None}
    if available:
        props = torch.cuda.get_device_properties(0)
        result.update(gpu_name=props.name, gpu_vram_bytes=props.total_memory)
    return result


def write_profile() -> dict:
    rows = [read_json(path) for path in sorted(TIMINGS.glob("*.json"))]
    accepted = [r["total_seconds"] for r in rows if r["status"] == "accepted"]
    by_stage = {name: [r["stages_seconds"][name] for r in rows
                       if r["stages_seconds"].get(name) is not None] for name in STAGES}
    profile = {"hardware": hardware(), "attempts_with_timing": len(rows),
               "timing_scope": "Measured process time only; asynchronous queue wait and agent review time are not measured and must not be inferred from these totals.",
               "attempts_with_unmeasured_agent_review": sum(
                   r["status"] == "accepted" and
                   r["stages_seconds"].get("agent_review") is None for r in rows),
               "historical_accepted_without_timing": max(0, 10 - sum(
                   r["status"] == "accepted" for r in rows)),
               "accepted_video_seconds": {
                   "count": len(accepted),
                   "median": statistics.median(accepted) if accepted else None,
                   "mean": statistics.mean(accepted) if accepted else None,
                   "p90": sorted(accepted)[min(len(accepted) - 1,
                                              int(0.9 * len(accepted)))] if accepted else None},
               "stage_seconds": {name: {"count": len(values),
                                        "median": statistics.median(values) if values else None,
                                        "mean": statistics.mean(values) if values else None}
                                 for name, values in by_stage.items()},
               "attempts": rows}
    atomic_json(PROFILE, profile)
    return profile
