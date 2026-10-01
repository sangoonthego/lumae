"""Bounded network and ffmpeg queues with single-thread manifest updates."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from .models import SEED, digest
from .source_pool import VIDEO_DIR, download, probe, save_candidate, selection
from .throughput import AttemptTimer, write_profile
from .visual_index import build_visual_index


def _fetch(row: dict) -> tuple:
    started = time.perf_counter()
    cached = (VIDEO_DIR / row["target_name"]).is_file()
    path = download(row)
    download_seconds = time.perf_counter() - started
    started = time.perf_counter()
    duration = probe(path)
    probe_seconds = time.perf_counter() - started
    if duration > 150:
        raise ValueError("duration over 150 seconds")
    return path, duration, digest(path), cached, download_seconds, probe_seconds


def prefetch(*, count: int = 20, download_workers: int = 4,
             ffmpeg_workers: int = 2, decode: bool = True, seed: int = SEED) -> dict:
    if count < 1 or not 1 <= download_workers <= 8 or not 1 <= ffmpeg_workers <= 4:
        raise ValueError("Unsafe prefetch count or worker count")
    state = selection(seed)
    rows = [r for r in state["candidates"] if r["status"] in ("unattempted", "downloaded")][:count]
    completed, failures = [], []
    with ThreadPoolExecutor(max_workers=download_workers) as network_pool, \
            ThreadPoolExecutor(max_workers=ffmpeg_workers) as decode_pool:
        network = {network_pool.submit(_fetch, row): row for row in rows}
        visual = {}
        for future in as_completed(network):
            row = network[future]
            video_id = row["source_video_id"]
            try:
                path, duration, sha, cache_hit, download_seconds, probe_seconds = future.result()
                timer = AttemptTimer(video_id)
                timer.mark_cache("download", cache_hit)
                timer.data["stages_seconds"]["download"] = round(download_seconds, 4)
                timer.data["stages_seconds"]["ffprobe"] = round(probe_seconds, 4)
                timer.data["total_seconds"] += download_seconds + probe_seconds
                save_candidate(state, video_id, status="downloaded", duration=duration, sha256=sha)
                if decode:
                    visual[decode_pool.submit(build_visual_index, path, duration, sha, timer)] = (
                        row, duration, timer)
                else:
                    timer.save(status="downloaded", duration=duration)
                    completed.append(video_id)
            except (OSError, RuntimeError, ValueError) as exc:
                failures.append({"source_video_id": video_id, "error": str(exc)})
        for future in as_completed(visual):
            row, duration, timer = visual[future]
            video_id = row["source_video_id"]
            try:
                index = future.result()
                save_candidate(state, video_id, status="visual_index_ready")
                timer.save(status="visual_index_ready", duration=duration,
                           frame_count=len(index["frames"]))
                completed.append(video_id)
            except (OSError, RuntimeError, ValueError) as exc:
                timer.save(status="downloaded", duration=duration)
                failures.append({"source_video_id": video_id, "error": str(exc)})
    write_profile()
    return {"requested": len(rows), "ready": len(completed),
            "failed": len(failures), "ready_video_ids": sorted(completed),
            "failures": failures, "download_workers": download_workers,
            "ffmpeg_workers": ffmpeg_workers if decode else 0}
