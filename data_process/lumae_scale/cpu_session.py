"""Persistent local annotation session with bounded preparation and real review timing.

The agent supplies visual observations; this module never invents query or review
decisions. A single owner writes the selection/ledger while preparation workers
write only distinct source caches. Existing accepted bytes are protected.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import math
from pathlib import Path
import statistics
import sys
import threading
import time

import requests
from PIL import Image, ImageDraw

from .models import ROOT, WORK, VISUAL_CACHE, atomic_json, digest, read_json
from .freeze import ACCEPTED
from .pipeline import run
from .query_generator import AgentEvidenceProvider
from .repo_paths import to_repo_relative
from .source_pool import download, probe, selection
from .visual_index import build_visual_index

SESSION_DIR = WORK / "cpu_sessions"


def overview(video_id: str, *, work: Path | None = None) -> Path:
    index = read_json(VISUAL_CACHE / video_id / "metadata.json")
    frames = index["frames"]
    tag = hashlib.sha256(json.dumps([(f["timestamp"], f["sha256"]) for f in frames]).encode()).hexdigest()[:12]
    path = (work or WORK) / "visual_index" / video_id / f"overview_{tag}.jpg"
    if path.exists():
        return path
    columns = min(5, len(frames))
    canvas = Image.new("RGB", (columns * 256, math.ceil(len(frames) / columns) * 166), "#171717")
    draw = ImageDraw.Draw(canvas)
    for n, frame in enumerate(frames):
        x, y = (n % columns) * 256, (n // columns) * 166
        with Image.open(VISUAL_CACHE / video_id / frame["path"]) as source:
            thumb = source.convert("RGB")
            thumb.thumbnail((250, 140))
            canvas.paste(thumb, (x + 3, y + 3))
        draw.text((x + 4, y + 146), f"{frame['timestamp']:.1f}s", fill="white")
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, quality=90)
    return path


class CPUSession:
    def __init__(self, prefetch: int = 16, download_workers: int = 6, decode_workers: int = 3):
        if prefetch != 16 or not 1 <= download_workers <= 8 or not 1 <= decode_workers <= 4:
            raise ValueError("CPU mode uses prefetch=16 and bounded worker counts")
        self.session_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.frozen = {p.name: digest(p) for p in ACCEPTED.glob("*.json")}
        self.pool = ThreadPoolExecutor(max_workers=download_workers)
        self.decode_slots = threading.Semaphore(decode_workers)
        self.jobs = {}
        self.active = None
        self.started = None
        self.views = 0
        self.compute = 0.0
        self.rows = []
        self.first_printed = False
        self.provider = AgentEvidenceProvider()
        self.work = WORK
        self.provider.config = {"provider": "Codex compact visual review", "model_id": "GPT-6",
                                "revision": "not exposed by runtime", "mode": "local_cpu_overview_v1"}
        self.fill_queue()

    def assert_immutable(self):
        for name, sha in self.frozen.items():
            if digest(ACCEPTED / name) != sha:
                raise RuntimeError(f"Existing accepted record changed: {name}")

    def _prepare(self, row):
        started = time.perf_counter()
        try:
            path = download(row)
            duration = probe(path)
            if duration > 150:
                return {"content_error": "duration over 150 seconds"}
            sha = digest(path)
            with self.decode_slots:
                index = build_visual_index(path, duration, sha)
            return {"duration": duration, "sha256": sha, "frames": len(index["frames"]),
                    "preparation_seconds": time.perf_counter() - started}
        except requests.RequestException as exc:
            return {"retryable": True, "error": str(exc)}
        except Exception as exc:
            # The active pipeline classifies errors; background work never rejects.
            return {"preparation_error": str(exc)}

    def fill_queue(self):
        rows = [r for r in selection()["candidates"]
                if r["status"] != "rejected" and not (ACCEPTED / (r["source_video_id"] + ".json")).exists()]
        for row in rows[:16]:
            vid = row["source_video_id"]
            if vid not in self.jobs:
                self.jobs[vid] = self.pool.submit(self._prepare, dict(row))

    def _accepted(self, video_id, record):
        elapsed = time.perf_counter() - self.started
        timing = {"session_id": self.session_id, "source_video_id": video_id,
                  "total_accepted_seconds": round(elapsed, 4),
                  "total_accepted_minutes": round(elapsed / 60, 4),
                  "visual_views_count": self.views, "timing_scope": "active source start through atomic accepted persistence; includes agent pauses",
                  "pipeline_call_seconds": round(self.compute + time.perf_counter() - self.call_started, 4),
                  "accepted_record_sha256": digest(ACCEPTED / (video_id + ".json"))}
        atomic_json(SESSION_DIR / self.session_id / (video_id + ".json"), timing)
        self.rows.append(timing)
        self.last_timing = timing
        if not self.first_printed:
            print(f"⏱ FIRST NEW ACCEPTED VIDEO: {video_id} | {elapsed:.1f}s | {elapsed / 60:.2f} min", flush=True)
            self.first_printed = True

    def _run(self):
        self.call_started = time.perf_counter()
        result = run(resume=True, max_candidates=1, provider=self.provider, on_accepted=self._accepted)
        self.compute += time.perf_counter() - self.call_started
        self.assert_immutable()
        if self.active and (ACCEPTED / (self.active + ".json")).exists():
            result["accepted_timing"] = self.last_timing
            if len(self.rows) == 1:
                t = self.last_timing
                result["first_accepted_line"] = f"⏱ FIRST NEW ACCEPTED VIDEO: {self.active} | {t['total_accepted_seconds']:.1f}s | {t['total_accepted_minutes']:.2f} min"
            self.active = None
        elif self.active:
            row = next(r for r in selection()["candidates"] if r["source_video_id"] == self.active)
            if row["status"] == "rejected":
                self.active = None
        self.fill_queue()
        return result

    def handle(self, action, payload):
        self.assert_immutable()
        if action == "next":
            if self.active is None:
                rows = [r for r in selection()["candidates"] if r["status"] != "rejected"
                        and not (ACCEPTED / (r["source_video_id"] + ".json")).exists()]
                if not rows:
                    return self._run()
                self.active = rows[0]["source_video_id"]
                self.started = time.perf_counter()
                self.views, self.compute = 0, 0.0
                self.fill_queue()
                job = self.jobs.get(self.active)
                if job:
                    job.result()
            result = self._run()
            if result.get("status") == "AWAITING_VISUAL_INSPECTION":
                result["overview_path"] = str(overview(result["video_id"]))
            return result
        if action == "stats":
            values = [r["total_accepted_seconds"] for r in self.rows]
            return {"accepted_this_session": len(values), "median": statistics.median(values) if values else None,
                    "mean": statistics.mean(values) if values else None,
                    "p90": sorted(values)[min(len(values)-1, int(.9*len(values)))] if values else None,
                    "median_views": statistics.median([r["visual_views_count"] for r in self.rows]) if values else None}
        if not self.active:
            raise ValueError("No active source; request next first")
        if payload.get("video_id") != self.active:
            raise ValueError("Request video does not match active deterministic source")
        if action == "events":
            index = read_json(VISUAL_CACHE / self.active / "metadata.json")
            path = overview(self.active, work=self.work)
            for event in payload["events"]:
                if not event.get("evidence_sheets"):
                    positions = [n for n, f in enumerate(index["frames"])
                                 if f["timestamp"] in event["evidence_timestamps"]]
                    event["evidence_sheets"] = sorted({index["contact_sheets"][n // 10] for n in positions})
            record = {"inspection_method": "agent_viewed_overview", "overview_path": to_repo_relative(path),
                      "overview_sha256": digest(path),
                      "covered_frame_timestamps": [f["timestamp"] for f in index["frames"]],
                      "events": payload["events"]}
            if "rejection_reason" in payload:
                record["rejection_reason"] = payload["rejection_reason"]
            atomic_json(self.work / "visual_index" / self.active / "visual_events.json", record)
            self.views += payload.get("views", 1)
            return self._run()
        if action == "verify":
            record = read_json(self.work / "visual_index" / self.active / "visual_events.json")
            query = payload.get("query", record["events"][0]["query"])
            auto = read_json(self.work / "visual_index" / self.active / "automatic_verification.json")
            verdict = {"pass": payload["pass"], "notes": payload["notes"],
                       "evidence_timestamps": auto["evidence_timestamps"]}
            if hasattr(self, "layout"):
                from .stage_cache import verifier_identity
                from .repo_paths import resolve_path
                pred = read_json(self.work / "predictions" / self.active / (hashlib.sha256(query.encode()).hexdigest()+".json"))
                verdict["cache_identity"] = verifier_identity(pred, digest(resolve_path(auto["frame_references"][0])), auto["evidence_timestamps"])
            path = self.work / "visual_index" / self.active / "verifier.json"
            verdicts = read_json(path) if path.exists() else {}
            verdicts[hashlib.sha256(query.encode()).hexdigest()] = verdict
            atomic_json(path, verdicts)
            self.views += payload.get("views", 1)
            return self._run()
        raise ValueError("Unknown action")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["serve", "client"])
    parser.add_argument("action", nargs="?", default="next")
    parser.add_argument("--request", type=Path)
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--dataset-version", choices=["lumae_ads_d200", "lumae_ads_d500"], default="lumae_ads_d200")
    args = parser.parse_args()
    if args.mode == "client":
        payload = read_json(args.request) if args.request else {}
        response = requests.post(f"http://127.0.0.1:{args.port}/{args.action}", json=payload, timeout=900)
        print(json.dumps(response.json(), indent=2, ensure_ascii=False))
        response.raise_for_status()
        return
    if args.dataset_version == "lumae_ads_d500":
        from .incremental_session import IncrementalSession
        session = IncrementalSession()
    else:
        session = CPUSession()
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            try:
                payload = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                result = session.handle(self.path.lstrip("/"), payload)
                status = 200
            except Exception as exc:
                result, status = {"error": str(exc), "type": type(exc).__name__}, 500
            encoded = json.dumps(result, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)
        def log_message(self, *args):
            pass
    print(json.dumps({"session_id": session.session_id, "accepted_preserved": len(session.frozen),
                      "prefetch": 16, "download_workers": 6, "decode_workers": 3}), flush=True)
    server = HTTPServer(("127.0.0.1", args.port), Handler)
    try:
        server.serve_forever()
    finally:
        server.server_close()
        if hasattr(session, "close"):
            session.close()
        else:
            session.pool.shutdown(wait=True, cancel_futures=True)


if __name__ == "__main__":
    main()
