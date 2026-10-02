"""D500 context for the existing CPU annotation protocol and frozen review gates."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import threading
import time

from .cpu_session import CPUSession, overview
from .freeze import accepted_records
from .incremental import ProtectedInputs, verify_parent, progress, append_ledger
from .layout import BuildLayout
from .models import atomic_json, digest, read_json, load_jsonl
from .pipeline import run
from .preparation import prepare_source, classify_failure, FailureKind
from .query_generator import AgentEvidenceProvider
from .source_pool import selection, save_candidate
from .telemetry import Telemetry, aggregate
from .visual_index import build_visual_index


class IncrementalSession(CPUSession):
    def __init__(self, layout=None, prefetch=16, download_workers=6, decode_workers=3):
        if prefetch != 16 or not 1 <= download_workers <= 8 or not 1 <= decode_workers <= 4:
            raise ValueError("CPU mode requires bounded prefetch=16 workers")
        self.layout = layout or BuildLayout()
        self.work = self.layout.work
        parent = verify_parent(self.layout)
        shadow = read_json(self.layout.reports / "shadow_replay.json")
        if shadow.get("status") != "PASS" or shadow.get("samples", 0) < 20:
            raise RuntimeError("D500 annotation requires a passing 20–30 sample shadow replay")
        self.guard = ProtectedInputs(self.layout)
        self.guard.verify(full=True)
        self.base_ids = {r["vid"].removeprefix("lumae_ads_") for r in load_jsonl(self.layout.base / "lumae_ads_d200_all.jsonl")}
        self.base_hashes = {r["sha256"] for r in parent["source_video_sha256"].values()}
        self.frozen = {p.name: digest(p) for p in self.layout.accepted.glob("*.json")}
        self.session_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.pool = ThreadPoolExecutor(max_workers=download_workers)
        self.decode_slots = threading.Semaphore(decode_workers)
        self.jobs, self.rows = {}, []
        self.active = self.timer = None
        self.started = None
        self.views = 0
        self.compute = 0.0
        self.first_printed = bool(self.frozen)
        self.closed = False
        self.provider = AgentEvidenceProvider(work=self.work)
        self.provider.config = {"provider": "Codex compact visual review", "model_id": "GPT-6",
                                "revision": "not exposed by runtime", "mode": "local_cpu_overview_v1"}
        self.reconcile()
        self.fill_queue()

    def assert_immutable(self):
        self.guard.verify()
        for name, sha in self.frozen.items():
            if digest(self.layout.accepted / name) != sha:
                raise RuntimeError("Accepted D500 annotation changed: " + name)

    def reconcile(self):
        state = selection(layout=self.layout)
        for row in state["candidates"]:
            path = self.layout.accepted / (row["source_video_id"] + ".json")
            if path.is_file() and row["status"] != "accepted":
                from .freeze import validate_annotation
                validate_annotation(read_json(path), visual_cache=self.layout.visual_cache)
                # Recovery after atomic acceptance but before selection persistence.
                row["status"] = "accepted"
                append_ledger(self.layout, "recovered_atomic_acceptance", row["source_video_id"])
        atomic_json(self.layout.selection, state)

    def eligible(self):
        now = time.time()
        return [r for r in selection(layout=self.layout)["candidates"]
                if r["status"] not in ("accepted", "rejected")
                and not (self.layout.accepted / (r["source_video_id"] + ".json")).is_file()
                and r.get("next_retry_utc_epoch", 0) <= now
                and max(r.get("network_retry_count", 0), r.get("io_retry_count", 0)) < 5]

    def _prepare(self, row):
        timer = Telemetry(self.work / "telemetry", row["source_video_id"], kind="preparation")
        try:
            path, metadata = prepare_source(row, self.layout, excluded_ids=self.base_ids,
                                            excluded_hashes=self.base_hashes, timer=timer)
            with self.decode_slots:
                index = build_visual_index(path, metadata["duration"], metadata["sha256"],
                                           timer=timer, work=self.work, strict=True)
            return {"frames": len(index["frames"]), "preparation": timer.finish("READY")}
        except Exception as exc:
            kind = classify_failure(exc)
            return {"failure_kind": kind.value, "error": str(exc), "preparation": timer.finish(kind.value)}

    def fill_queue(self):
        if self.closed:
            return
        if len(accepted_records(self.layout.accepted)) >= 300:
            return
        wanted = self.eligible()[:16]
        wanted_ids = {r["source_video_id"] for r in wanted} | ({self.active} if self.active else set())
        for vid, job in list(self.jobs.items()):
            if vid not in wanted_ids and (job.done() or job.cancel()):
                del self.jobs[vid]
        for row in wanted:
            vid = row["source_video_id"]
            if vid not in self.jobs and len(self.jobs) < 16:
                self.jobs[vid] = self.pool.submit(self._prepare, dict(row))
        if len(self.jobs) > 16:
            raise RuntimeError("Prefetch queue exceeded bound")

    def _accepted(self, video_id, record):
        # Attempt completion is written after the persistence stage has closed.
        self.frozen[video_id + ".json"] = digest(self.layout.accepted / (video_id + ".json"))

    def _run(self):
        self.call_started = time.perf_counter()
        result = run(target=500, resume=True, max_candidates=1, provider=self.provider,
                     layout=self.layout, attempt_timer=self.timer, candidate_id=self.active,
                     on_accepted=self._accepted)
        self.compute += time.perf_counter() - self.call_started
        if self.active:
            state = selection(layout=self.layout)
            row = next(r for r in state["candidates"] if r["source_video_id"] == self.active)
            if row["status"] in ("accepted", "rejected", "download_deferred"):
                outcome = "ACCEPT" if row["status"] == "accepted" else row.get("failure_kind") or "REJECT"
                t = self.timer.finish(outcome, visual_views_count=self.views, pipeline_call_seconds=self.compute)
                if outcome == "ACCEPT":
                    atomic_json(self.work / "attempts" / (self.active + ".json"), t)
                    self.rows.append(t)
                    result["accepted_timing"] = t
                    if not self.first_printed:
                        result["first_accepted_line"] = f"FIRST NEW ACCEPTED VIDEO: {self.active} | {t['total_ms']/1000:.1f}s"
                        self.first_printed = True
                if row["status"] == "download_deferred":
                    retries = max(row.get("network_retry_count", 0), row.get("io_retry_count", 0))
                    save_candidate(state, self.active, layout=self.layout,
                                   next_retry_utc_epoch=time.time() + min(300, 2 ** retries * 5))
                self.jobs.pop(self.active, None)
                self.active = self.timer = None
        self.assert_immutable()
        self.fill_queue()
        return result

    def handle(self, action, payload):
        self.assert_immutable()
        if action == "stats":
            return {**progress(self.layout), "telemetry": aggregate(self.work / "telemetry"), "queued_jobs": len(self.jobs)}
        if action == "next":
            if len(accepted_records(self.layout.accepted)) == 300:
                return self._run()
            if self.active is None:
                rows = self.eligible()
                if not rows:
                    return {"status": "AWAITING_RETRY_OR_SOURCE", "report": progress(self.layout)}
                self.active = rows[0]["source_video_id"]
                self.timer = Telemetry(self.work / "telemetry", self.active,
                                       qid=f"{self.layout.version}_{self.active}")
                self.views, self.compute = 0, 0.0
                self.fill_queue()
                if self.active in self.jobs:
                    self.jobs[self.active].result()
            result = self._run()
            if result.get("status") == "AWAITING_VISUAL_INSPECTION":
                with self.timer.stage("overview"):
                    result["overview_path"] = str(overview(self.active, work=self.work))
            return result
        if action in ("events", "verify"):
            if self.active is None or payload.get("video_id") != self.active:
                raise ValueError("Review does not match active source")
            self.timer.usage(visual_calls=payload.get("views", 1),
                             input_tokens=payload.get("input_tokens"), output_tokens=payload.get("output_tokens"))
        return super().handle(action, payload)

    def close(self):
        self.closed = True
        self.pool.shutdown(wait=True, cancel_futures=True)
        if self.timer:
            self.timer.finish("INTERRUPTED", visual_views_count=self.views)
        self.assert_immutable()
        atomic_json(self.layout.reports / "telemetry_summary.json", aggregate(self.work / "telemetry"))
