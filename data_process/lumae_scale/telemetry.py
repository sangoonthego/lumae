"""Append-only per-attempt telemetry, independent of canonical annotations."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import time
import uuid

from .models import atomic_json, read_json

TIMING_NAMES = {
    "source_preflight": "source_preflight_ms", "download": "download_ms",
    "ffprobe": "probe_ms", "frame_extraction": "decode_or_visual_index_ms",
    "contact_sheet": "decode_or_visual_index_ms", "clip": "clip_ms",
    "overview": "overview_ms", "query": "query_generation_ms",
    "semantic": "semantic_v3_ms", "refinement": "boundary_refinement_ms",
    "verification": "verifier_ms", "write": "persistence_ms",
}


def utc(): return datetime.now(timezone.utc).isoformat()


class Telemetry:
    def __init__(self, directory: Path, source_id: str, kind="annotation", qid=None):
        self.directory = directory / source_id
        self.source_id, self.kind, self.qid = source_id, kind, qid
        self.attempt_id = uuid.uuid4().hex
        self.started = time.perf_counter()
        self.started_utc = utc()
        self.events = []
        self.errors = []

    def event(self, kind, **values):
        record = {"source_video_id": self.source_id, "attempt_id": self.attempt_id,
                  "attempt_kind": self.kind, "qid": self.qid,
                  "pipeline_version": "d500_incremental_v1",
                  "annotation_version": "d200_pipeline_v1", "timestamp_utc": utc(),
                  "event": kind, **values}
        self.events.append(record)
        # Telemetry failure cannot undo an atomically accepted annotation.
        try:
            atomic_json(self.directory / f"{self.attempt_id}_{len(self.events):04d}.json", record)
        except OSError as exc:
            self.errors.append(str(exc))
        return record

    @contextmanager
    def stage(self, name):
        started = time.perf_counter()
        outcome = "OK"
        try:
            yield
        except BaseException:
            outcome = "ERROR"
            raise
        finally:
            self.event("stage", stage=name, field=TIMING_NAMES.get(name, name + "_ms"),
                       elapsed_ms=round((time.perf_counter()-started)*1000, 4), outcome=outcome)

    def mark_cache(self, name, hit, fingerprint=None):
        self.event("cache", stage=name, hit=bool(hit), fingerprint=fingerprint)

    def usage(self, *, visual_calls=0, api_calls=None, input_tokens=None, output_tokens=None):
        self.event("usage", visual_calls=visual_calls, api_calls=api_calls,
                   input_tokens=input_tokens, output_tokens=output_tokens,
                   token_scope="Only provider-supplied usage; unavailable agent token counts remain null")

    def finish(self, outcome, **extra):
        timing = {name: None for name in set(TIMING_NAMES.values())}
        for e in self.events:
            if e["event"] == "stage":
                timing[e["field"]] = (timing.get(e["field"]) or 0) + e["elapsed_ms"]
        return self.event("attempt_complete", outcome=outcome, started_utc=self.started_utc,
                          total_ms=round((time.perf_counter()-self.started)*1000, 4),
                          timings_ms=timing, telemetry_write_errors=self.errors,
                          timing_scope="Current active attempt monotonic wall time; producer preparation is a separate overlapping attempt", **extra)


def aggregate(directory: Path) -> dict:
    rows = []
    malformed = []
    for p in sorted(directory.glob("*/*.json")):
        try: rows.append(read_json(p))
        except (OSError, json.JSONDecodeError) as exc: malformed.append({"path":p.name,"error":str(exc)})
    accepted = [r for r in rows if r["event"] == "attempt_complete" and r["outcome"] == "ACCEPT"]
    values = sorted(r["total_ms"] for r in accepted)
    def percentile(p):
        if not values: return None
        index = (len(values)-1)*p
        a, b = int(index), min(len(values)-1, int(index)+1)
        return values[a] + (values[b]-values[a])*(index-a)
    caches = [r for r in rows if r["event"] == "cache"]
    cache_rates = {s:{"hits":sum(r["hit"] for r in caches if r["stage"]==s),
                      "checks":sum(r["stage"]==s for r in caches)} for s in sorted({r["stage"] for r in caches})}
    for v in cache_rates.values(): v["hit_rate"] = v["hits"]/v["checks"]
    usage = [r for r in rows if r["event"]=="usage"]
    return {"accepted_attempts":len(accepted), "p50_total_ms":percentile(.5),
            "mean_total_ms":statistics.mean(values) if values else None,
            "p90_total_ms":percentile(.9), "p95_total_ms":percentile(.95),
            "active_accepted_samples_per_hour":3600000*len(values)/sum(values) if values else None,
            "throughput_scope":"Accepted active wall times; excludes rejected attempts and overlapping producer time",
            "cache":cache_rates, "visual_calls":sum(r["visual_calls"] for r in usage),
            "input_tokens":sum(r["input_tokens"] for r in usage) if usage and all(r["input_tokens"] is not None for r in usage) else None,
            "output_tokens":sum(r["output_tokens"] for r in usage) if usage and all(r["output_tokens"] is not None for r in usage) else None,
            "retry_events":sum(r["event"]=="retry" for r in rows),
            "timings_ms_by_stage":{s:{"measurements":len(v),"mean":statistics.mean(v)}
                  for s in sorted({r["field"] for r in rows if r["event"]=="stage"})
                  if (v:=[r["elapsed_ms"] for r in rows if r["event"]=="stage" and r["field"]==s])},
            "malformed_telemetry":malformed}
