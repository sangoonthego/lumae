"""Crash-resumable D200 candidate processing with explicit visual review gates."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw
import requests

from data_process.annotation.assisted_deployment import verify_frozen_ranker

from . import VERSION
from .clip_runtime import get_clip_runtime
from .foundation import verify_d48
from .freeze import ACCEPTED, accepted_records, freeze
from .models import ROOT, SEED, VISUAL_CACHE, WORK, atomic_json, digest, read_json
from .quality_gate import interval_errors, query_errors
from .reporting import write_report
from .repo_paths import resolve_path, to_repo_relative
from .state_machine import is_network_or_io_retryable
from .query_generator import (AgentEvidenceProvider, QueryGenerator,
                              require_agent_inspection, validate_events)
from .source_pool import download, probe, save_candidate, selection
from .visual_index import build_visual_index, extract_clip_features
from .temporal_labeler import localize
from .throughput import AttemptTimer, write_profile
from .router import route_candidate
from .visual_verifier import verify as automatic_visual_verify

LEDGER = WORK / "build_ledger.jsonl"


class PipelineBlocker(RuntimeError):
    """Environment or provider failure; do not mislabel a good video as rejected."""


def _finish(result: dict) -> dict:
    write_profile()
    write_report()
    return result


def _ledger(kind: str, video_id: str, **details: object) -> None:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    row = {"time_utc": datetime.now(timezone.utc).isoformat(), "event": kind,
           "source_video_id": video_id, **details}
    with LEDGER.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _verification_image(video_id: str, prediction: dict,
                        event: dict) -> tuple[Path, dict]:
    index = read_json(VISUAL_CACHE / video_id / "metadata.json")
    start, end = prediction["start"], prediction["end"]
    frames = index["frames"]
    evidence = sorted(float(t) for t in event["evidence_timestamps"]
                      if start <= float(t) <= end)
    if not evidence:
        evidence = [(start + end) / 2]
    picks = [("before", max(0, start - 2)),
             ("inside", evidence[0]),
             ("inside", (start + end) / 2),
             ("inside", evidence[-1]),
             ("after", min(index["duration"], end + 2))]
    chosen = []
    for name, desired in picks:
        frame = min(frames, key=lambda f: abs(f["timestamp"] - desired))
        if chosen and chosen[-1][0] == name and chosen[-1][1]["timestamp"] == frame["timestamp"]:
            continue
        chosen.append((name, frame))
    canvas = Image.new("RGB", (320 * len(chosen), 240), "#171717")
    draw = ImageDraw.Draw(canvas)
    for i, (name, frame) in enumerate(chosen):
        with Image.open(VISUAL_CACHE / video_id / frame["path"]) as source:
            image = source.convert("RGB")
            image.thumbnail((310, 200))
            canvas.paste(image, (i * 320 + 5, 5))
        draw.text((i * 320 + 5, 210), f"{name}: {frame['timestamp']:.1f}s", fill="white")
    tag = hashlib.sha256(json.dumps(
        ["verification_v4", prediction["query"], prediction["start"], prediction["end"],
         evidence]
    ).encode()).hexdigest()[:12]
    path = WORK / "visual_index" / video_id / f"verification_{tag}.jpg"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        canvas.save(path, quality=90)
    evidence = {"before": [], "inside": [], "after": []}
    for name, frame in chosen:
        evidence[name].append(frame["timestamp"])
    return path, evidence


def run(*, target: int = 200, seed: int = SEED, resume: bool = True,
        provider: QueryGenerator | None = None, max_candidates: int | None = None,
        on_accepted=None) -> dict:
    if target != 200 or seed != SEED:
        raise ValueError("D200 build requires --target-videos 200 --seed 20260930")
    if not resume and (WORK / "source_selection.json").exists():
        raise ValueError("Existing state requires --resume; never overwrite accepted data")
    verify_d48()
    verify_frozen_ranker(ROOT)
    state = selection(seed)
    provider = provider or AgentEvidenceProvider()
    clip_runtime = get_clip_runtime()
    attempted = 0
    for row in state["candidates"]:
        if len(accepted_records()) == 152:
            manifest = freeze(target, seed)
            return _finish({"status": "COMPLETE", "dataset_sha256": manifest["dataset_sha256"]})
        video_id = row["source_video_id"]
        if (ACCEPTED / f"{video_id}.json").is_file() or row["status"] == "rejected":
            continue
        if max_candidates is not None and attempted >= max_candidates:
            break
        attempted += 1
        timer = AttemptTimer(video_id)
        duration = None
        index = None
        try:
            from .source_pool import VIDEO_DIR
            timer.mark_cache("download", (VIDEO_DIR / row["target_name"]).is_file())
            with timer.stage("download"):
                path = download(row)
            with timer.stage("ffprobe"):
                duration = probe(path)
            if duration > 150:
                raise ValueError("duration over 150 seconds")
            video_sha = digest(path)
            save_candidate(state, video_id, status="downloaded", duration=duration, sha256=video_sha)
            index = build_visual_index(path, duration, video_sha, timer=timer)
            save_candidate(state, video_id, status="visual_index_ready")
            if not isinstance(provider, AgentEvidenceProvider):
                try:
                    require_agent_inspection(video_id)
                except ValueError:
                    save_candidate(state, video_id, status="awaiting_visual_inspection")
                    return _finish({"status": "AWAITING_VISUAL_INSPECTION", "video_id": video_id,
                                    "contact_sheets": index["contact_sheets"]})
            try:
                with timer.stage("query"):
                    events = provider.events(video_id)
            except (OSError, RuntimeError) as exc:
                raise PipelineBlocker(f"Visual provider unavailable: {exc}") from exc
            if not events:
                if (isinstance(provider, AgentEvidenceProvider) and
                        (WORK / "visual_index" / video_id / "visual_events.json").is_file()):
                    review = read_json(WORK / "visual_index" / video_id / "visual_events.json")
                    _ledger("visual_inspection", video_id, candidate_count=0)
                    raise ValueError("Agent rejected video: " + review.get(
                        "rejection_reason", "no concrete localizable visual event"))
                save_candidate(state, video_id, status="awaiting_visual_inspection")
                _ledger("awaiting_visual_inspection", video_id,
                        contact_sheets=index["contact_sheets"])
                return _finish({"status": "AWAITING_VISUAL_INSPECTION", "video_id": video_id,
                                "contact_sheets": index["contact_sheets"]})
            events = validate_events(video_id, events, duration)
            if not events:
                raise ValueError("No valid visually grounded event candidates")
            _ledger("visual_inspection", video_id, candidate_count=len(events))
            try:
                index = extract_clip_features(video_id, timer=timer, clip_runtime=clip_runtime)
            except (OSError, RuntimeError) as exc:
                raise PipelineBlocker(f"CLIP extractor unavailable: {exc}") from exc
            failures = []
            for event_index, event in enumerate(events):
                query = event["query"].strip()
                errors = query_errors(query)
                if errors:
                    failures.append({"query": query, "errors": errors})
                    _ledger("query_rejected", video_id, query=query, reasons=errors)
                    continue
                try:
                    prediction = localize(video_id, query, events, duration, event_index,
                                          timer=timer, clip_runtime=clip_runtime)
                    if interval_errors(prediction["start"], prediction["end"], duration):
                        raise ValueError("Invalid temporal window")
                    pred_path = WORK / "predictions" / video_id / f"{hashlib.sha256(query.encode()).hexdigest()}.json"
                    with timer.stage("write"):
                        atomic_json(pred_path, prediction)
                    with timer.stage("verification"):
                        verify_path, evidence_times = _verification_image(video_id, prediction,
                                                                          event)
                        automatic_verdict = automatic_visual_verify(
                            query=query, prediction=prediction, duration=duration,
                            verification_image=verify_path, evidence_timestamps=evidence_times)
                        automatic_verdict["frame_references"] = [
                            to_repo_relative(p) for p in automatic_verdict.get("frame_references", [])]
                        atomic_json(WORK / "visual_index" / video_id /
                                    "automatic_verification.json", automatic_verdict)
                    routed = route_candidate(query=query, prediction=prediction,
                                              duration=duration,
                                              visual_evidence={"timestamps": event["evidence_timestamps"],
                                                               "contact_sheets": event["evidence_sheets"]},
                                              automatic_verifier=automatic_verdict)
                    atomic_json(WORK / "visual_index" / video_id / "router.json", routed)
                    if routed["route"] == "LOW":
                        failures.append({"query": query, "errors": [routed["reason"]]})
                        continue
                    _ledger("semantic_v3_prediction", video_id, query=query,
                            prediction_path=to_repo_relative(pred_path),
                            verification_image=to_repo_relative(verify_path))
                    verifier_path = WORK / "visual_index" / video_id / "verifier.json"
                    verdicts = read_json(verifier_path) if verifier_path.exists() else {}
                    verdict = verdicts.get(hashlib.sha256(query.encode()).hexdigest())
                    if verdict is None:
                        save_candidate(state, video_id, status="awaiting_interval_verification")
                        return _finish({"status": "AWAITING_INTERVAL_VERIFICATION", "video_id": video_id,
                                        "query": query, "prediction": prediction,
                                        "verification_image": str(verify_path)})
                    if verdict.get("pass") is not True or not verdict.get("notes"):
                        failures.append({"query": query, "errors": ["visual verifier failed"]})
                        _ledger("verification_failed", video_id, query=query)
                        continue
                    if verdict.get("evidence_timestamps") != evidence_times:
                        raise ValueError("Verifier evidence timestamps do not match displayed image")
                    accepted = {
                        "source_dataset": "AdsQA", "source_video_id": video_id,
                        "source_video_path": to_repo_relative(path), "source_video_sha256": video_sha,
                        "source_url": row["source_url"], "source_metadata": {
                            "source_video_id": video_id, "target_name": row["target_name"],
                            "source_split": row.get("source_split"), "source_url": row["source_url"]},
                        "query": query, "duration": duration,
                        "window": [prediction["start"], prediction["end"]],
                        "query_source": "AI_GENERATED",
                        "query_review_status": "AI_AUTO_ACCEPTED_FOR_TRAINING",
                        "query_generator": type(provider).__name__,
                        "query_generator_version": VERSION,
                        "query_provider_config": getattr(provider, "config", {
                            "provider": "Codex agent visual inspection",
                            "model_id": "GPT-6",
                            "revision": "not exposed by runtime"}),
                        "visual_evidence": {"timestamps": event["evidence_timestamps"],
                                            "contact_sheets": event["evidence_sheets"],
                                            "note": event.get("evidence_note", "")},
                        "approximate_event_phase_seconds": [event["coarse_start"],
                                                            event["coarse_end"]],
                        "query_sanitation": {"passed": True, "errors": []},
                        "sampled_frame_strategy": "2s_timestamped_contact_sheets",
                        "visual_index_sha256": digest(VISUAL_CACHE / video_id / "metadata.json"),
                        "temporal_label_source": "SEMANTIC_V3_PSEUDO_LABEL",
                        "review_provenance": "AI_PSEUDO_LABELED", "reviewer_id": "AI_PIPELINE",
                        "semantic_v3_algorithm_sha256": prediction["semantic_v3_algorithm_sha256"],
                        "semantic_artifact_path": to_repo_relative(pred_path),
                        "semantic_artifact_sha256": digest(pred_path),
                        "semantic_v3_prediction_sha256": digest(pred_path),
                        "semantic_confidence": prediction["semantic_confidence"],
                        "ranking_score": prediction["ranking_score"],
                        "ranking_margin": prediction["ranking_margin"],
                        "clip_signals": prediction["clip_signals"],
                        "verifier": verdict,
                        "automatic_verifier": automatic_verdict,
                        "router": routed,
                        "verification_image_path": to_repo_relative(verify_path),
                        "verification_image_sha256": digest(verify_path),
                        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                        "pipeline_version": VERSION,
                    }
                    with timer.stage("write"):
                        atomic_json(ACCEPTED / f"{video_id}.json", accepted)
                        if on_accepted is not None:
                            on_accepted(video_id, accepted)
                        save_candidate(state, video_id, status="accepted")
                    _ledger("accepted", video_id, query=query,
                            window=accepted["window"], reason="visual and temporal gates passed")
                    break
                except (ValueError, RuntimeError) as exc:
                    failures.append({"query": query, "errors": [str(exc)]})
                    _ledger("query_failed", video_id, query=query, error=str(exc))
            else:
                save_candidate(state, video_id, status="rejected", rejection_reason="all query candidates failed")
                _ledger("rejected", video_id, failures=failures)
        except requests.RequestException as exc:
            _ledger("download_deferred", video_id, reason=str(exc))
            return _finish({"status": "AWAITING_DOWNLOAD", "video_id": video_id,
                            "reason": str(exc)})
        except (ValueError, OSError, RuntimeError) as exc:
            if isinstance(exc, (PipelineBlocker, PermissionError)):
                raise
            if is_network_or_io_retryable(exc):
                _ledger("download_deferred", video_id, reason=str(exc))
                return _finish({"status": "AWAITING_DOWNLOAD", "video_id": video_id,
                                "reason": str(exc)})
            save_candidate(state, video_id, status="rejected", rejection_reason=str(exc))
            _ledger("rejected", video_id, reason=str(exc))
        finally:
            timer.save(status=row["status"], duration=duration,
                       frame_count=len(index["frames"]) if index and "frames" in index else None,
                       route=(read_json(WORK / "visual_index" / video_id / "router.json")["route"]
                              if (WORK / "visual_index" / video_id / "router.json").exists() else None))
            write_profile()
    return _finish({"status": "INCOMPLETE", "accepted_new": len(accepted_records()),
                    "attempted_this_run": attempted})
