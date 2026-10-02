"""Bind engineering cache reuse to complete frozen annotation inputs."""
from __future__ import annotations

from .cache_identity import deterministic_fingerprint, VisualIndexIdentity, ClipCacheIdentity
from .models import read_json, atomic_json, digest
from .temporal_labeler import WRAPPER_VERSION, localize
from data_process.annotation.assisted_deployment import ALGORITHM_SHA256


def frame_fingerprint(index):
    return deterministic_fingerprint({"visual": VisualIndexIdentity(index["video_sha256"]).fingerprint(),
                                      "frames": index["frames"], "duration": index["duration"]})


def clip_fingerprint(index):
    return ClipCacheIdentity(index["video_sha256"], frame_fingerprint(index)).fingerprint()


def verifier_identity(prediction, image_sha256, times):
    return deterministic_fingerprint({"prediction": prediction, "image_sha256": image_sha256,
                                      "evidence": times, "version": "d200_verifier_v1", "provider": "Codex agent visual inspection"})


def cached_localize(layout, video_id, query, events, duration, event_index, timer, runtime):
    index = read_json(layout.visual_cache / video_id / "metadata.json")
    query_identity = deterministic_fingerprint({"frames": frame_fingerprint(index), "events": events,
                                               "provider": "AgentEvidenceProvider", "version": "d200_pipeline_v1"})
    identity = deterministic_fingerprint({"query": query, "query_identity": query_identity,
                                         "clip_sha256": index["clip_features_sha256"], "duration": duration,
                                         "event_index": event_index, "semantic_v3": ALGORITHM_SHA256,
                                         "boundary": WRAPPER_VERSION})
    path = layout.work / "stage_cache" / video_id / (identity + ".json")
    prediction = None
    if path.is_file():
        try:
            record = read_json(path)
            if record["identity"] == identity and record["prediction_sha256"] == deterministic_fingerprint(record["prediction"]):
                prediction = record["prediction"]
        except (ValueError, OSError, KeyError, TypeError):
            pass
    query_path = layout.work / "stage_cache" / video_id / "agent_events.json"
    reviewed = query_path.is_file()
    if reviewed:
        try:
            old = read_json(query_path)
            reviewed = old.get("identity") == query_identity and old.get("events") == events
        except (OSError, ValueError):
            reviewed = False
    timer.mark_cache("query_gen", reviewed, query_identity)
    if not reviewed:
        atomic_json(query_path, {"identity": query_identity, "events": events})
    timer.mark_cache("semantic_refinement", prediction is not None, identity)
    if prediction is None:
        prediction = localize(video_id, query, events, duration, event_index, timer=timer, clip_runtime=runtime)
        atomic_json(path, {"identity": identity, "prediction": prediction,
                           "prediction_sha256": deterministic_fingerprint(prediction)})
    return prediction
