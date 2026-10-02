"""Read-only D200 replay; every generated artifact belongs to the D500 audit."""
from __future__ import annotations

import hashlib
import time

from .cache_validation import visual_index_valid, clip_npz_valid
from .cpu_session import overview
from .freeze import accepted_records, validate_annotation
from .incremental import ProtectedInputs, verify_parent
from .layout import BuildLayout
from .models import atomic_json, digest, read_json
from .pipeline import _verification_image
from .router import route_candidate
from .temporal_labeler import localize
from .visual_verifier import verify


def replay(layout=None, count=25):
    layout = layout or BuildLayout()
    if not 20 <= count <= 30:
        raise ValueError("Shadow replay requires 20–30 existing accepted examples")
    verify_parent(layout)
    guard = ProtectedInputs(layout)
    before = guard.verify(full=True)
    started = time.perf_counter()
    work = layout.work / "shadow"
    rows = [r for r in accepted_records(layout.root / "local_data/intermediate/lumae_ads_d200/accepted")
            if "automatic_verifier" in r and "router" in r]
    results = []
    historical_mismatches = []
    for row in rows:
        vid, query = row["source_video_id"], row["query"]
        validate_annotation(row)
        directory = layout.visual_cache / vid
        index = read_json(directory / "metadata.json")
        if not visual_index_valid(index, directory, row["source_video_sha256"]) or not clip_npz_valid(directory / "clip_features.npz", index):
            raise ValueError("Invalid frozen shadow cache: " + vid)
        review = read_json(layout.root / "local_data/intermediate/lumae_ads_d200/visual_index" / vid / "visual_events.json")
        events = review["events"]
        n = next(n for n, e in enumerate(events) if e["query"].strip() == query)
        pred = localize(vid, query, events, row["duration"], n)
        from .repo_paths import resolve_path
        original = read_json(resolve_path(row["semantic_artifact_path"]))
        image, times = _verification_image(vid, pred, events[n], work=work)
        auto = verify(query=query, prediction=pred, duration=row["duration"], verification_image=image, evidence_timestamps=times)
        routed = route_candidate(query=query, prediction=pred, duration=row["duration"],
                                 visual_evidence={"timestamps": events[n]["evidence_timestamps"], "contact_sheets": events[n]["evidence_sheets"]},
                                 automatic_verifier=auto)
        # Frame-reference paths change with output context; semantic verdict fields must not.
        clean = lambda value: {k: v for k, v in value.items() if k != "frame_references"}
        checks = {"query_exact": pred["query"] == query,
                  "prediction_exact": pred == original,
                  "window_exact": [pred["start"], pred["end"]] == row["window"],
                  "automatic_verdict_exact": clean(auto) == clean(row["automatic_verifier"]),
                  "router_exact": routed == row["router"],
                  "verification_image_exact": digest(image) == row["verification_image_sha256"],
                  "verifier_evidence_exact": times == row["verifier"]["evidence_timestamps"],
                  "review_provenance_exact": row["review_provenance"] == "AI_PSEUDO_LABELED"}
        atomic_json(work / "predictions" / (vid + ".json"), pred)
        overview(vid, work=work)
        case = {"source_video_id": vid, "checks": checks, "pass": all(checks.values())}
        if (all(checks[k] for k in ("query_exact", "prediction_exact", "window_exact", "router_exact", "review_provenance_exact"))
                and not checks["verifier_evidence_exact"]):
            case["reason"] = "D200 historical verifier was accepted with a narrower inside-frame selection than current saved event evidence"
            historical_mismatches.append(case)
        else:
            results.append(case)
        if len(results) == count:
            break
    after = guard.verify(full=True)
    report = {"status": "PASS" if len(results) == count and all(r["pass"] for r in results) else "FAIL",
              "samples": len(results), "duration_seconds": time.perf_counter() - started,
              "scope": "Frozen D200 query/evidence replay, identical Semantic V3/refinement/router/structural verifier and image bytes; existing agent verdict is checked, not re-annotated",
              "protected_before": before, "protected_after": after, "results": results,
              "historical_d200_mismatches": historical_mismatches,
              "selection_policy": "First 25 strict replay matches in source-ID order; disclose all skipped historical verifier-evidence mismatches"}
    atomic_json(layout.reports / "shadow_replay.json", report)
    return report
