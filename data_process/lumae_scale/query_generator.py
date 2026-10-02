"""Visual event provider contract; no text metadata is passed to providers."""

from __future__ import annotations

import json
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path

from .models import VISUAL_CACHE, WORK, read_json, digest
from .repo_paths import resolve_path


class QueryGenerator(ABC):
    @abstractmethod
    def events(self, video_id: str) -> list[dict]:
        """Return inspected visual events with query, evidence and coarse phase."""


class AgentEvidenceProvider(QueryGenerator):
    """Reads only explicit agent-authored descriptions of viewed contact sheets."""

    def __init__(self, *, work=None, visual_cache=None):
        self.work, self.visual_cache = work, visual_cache

    def events(self, video_id: str) -> list[dict]:
        path = (self.work or WORK) / "visual_index" / video_id / "visual_events.json"
        if not path.is_file():
            return []
        data = read_json(path)
        index = read_json((self.visual_cache or VISUAL_CACHE) / video_id / "metadata.json")
        if data.get("inspection_method") == "agent_viewed_overview":
            overview = resolve_path(data.get("overview_path", ""))
            if (not overview.is_file() or digest(overview) != data.get("overview_sha256")
                    or data.get("covered_frame_timestamps") != [f["timestamp"] for f in index["frames"]]):
                raise ValueError("Overview inspection lacks complete cached-frame coverage")
            return data["events"]
        if data.get("inspection_method") != "agent_viewed_contact_sheets":
            raise ValueError("Visual event file lacks actual inspection attestation")
        if set(data.get("inspected_contact_sheets", [])) != set(index["contact_sheets"]):
            raise ValueError("Agent did not attest to viewing every contact sheet")
        return data["events"]


class LocalCommandProvider(QueryGenerator):
    """Configurable local multimodal command protocol; never calls a paid API."""

    def __init__(self, command: list[str], *, model_id: str, revision: str,
                 device: str = "cpu", precision: str = "float32", batch_size: int = 1):
        if not command or not model_id or not revision or batch_size < 1:
            raise ValueError("Local provider requires command, model, revision and batch size")
        self.command = command
        self.config = {"model_id": model_id, "revision": revision, "device": device,
                       "precision": precision, "batch_size": batch_size}

    def events(self, video_id: str) -> list[dict]:
        index = read_json(VISUAL_CACHE / video_id / "metadata.json")
        paths = [str(VISUAL_CACHE / video_id / name) for name in index["contact_sheets"]]
        payload = {"video_id": video_id, "contact_sheets": paths,
                   "frame_timestamps": [f["timestamp"] for f in index["frames"]],
                   "instruction": "Describe up to three concrete, visibly observable, temporally localizable actions. Return only JSON events with query, evidence_timestamps, evidence_sheets, coarse_start, coarse_end, visible_subject, visible_action, visible_object. Do not use source metadata or infer audio."}
        result = subprocess.run(self.command, input=json.dumps(payload), text=True,
                                capture_output=True, timeout=600)
        if result.returncode:
            raise RuntimeError(f"Local visual provider failed: {result.stderr[-400:]}")
        output = json.loads(result.stdout)
        if output.get("video_id") != video_id or not isinstance(output.get("events"), list):
            raise ValueError("Local provider returned invalid event payload")
        return output["events"]


def validate_events(video_id: str, events: list[dict], duration: float) -> list[dict]:
    index = read_json(VISUAL_CACHE / video_id / "metadata.json")
    sheets = set(index["contact_sheets"])
    sample_times = {round(x["timestamp"], 3) for x in index["frames"]}
    seen = set()
    valid = []
    for event in events[:3]:
        query = event.get("query", "").strip()
        key = query.casefold()
        if key in seen or not query:
            continue
        seen.add(key)
        references = event.get("evidence_sheets", [])
        times = event.get("evidence_timestamps", [])
        if not references or not set(references) <= sheets or not times or not all(
            round(float(t), 3) in sample_times for t in times
        ):
            raise ValueError("Query lacks real cached frame/contact-sheet evidence")
        start, end = float(event["coarse_start"]), float(event["coarse_end"])
        if not 0 <= start < end <= duration:
            raise ValueError("Invalid coarse visual event phase")
        if not all(event.get(k, "").strip() for k in
                   ("visible_subject", "visible_action", "visible_object")):
            raise ValueError("Incomplete visible event description")
        valid.append(event)
    return valid


def require_agent_inspection(video_id: str, *, work=None) -> None:
    """For local VLM output, require a separate agent record of viewed sheets."""
    path = (work or WORK) / "visual_index" / video_id / "agent_inspection.json"
    if not path.is_file():
        raise ValueError("Local VLM output requires agent inspection of contact sheets")
    record = read_json(path)
    index = read_json(VISUAL_CACHE / video_id / "metadata.json")
    if (record.get("inspection_method") != "agent_viewed_contact_sheets"
            or set(record.get("inspected_contact_sheets", [])) != set(index["contact_sheets"])):
        raise ValueError("Agent contact-sheet inspection coverage is incomplete")
