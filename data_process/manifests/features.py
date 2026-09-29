"""Feature extraction manifest representing planned video and text features."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Callable, Sequence

from ..schemas.temporal_sample import CanonicalTemporalSample


@dataclass
class VideoFeatureEntry:
    """Representation of an expected or cached video feature file."""
    ready: bool
    path: str
    source_video_path: str | None = None
    dim: int | None = None
    clip_duration_seconds: float = 2.0

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "ready": self.ready,
            "path": self.path,
            "clip_duration_seconds": self.clip_duration_seconds,
        }
        if self.source_video_path:
            d["source_video_path"] = self.source_video_path
        if self.dim is not None:
            d["dim"] = self.dim
        return d


@dataclass
class TextFeatureEntry:
    """Representation of an expected or cached query text feature file."""
    qid: int | str
    query_text: str
    ready: bool
    path: str
    dim: int | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "qid": self.qid,
            "query_text": self.query_text,
            "ready": self.ready,
            "path": self.path,
        }
        if self.dim is not None:
            d["dim"] = self.dim
        return d


@dataclass
class VideoQueriesFeatureGroup:
    """Group of video feature and associated query features."""
    vid: str
    video_feature: VideoFeatureEntry
    queries: list[TextFeatureEntry] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "vid": self.vid,
            "video_feature": self.video_feature.to_dict(),
            "queries": [q.to_dict() for q in self.queries],
        }


@dataclass
class FeatureManifest:
    """Complete manifest of planned/cached features for a dataset split."""
    version: str = "1.0"
    dataset_name: str = "ego4d_nlq"
    items: list[VideoQueriesFeatureGroup] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "dataset_name": self.dataset_name,
            "total_videos": len(self.items),
            "total_queries": sum(len(item.queries) for item in self.items),
            "items": [item.to_dict() for item in self.items],
        }

    def save(self, path: Path | str) -> None:
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path | str) -> FeatureManifest:
        dest = Path(path)
        data = json.loads(dest.read_text(encoding="utf-8"))
        items = []
        for raw in data.get("items", []):
            vf_raw = raw["video_feature"]
            vf = VideoFeatureEntry(
                ready=vf_raw["ready"],
                path=vf_raw["path"],
                source_video_path=vf_raw.get("source_video_path"),
                dim=vf_raw.get("dim"),
                clip_duration_seconds=vf_raw.get("clip_duration_seconds", 2.0),
            )
            queries = [
                TextFeatureEntry(
                    qid=q["qid"],
                    query_text=q["query_text"],
                    ready=q["ready"],
                    path=q["path"],
                    dim=q.get("dim"),
                )
                for q in raw.get("queries", [])
            ]
            items.append(VideoQueriesFeatureGroup(vid=raw["vid"], video_feature=vf, queries=queries))
        return cls(version=data.get("version", "1.0"), dataset_name=data.get("dataset_name", ""), items=items)


def generate_feature_manifest(
    samples: Sequence[CanonicalTemporalSample],
    dataset_name: str,
    video_feature_dir: str = "Features/Ego4D/video",
    text_feature_dir: str = "Features/Ego4D/text",
    video_path_resolver: Callable[[str], str | None] | None = None,
    clip_duration_seconds: float = 2.0,
    expected_dim: int = 512,  # Default CLIP ViT-B/32 feature dimension
) -> FeatureManifest:
    """Generate a planned FeatureManifest from a collection of canonical samples.

    Does not perform feature extraction; outputs relative paths and requirements.
    """
    by_vid: dict[str, list[CanonicalTemporalSample]] = defaultdict(list)
    for s in samples:
        by_vid[s.vid].append(s)

    items: list[VideoQueriesFeatureGroup] = []
    for vid, group_samples in sorted(by_vid.items()):
        source_path = video_path_resolver(vid) if video_path_resolver else None
        v_entry = VideoFeatureEntry(
            ready=False,
            path=f"{video_feature_dir.rstrip('/')}/{vid}.npz",
            source_video_path=source_path,
            dim=expected_dim,
            clip_duration_seconds=clip_duration_seconds,
        )

        seen_qids: set[str | int] = set()
        q_entries: list[TextFeatureEntry] = []
        for s in group_samples:
            if s.qid in seen_qids:
                continue
            seen_qids.add(s.qid)
            q_entries.append(
                TextFeatureEntry(
                    qid=s.qid,
                    query_text=s.query,
                    ready=False,
                    path=f"{text_feature_dir.rstrip('/')}/qid_{s.qid}.npz",
                    dim=expected_dim,
                )
            )

        items.append(VideoQueriesFeatureGroup(vid=vid, video_feature=v_entry, queries=q_entries))

    return FeatureManifest(dataset_name=dataset_name, items=items)
