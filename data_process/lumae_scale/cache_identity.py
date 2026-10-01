"""Stage-aware cache identity, fingerprinting, and invalidation rules for D200.

Replaces naive video_id/filename cache keys with cryptographically anchored,
stage-isolated fingerprints so changes in downstream components (e.g. verifiers)
do NOT invalidate expensive upstream artifacts (e.g. downloaded videos or visual frames).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from data_process.annotation.assisted_deployment import ALGORITHM_SHA256
from .temporal_labeler import WRAPPER_VERSION

# Frozen reference algorithm hash
EXPECTED_SEMANTIC_V3_SHA256 = ALGORITHM_SHA256


def deterministic_fingerprint(data: Dict[str, Any]) -> str:
    """Compute deterministic SHA256 of sorted JSON representation."""
    encoded = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class SourceIdentity:
    source_video_id: str
    source_sha256: str

    def fingerprint(self) -> str:
        return deterministic_fingerprint(asdict(self))


@dataclass(frozen=True)
class VisualIndexIdentity:
    source_sha256: str
    sampling_step_seconds: float = 2.0
    frame_filter: str = "fps=1/2,scale=320:-2"
    extraction_version: str = "d200_visual_index_v1"

    def fingerprint(self) -> str:
        return deterministic_fingerprint(asdict(self))


@dataclass(frozen=True)
class ClipCacheIdentity:
    source_sha256: str
    visual_index_sha256: str
    clip_model: str = "ViT-B/32"
    clip_version: str = "openai_clip_vit_b32_v1"

    def fingerprint(self) -> str:
        return deterministic_fingerprint(asdict(self))


@dataclass(frozen=True)
class QueryGenIdentity:
    visual_index_fingerprint: str
    provider: str = "AgentEvidenceProvider"
    model_id: str = "GPT-6"
    generation_version: str = "d200_pipeline_v1"

    def fingerprint(self) -> str:
        return deterministic_fingerprint(asdict(self))


@dataclass(frozen=True)
class SemanticCacheIdentity:
    query_sha256: str
    clip_features_sha256: str
    semantic_v3_algorithm_sha256: str = EXPECTED_SEMANTIC_V3_SHA256

    def fingerprint(self) -> str:
        return deterministic_fingerprint(asdict(self))


@dataclass(frozen=True)
class RefinementCacheIdentity:
    semantic_prediction_fingerprint: str
    clip_features_sha256: str
    boundary_wrapper_version: str = WRAPPER_VERSION

    def fingerprint(self) -> str:
        return deterministic_fingerprint(asdict(self))


@dataclass(frozen=True)
class VerifierCacheIdentity:
    query_sha256: str
    window_start: float
    window_end: float
    verifier_provider: str
    verifier_model_id: str
    verifier_version: str = "d200_verifier_v1"

    def fingerprint(self) -> str:
        return deterministic_fingerprint(asdict(self))


# Invalidation dependency graph: stage -> set of upstream stages it depends on
STAGE_DEPENDENCIES = {
    "source": set(),
    "visual_index": {"source"},
    "clip": {"source", "visual_index"},
    "query_gen": {"source", "visual_index"},
    "semantic": {"source", "visual_index", "clip", "query_gen"},
    "refinement": {"source", "visual_index", "clip", "query_gen", "semantic"},
    "verifier": {"source", "visual_index", "clip", "query_gen", "semantic", "refinement"},
}


def does_invalidation_affect(changed_stage: str, target_stage: str) -> bool:
    """Return True if changing `changed_stage` invalidates `target_stage`."""
    if changed_stage == target_stage:
        return True
    return changed_stage in STAGE_DEPENDENCIES.get(target_stage, set())


def check_cache_validity(
    stage: str,
    cached_metadata: Dict[str, Any],
    expected_identity: Any,
) -> Tuple[bool, str]:
    """Check if cached metadata matches the expected stage identity fingerprint."""
    expected_fp = expected_identity.fingerprint()
    stored_fp = cached_metadata.get(f"{stage}_fingerprint")

    if not stored_fp:
        return False, f"Missing {stage}_fingerprint in cache metadata"

    if stored_fp != expected_fp:
        return False, f"Cache fingerprint mismatch for stage '{stage}': expected {expected_fp}, found {stored_fp}"

    return True, "VALID"
