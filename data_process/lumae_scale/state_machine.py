"""D200 annotation candidate state machine and transition lifecycle guards.

Explicitly models candidate stages from selection to acceptance, with guards
preventing illegal transitions and protecting network retries from false rejections.
"""

from __future__ import annotations

import logging
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class CandidateStatus(str, Enum):
    UNATTEMPTED = "unattempted"
    DOWNLOADING = "downloading"
    SOURCE_READY = "downloaded"  # Repository standard: "downloaded"
    VISUAL_INDEX_READY = "visual_index_ready"
    AWAITING_VISUAL_INSPECTION = "awaiting_visual_inspection"
    ANNOTATION_CANDIDATE_READY = "annotation_candidate_ready"
    AWAITING_INTERVAL_VERIFICATION = "awaiting_interval_verification"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    DOWNLOAD_DEFERRED = "download_deferred"


# Allowed state transitions: from_state -> set of valid next states
VALID_TRANSITIONS: Dict[CandidateStatus, Set[CandidateStatus]] = {
    CandidateStatus.UNATTEMPTED: {
        CandidateStatus.DOWNLOADING,
        CandidateStatus.SOURCE_READY,
        CandidateStatus.DOWNLOAD_DEFERRED,
        CandidateStatus.REJECTED,
    },
    CandidateStatus.DOWNLOADING: {
        CandidateStatus.SOURCE_READY,
        CandidateStatus.DOWNLOAD_DEFERRED,
        CandidateStatus.UNATTEMPTED,  # Retry
        CandidateStatus.REJECTED,  # Only for corrupt video content or duration > 150s
    },
    CandidateStatus.DOWNLOAD_DEFERRED: {
        CandidateStatus.DOWNLOADING,
        CandidateStatus.UNATTEMPTED,
        CandidateStatus.SOURCE_READY,
    },
    CandidateStatus.SOURCE_READY: {
        CandidateStatus.DOWNLOAD_DEFERRED,
        CandidateStatus.VISUAL_INDEX_READY,
        CandidateStatus.REJECTED,  # Frame extraction failure / corrupt file
    },
    CandidateStatus.VISUAL_INDEX_READY: {
        CandidateStatus.DOWNLOAD_DEFERRED,
        CandidateStatus.AWAITING_VISUAL_INSPECTION,
        CandidateStatus.ANNOTATION_CANDIDATE_READY,
        CandidateStatus.REJECTED,
    },
    CandidateStatus.AWAITING_VISUAL_INSPECTION: {
        CandidateStatus.DOWNLOAD_DEFERRED,
        CandidateStatus.ANNOTATION_CANDIDATE_READY,
        CandidateStatus.AWAITING_INTERVAL_VERIFICATION,
        CandidateStatus.REJECTED,  # Agent rejects video (no valid event)
    },
    CandidateStatus.ANNOTATION_CANDIDATE_READY: {
        CandidateStatus.DOWNLOAD_DEFERRED,
        CandidateStatus.AWAITING_INTERVAL_VERIFICATION,
        CandidateStatus.ACCEPTED,
        CandidateStatus.REJECTED,
    },
    CandidateStatus.AWAITING_INTERVAL_VERIFICATION: {
        CandidateStatus.DOWNLOAD_DEFERRED,
        CandidateStatus.ACCEPTED,
        CandidateStatus.REJECTED,  # Verifier failed
        CandidateStatus.ANNOTATION_CANDIDATE_READY,  # Re-evaluate
    },
    CandidateStatus.ACCEPTED: set(),  # Terminal state
    CandidateStatus.REJECTED: set(),  # Terminal state
}


def can_transition(current: str, target: str) -> bool:
    """Validate whether transitioning from current to target status is permitted."""
    try:
        curr_enum = CandidateStatus(current)
        target_enum = CandidateStatus(target)
    except ValueError:
        return False
    return target_enum in VALID_TRANSITIONS.get(curr_enum, set())


def validate_transition(current: str, target: str, video_id: str = "") -> None:
    """Raise ValueError if the transition is illegal."""
    if current == target:
        return
    if not can_transition(current, target):
        raise ValueError(
            f"Illegal state machine transition for {video_id or 'candidate'}: "
            f"'{current}' -> '{target}'"
        )


def is_network_or_io_retryable(error: Exception) -> bool:
    """Return True if the exception represents a transient network/connection error."""
    import requests
    if isinstance(error, (requests.RequestException, ConnectionError, TimeoutError)):
        return True
    err_str = str(error).casefold()
    retryable_markers = [
        "connection reset",
        "connection refused",
        "timed out",
        "timeout",
        "remote end closed",
        "network is unreachable",
        "name or service not known",
        "temporary failure in name resolution",
        "ssl: decryption_failed",
    ]
    return any(marker in err_str for marker in retryable_markers)
