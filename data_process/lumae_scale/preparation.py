"""Source preflight and validated producer cache; workers never write selection."""
from __future__ import annotations

from enum import Enum
import errno
import re
from urllib.parse import urlparse
import requests
import os

from .cache_identity import SourceIdentity
from .models import atomic_json, digest, read_json
from .repo_paths import to_repo_relative
from .source_pool import download, probe


class FailureKind(str, Enum):
    CONTENT_REJECTED = "CONTENT_REJECTED"
    RETRYABLE_NETWORK_ERROR = "RETRYABLE_NETWORK_ERROR"
    RETRYABLE_IO_ERROR = "RETRYABLE_IO_ERROR"
    PERMANENT_SOURCE_FAILURE = "PERMANENT_SOURCE_FAILURE"
    DUPLICATE = "DUPLICATE"


class SourceFailure(ValueError):
    def __init__(self, kind: FailureKind, message: str):
        self.kind = kind
        super().__init__(message)


def classify_failure(exc: Exception) -> FailureKind:
    if isinstance(exc,SourceFailure): return exc.kind
    if isinstance(exc,requests.HTTPError) and exc.response is not None and exc.response.status_code in (404,410):
        return FailureKind.PERMANENT_SOURCE_FAILURE
    if isinstance(exc,(requests.RequestException,TimeoutError,ConnectionError)):
        return FailureKind.RETRYABLE_NETWORK_ERROR
    if isinstance(exc,OSError): return FailureKind.RETRYABLE_IO_ERROR
    return FailureKind.CONTENT_REJECTED


def metadata_preflight(row: dict, excluded_ids: set[str]) -> None:
    vid = row.get("source_video_id","")
    if vid in excluded_ids: raise SourceFailure(FailureKind.DUPLICATE,"Source already belongs to frozen D200")
    if not re.fullmatch(r"[a-f0-9]{32}",vid) or row.get("target_name") != vid+".mp4":
        raise SourceFailure(FailureKind.PERMANENT_SOURCE_FAILURE,"Invalid source ID or filename")
    u = urlparse(row.get("source_url",""))
    if u.scheme != "https" or u.hostname != "video.adsoftheworld.com":
        raise SourceFailure(FailureKind.PERMANENT_SOURCE_FAILURE,"Invalid AdsQA source URL")


def prepare_source(row, layout, *, excluded_ids, excluded_hashes, timer):
    with timer.stage("source_preflight"):
        metadata_preflight(row,excluded_ids)
    output = layout.work / "source_cache" / (row["source_video_id"]+".json")
    video = layout.videos / row["target_name"]
    if output.is_file() and video.is_file():
        try:
            cache = read_json(output)
            valid = (cache["bytes"]==video.stat().st_size and cache["source_sha256"]==digest(video)
                     and cache["identity"]==SourceIdentity(row["source_video_id"],cache["source_sha256"]).fingerprint()
                     and cache["probe_version"]=="ffprobe_duration_v1" and 0<cache["duration"]<=150)
        except (OSError,ValueError,KeyError): valid=False
        if valid:
            if cache["source_sha256"] in excluded_hashes:
                raise SourceFailure(FailureKind.DUPLICATE,"Duplicate frozen or accepted source content")
            timer.mark_cache("video",True,cache["identity"])
            timer.mark_cache("probe",True,cache["identity"])
            return video,cache
    timer.mark_cache("video",video.is_file() and video.stat().st_size>0)
    with timer.stage("download"):
        bridge = os.environ.get("LUMAE_D500_DOWNLOAD_BRIDGE")
        if bridge and not video.is_file():
            response = requests.post(bridge.rstrip("/")+"/fetch",
                                     json={"source_video_id":row["source_video_id"],"source_url":row["source_url"]},
                                     timeout=240)
            if response.status_code == 404:
                raise SourceFailure(FailureKind.PERMANENT_SOURCE_FAILURE, response.json().get("reason","404"))
            response.raise_for_status()
            if response.json().get("status") != "READY" or not video.is_file():
                raise OSError("Network fetch bridge did not persist source video")
        video = download(row,video_dir=layout.videos)
    with timer.stage("ffprobe"):
        duration = probe(video)
    if duration>150: raise SourceFailure(FailureKind.CONTENT_REJECTED,"duration over 150 seconds")
    sha=digest(video)
    if sha in excluded_hashes: raise SourceFailure(FailureKind.DUPLICATE,"Duplicate frozen or accepted source content")
    cache={"source_video_id":row["source_video_id"],"source_sha256":sha,"duration":duration,
           "bytes":video.stat().st_size,"video_path":to_repo_relative(video,root=layout.root),
           "identity":SourceIdentity(row["source_video_id"],sha).fingerprint(),"probe_version":"ffprobe_duration_v1"}
    atomic_json(output,cache)
    return video,cache
