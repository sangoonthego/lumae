"""API v1 boundary for the official Moment-DETR custom-video predictor."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import json
import logging
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from threading import Lock
from unittest.mock import patch
from uuid import uuid4

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

LOG = logging.getLogger("lumae.api")
MANIFEST = json.loads(Path(__file__).with_name("model_manifest.json").read_text(encoding="utf-8"))
MODEL = MANIFEST["model"]
HOOK_EARLY_RATIO = 0.35
HOOK_MIN_CONFIDENCE = 0.50
HOOK_MIN_EARLY_OVERLAP = 0.50
SAMPLE_TOLERANCE_SECONDS = 2.0
INTENTS = {"hook_candidate", "demo_moment", "benefit_moment", "custom"}
ERROR_STATUSES = {
    "INVALID_QUERY": 400, "UNSUPPORTED_VIDEO": 400, "VIDEO_TOO_LONG": 422,
    "MODEL_UNAVAILABLE": 503, "ANALYSIS_TIMEOUT": 504,
    "MALFORMED_RESPONSE": 502, "UNKNOWN_ERROR": 500,
}


class ApiFailure(Exception):
    def __init__(self, code: str, message: str, retryable: bool = False):
        super().__init__(message)
        self.code, self.message, self.retryable = code, message, retryable


def error_response(failure: ApiFailure, status: int | None = None) -> JSONResponse:
    return JSONResponse(
        status_code=status or ERROR_STATUSES[failure.code],
        content={"schema_version": "1.0", "error": {
            "code": failure.code, "message": failure.message, "retryable": failure.retryable,
        }},
    )


def validate_queries(value: str) -> list[dict]:
    try:
        queries = json.loads(value)
    except (TypeError, ValueError) as exc:
        raise ApiFailure("INVALID_QUERY", "Queries must be a JSON array.") from exc
    if not isinstance(queries, list) or not queries:
        raise ApiFailure("INVALID_QUERY", "At least one query is required.")
    ids: set[str] = set()
    for item in queries:
        if not isinstance(item, dict) or not isinstance(item.get("id"), str) or not item["id"].strip():
            raise ApiFailure("INVALID_QUERY", "Each query needs a non-empty id.")
        if item["id"] in ids:
            raise ApiFailure("INVALID_QUERY", "Query ids must be unique.")
        ids.add(item["id"])
        if not isinstance(item.get("query"), str) or not item["query"].strip():
            raise ApiFailure("INVALID_QUERY", "Each query needs non-empty text.")
        if item.get("intent") not in INTENTS:
            raise ApiFailure("INVALID_QUERY", "Query intent is unsupported.")
    return queries


def probe_video(path: str, filename: str) -> dict:
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=codec_name,width,height,codec_type",
             "-of", "json", path], capture_output=True, text=True, check=True, timeout=30,
        )
        data = json.loads(result.stdout)
        video = next(s for s in data["streams"] if s.get("codec_type") == "video")
        duration = float(data["format"]["duration"])
        width, height = int(video["width"]), int(video["height"])
        codec = str(video["codec_name"])
        if not math.isfinite(duration) or duration <= 0 or width <= 0 or height <= 0 or not codec:
            raise ValueError("Invalid video metadata")
        return {"filename": filename, "duration_seconds": duration, "width": width,
                "height": height, "codec": codec}
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, StopIteration, TypeError) as exc:
        raise ApiFailure("UNSUPPORTED_VIDEO", "Could not read MP4 video metadata.") from exc


def finite_number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ApiFailure("MALFORMED_RESPONSE", "Model returned non-finite or invalid values.")
    return float(value)


def adapt_predictions(raw: object, queries: list[dict], duration: float) -> list[dict]:
    """Convert official run_on_video/run.py output once, preserving saliency values."""
    if not isinstance(raw, list) or len(raw) != len(queries):
        raise ApiFailure("MALFORMED_RESPONSE", "Model returned an unexpected query count.")
    result = []
    for item, query in zip(raw, queries):
        if not isinstance(item, dict) or item.get("query") != query["query"]:
            raise ApiFailure("MALFORMED_RESPONSE", "Model query output did not match the request.")
        windows_raw = item.get("pred_relevant_windows")
        saliency_raw = item.get("pred_saliency_scores")
        if not isinstance(windows_raw, list) or not isinstance(saliency_raw, list):
            raise ApiFailure("MALFORMED_RESPONSE", "Model omitted temporal output or saliency.")
        windows = []
        for row in windows_raw:
            if not isinstance(row, (list, tuple)) or len(row) != 3:
                raise ApiFailure("MALFORMED_RESPONSE", "Model returned an invalid temporal window.")
            start, end, confidence = (finite_number(x) for x in row)
            if not (0 <= start < end <= duration + SAMPLE_TOLERANCE_SECONDS and 0 <= confidence <= 1):
                raise ApiFailure("MALFORMED_RESPONSE", "Model returned an out-of-range temporal window.")
            windows.append({"start": start, "end": end, "confidence": confidence})
        saliency = [finite_number(x) for x in saliency_raw]
        result.append({"query_id": query["id"], "intent": query["intent"], "query": query["query"],
                       "windows": windows, "raw_saliency_scores": saliency,
                       "saliency_clip_duration_seconds": MODEL["temporal_resolution_seconds"]})
    return result


def interpretation(predictions: list[dict], duration: float) -> dict:
    hook = None
    demo = None
    benefit = None
    for prediction in predictions:
        windows = prediction["windows"]
        intent = prediction["intent"]
        if intent == "hook_candidate" and hook is None:
            for window in windows:
                overlap = max(0, min(window["end"], duration * HOOK_EARLY_RATIO) - window["start"])
                if window["confidence"] >= HOOK_MIN_CONFIDENCE and overlap / (window["end"] - window["start"]) >= HOOK_MIN_EARLY_OVERLAP:
                    hook = window.copy()
                    break
        if intent == "demo_moment" and demo is None and windows:
            demo = {"provenance": "MODEL_OUTPUT", **windows[0]}
        if intent == "benefit_moment" and benefit is None and windows:
            benefit = {"provenance": "MODEL_OUTPUT", **windows[0]}
    core = None
    if demo and benefit:
        start, end = max(demo["start"], benefit["start"]), min(demo["end"], benefit["end"])
        if start < end:
            core = {"provenance": "DERIVED", "start": start, "end": end}
    return {
        "hook": {"provenance": "HEURISTIC", "status": "detected" if hook else "not_confidently_detected", "candidate": hook},
        "demo_moment": demo, "result_moment": benefit, "core_action_region": core,
    }


def load_predictor(root: Path, checkpoint: Path):
    """Load official predictor; scope the PyTorch 2.6 compatibility override to this trusted checkpoint."""
    if not root.is_dir() or not checkpoint.is_file():
        raise FileNotFoundError("Moment-DETR root or trusted checkpoint is unavailable")
    sys.path.insert(0, str(root))
    from run_on_video import run, model_utils  # type: ignore[import-not-found]
    import torch  # type: ignore[import-not-found]

    original_builder = run.build_inference_model
    original_load = torch.load
    trusted_path = checkpoint.resolve()

    def trusted_builder(path, **kwargs):
        def scoped_load(candidate, *args, **load_kwargs):
            if Path(candidate).resolve() == trusted_path:
                load_kwargs["weights_only"] = False
            return original_load(candidate, *args, **load_kwargs)
        with patch.object(model_utils.torch, "load", side_effect=scoped_load):
            return original_builder(path, **kwargs)

    # The official predictor imports build_inference_model into run.py. Replace only during construction.
    with patch.object(run, "build_inference_model", side_effect=trusted_builder):
        return run.MomentDETRPredictor(ckpt_path=str(trusted_path), device="cuda" if torch.cuda.is_available() else "cpu")


class ModelRuntime:
    def __init__(self):
        self.predictor = None
        self.device = None
        self.lock = Lock()

    def load(self):
        root = Path(os.getenv("MOMENT_DETR_ROOT", "/content/moment_detr"))
        checkpoint = Path(os.getenv("MOMENT_DETR_CHECKPOINT", str(root / "run_on_video/moment_detr_ckpt/model_best.ckpt")))
        try:
            self.predictor = load_predictor(root, checkpoint)
            self.device = self.predictor.device
            LOG.info("Moment-DETR predictor loaded once; device=%s", self.device)
        except Exception:
            LOG.exception("Moment-DETR predictor load failed")

    def predict(self, video_path: str, queries: list[dict]):
        if self.predictor is None:
            raise ApiFailure("MODEL_UNAVAILABLE", "Moment-DETR is not ready.", True)
        with self.lock:
            return self.predictor.localize_moment(video_path=video_path, query_list=[q["query"] for q in queries])


runtime = ModelRuntime()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    await asyncio.to_thread(runtime.load)
    yield


app = FastAPI(lifespan=lifespan)
origins = [x.strip() for x in os.getenv("LUMAE_ALLOWED_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",") if x.strip()]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["GET", "POST"], allow_headers=["*"])


@app.exception_handler(ApiFailure)
async def api_failure_handler(_request: Request, failure: ApiFailure):
    return error_response(failure)


@app.exception_handler(RequestValidationError)
async def request_validation_handler(_request: Request, _failure: RequestValidationError):
    return error_response(ApiFailure("INVALID_QUERY", "Required multipart fields are missing or invalid."), 422)


@app.exception_handler(Exception)
async def unexpected_handler(_request: Request, failure: Exception):
    LOG.exception("Unexpected API failure", exc_info=failure)
    return error_response(ApiFailure("UNKNOWN_ERROR", "Analysis failed unexpectedly.", True))


@app.get("/health")
def health():
    if runtime.predictor is None:
        return {"status": "degraded", "model_loaded": False}
    return {"status": "ok", "model_loaded": True, "model_id": MODEL["id"], "device": runtime.device}


@app.get("/api/v1/model")
def model_info():
    return MANIFEST


@app.post("/api/v1/analyze")
async def analyze(video: UploadFile = File(...), queries: str = Form(...)):
    started = time.perf_counter()
    LOG.info("Analysis request received")
    query_items = validate_queries(queries)
    filename = Path(video.filename or "").name
    if not filename.lower().endswith(".mp4") or video.content_type not in ("video/mp4", "application/octet-stream"):
        raise ApiFailure("UNSUPPORTED_VIDEO", "Upload an MP4 video.")
    path = None
    try:
        with tempfile.NamedTemporaryFile(prefix="lumae_", suffix=".mp4", delete=False) as temp:
            path = temp.name
            while chunk := await video.read(1024 * 1024):
                temp.write(chunk)
        metadata = await asyncio.to_thread(probe_video, path, filename)
        if metadata["duration_seconds"] > MODEL["max_video_duration_seconds"]:
            raise ApiFailure("VIDEO_TOO_LONG", "Video duration exceeds the 150 second model limit.")
        LOG.info("Video validation time %.3fs", time.perf_counter() - started)
        inference_start = time.perf_counter()
        # Keep the upload alive until GPU work finishes. Browser timeout is configured separately;
        # cancelling this thread would otherwise delete the MP4 while CUDA still reads it.
        inference_task = asyncio.create_task(asyncio.to_thread(runtime.predict, path, query_items))
        try:
            raw = await asyncio.shield(inference_task)
        except asyncio.CancelledError:
            # Keep the temporary upload until the worker has stopped reading it.
            try:
                await inference_task
            finally:
                raise
        LOG.info("Feature extraction and inference time %.3fs", time.perf_counter() - inference_start)
        interpretation_start = time.perf_counter()
        predictions = adapt_predictions(raw, query_items, metadata["duration_seconds"])
        interpreted = interpretation(predictions, metadata["duration_seconds"])
        LOG.info("Interpretation time %.3fs; total request time %.3fs", time.perf_counter() - interpretation_start, time.perf_counter() - started)
        return {"schema_version": "1.0", "analysis_id": "analysis_" + uuid4().hex,
                "model": {"id": MODEL["id"], "stage": MODEL["stage"]},
                "video": metadata, "predictions": predictions, "interpretation": interpreted}
    finally:
        await video.close()
        if path:
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=os.getenv("LUMAE_API_HOST", "127.0.0.1"),
                port=int(os.getenv("LUMAE_API_PORT", "8000")), workers=1)
