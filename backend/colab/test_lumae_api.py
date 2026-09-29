"""Contract tests use a fake predictor only; they do not claim model inference."""

import json
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from backend.colab import lumae_api as api


QUERIES = [
    {"id": "hook", "query": "An opening", "intent": "hook_candidate"},
    {"id": "demo", "query": "Product demonstration", "intent": "demo_moment"},
    {"id": "result", "query": "Product result", "intent": "benefit_moment"},
    {"id": "other", "query": "A custom question", "intent": "custom"},
]


def test_manifest_and_query_validation():
    assert api.MANIFEST["schema_version"] == "1.0"
    assert api.MANIFEST["benchmark"]["MR_R1_0_5"] == 53.23
    assert api.validate_queries(json.dumps(QUERIES)) == QUERIES
    for bad in ("[]", "no JSON", json.dumps([{"id": "x", "query": " ", "intent": "custom"}]),
                json.dumps([QUERIES[0], QUERIES[0]])):
        with pytest.raises(api.ApiFailure) as error:
            api.validate_queries(bad)
        assert error.value.code == "INVALID_QUERY"


def test_raw_conversion_and_interpretation():
    raw = [
        {"query": "An opening", "pred_relevant_windows": [[7, 15, .9], [1, 3, .8]], "pred_saliency_scores": [-.8, .3]},
        {"query": "Product demonstration", "pred_relevant_windows": [[7.7, 15.46, .9275]], "pred_saliency_scores": [-.5, .4]},
        {"query": "Product result", "pred_relevant_windows": [[8, 17.06, .999]], "pred_saliency_scores": [-.9, .2]},
        {"query": "A custom question", "pred_relevant_windows": [], "pred_saliency_scores": []},
    ]
    predictions = api.adapt_predictions(raw, QUERIES, 19.133)
    assert predictions[0]["raw_saliency_scores"] == [-.8, .3]
    assert predictions[1]["windows"][0] == {"start": 7.7, "end": 15.46, "confidence": .9275}
    result = api.interpretation(predictions, 19.133)
    assert result["hook"]["candidate"] == {"start": 1.0, "end": 3.0, "confidence": .8}
    assert result["core_action_region"] == {"provenance": "DERIVED", "start": 8.0, "end": 15.46}
    predictions[0]["windows"] = [{"start": 7.0, "end": 15.0, "confidence": .9}]
    assert api.interpretation(predictions, 19.133)["hook"]["candidate"] is None


def test_bad_raw_output_and_error_serialization():
    with pytest.raises(api.ApiFailure) as error:
        api.adapt_predictions([{"query": "An opening", "pred_relevant_windows": [[0, 200, .9]],
                                "pred_saliency_scores": [-1]}], QUERIES[:1], 20)
    assert error.value.code == "MALFORMED_RESPONSE"
    response = api.error_response(api.ApiFailure("INVALID_QUERY", "Bad query"))
    assert response.status_code == 400
    assert json.loads(response.body)["error"]["retryable"] is False


def test_api_smoke_and_cleanup(monkeypatch):
    class FakePredictor:
        device = "cpu"
        calls = 0
        paths = []

        def localize_moment(self, video_path, query_list):
            self.calls += 1
            assert Path(video_path).exists()
            self.paths.append(video_path)
            return [{"query": text, "pred_relevant_windows": [[1, 3, .8]],
                     "pred_saliency_scores": [-.8, .4]} for text in query_list]

    fake = FakePredictor()
    monkeypatch.setattr(api.runtime, "predictor", fake)
    monkeypatch.setattr(api.runtime, "device", "cpu")
    monkeypatch.setattr(api, "probe_video", lambda path, name: {
        "filename": name, "duration_seconds": 10, "width": 576, "height": 1024, "codec": "h264"})
    monkeypatch.setattr(api.tempfile, "tempdir", str(Path.cwd()))
    with TestClient(api.app) as client:
        # Lifespan loader is bypassed for this fake smoke test.
        assert client.get("/health").json()["model_loaded"] is True
        assert client.get("/api/v1/model").json() == api.MANIFEST
        for _ in range(2):
            response = client.post("/api/v1/analyze", data={"queries": json.dumps(QUERIES[:1])},
                                   files={"video": ("test.mp4", b"test", "video/mp4")})
            assert response.status_code == 200
            assert response.json()["predictions"][0]["raw_saliency_scores"] == [-.8, .4]
        assert fake.calls == 2
        assert all(not Path(path).exists() for path in fake.paths)
        invalid = client.post("/api/v1/analyze", data={"queries": "[]"},
                              files={"video": ("test.mp4", b"test", "video/mp4")})
        assert invalid.status_code == 400
        assert invalid.json()["error"]["code"] == "INVALID_QUERY"


def test_duration_guard_before_inference(monkeypatch):
    monkeypatch.setattr(api, "probe_video", lambda path, name: {
        "filename": name, "duration_seconds": 151, "width": 1, "height": 1, "codec": "h264"})
    with TestClient(api.app) as client:
        response = client.post("/api/v1/analyze", data={"queries": json.dumps(QUERIES[:1])},
                               files={"video": ("test.mp4", b"test", "video/mp4")})
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "VIDEO_TOO_LONG"
