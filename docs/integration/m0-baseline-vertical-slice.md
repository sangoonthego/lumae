# M0 baseline vertical slice — integration record

Date: 2026-09-17 (Asia/Saigon). API schema: 1.0. Model ID: `moment-detr-clip-baseline`.

## Current status

- Gate 0: `MANUAL_QA_UNAVAILABLE`; the computer-use environment exposes no browser.
- Backend: API adapter and local contract tests implemented. No Colab runtime or checkpoint was available in this workspace, so real predictor startup and GPU output were not verified.
- Frontend: real API client already uses API v1 and the shared UI renderer. Browser analysis timeout is now configurable. No live Colab URL was supplied, so real frontend connection is unverified.
- P7/M0: blocked on a running Colab predictor, the mini-vacuum video, and interactive browser verification. The bundled bottle still-frame MP4 is only a mock interaction asset and is not a substitute.

## Local API evidence

- A local Uvicorn process returned `GET /health` HTTP 200 with `status: degraded`, `model_loaded: false` because the checkpoint was absent.
- `GET /api/v1/model` returned HTTP 200 with the manifest model ID. The frontend runtime parser accepted this exact manifest in an automated test.
- `POST /api/v1/analyze` with the bundled MP4 and a valid query returned HTTP 503 `MODEL_UNAVAILABLE`, without an analysis ID or fake predictions. A missing multipart payload returned HTTP 422 in the standard error schema.
- Backend contract tests used a fake predictor to exercise the successful response shape, two sequential requests, temporary-file cleanup, and error paths. These results are not real model inference.

## Real test still required

Configure `NEXT_PUBLIC_LUMAE_USE_MOCK=false` and `NEXT_PUBLIC_LUMAE_API_BASE_URL` to the current HTTPS tunnel. Record `/health` readiness, `/api/v1/model` validation, and `/api/v1/analyze` status for a real approximately 19.13-second portrait mini-vacuum MP4. Submit Hook, Demo, Result, and arbitrary custom queries. Record analysis ID, video metadata, returned windows, raw saliency lengths and sample negative values, Hook status, and Demo/Result/Core Action interpretation. Repeat analysis and inspect server logs for one model load. Verify timeline, seeking, query switching, saliency, benchmark independence, and offline/malformed-response states in a browser. Never paste private uploaded video, credentials, or a temporary tunnel URL into this file.

Known baseline limits: Colab lifetime and Quick Tunnel URL are temporary; visual-only CLIP features sample about every 2 seconds, have a 150-second limit and weak short-moment benchmark performance; Hook is heuristic; no per-video quantitative ground truth is available.
