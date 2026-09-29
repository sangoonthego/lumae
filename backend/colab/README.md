# Colab Moment-DETR API v1

This adapter calls the existing `run_on_video.run.MomentDETRPredictor` and its `localize_moment(video_path, query_list)` method. The official predictor returns one dictionary per query with `pred_relevant_windows` and `pred_saliency_scores`; the adapter converts these at a single boundary. [Official predictor source](https://github.com/jayleicn/moment_detr/blob/main/run_on_video/run.py)

## Start in Colab

1. Select a GPU runtime. Make the LUMAE repository available in Colab alongside the existing Moment-DETR checkout. Ensure the official or otherwise trusted local checkpoint exists. Do not commit or overwrite the checkpoint.
2. From the LUMAE repository directory, install the small API layer: `pip install -r backend/colab/requirements.txt`. Keep the working predictor and its existing PyTorch, CLIP, and ffmpeg dependencies installed. Confirm `ffprobe -version` works.
3. Set `MOMENT_DETR_ROOT` (default `/content/moment_detr`), `MOMENT_DETR_CHECKPOINT` (default `<root>/run_on_video/moment_detr_ckpt/model_best.ckpt`), and `LUMAE_ALLOWED_ORIGINS` (comma-separated frontend origins, default localhost ports 3000). Optionally set `LUMAE_API_HOST` and `LUMAE_API_PORT` (default `127.0.0.1:8000`).
4. Start one worker from the LUMAE repository directory: `python -m backend.colab.lumae_api`. Check `http://127.0.0.1:8000/health` and `http://127.0.0.1:8000/api/v1/model`. `/health` must say `model_loaded: true` before real analysis.
5. For a temporary HTTPS research demo, install [cloudflared](https://developers.cloudflare.com/tunnel/downloads/) if needed and run `cloudflared tunnel --url http://localhost:8000` in another Colab process. Copy the printed `https://...trycloudflare.com` URL to the local frontend environment. Quick Tunnel URLs expire and must never be committed. See [Cloudflare Quick Tunnels](https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/do-more-with-tunnels/trycloudflare/).

If the tunnel sees a remote frontend origin, set that exact origin in `LUMAE_ALLOWED_ORIGINS` and restart the API. The public tunnel exposes an unauthenticated research endpoint; use it only for controlled demos. There is no persistent uploaded-video storage: the request's temporary MP4 is removed after inference completes.

For local development, run the API on this machine and set `NEXT_PUBLIC_LUMAE_API_BASE_URL=http://127.0.0.1:8000` in `.env.local`. Use `NEXT_PUBLIC_LUMAE_USE_MOCK=false`, then restart Next.js; `NEXT_PUBLIC_` variables are bundled into the browser build. Set `NEXT_PUBLIC_LUMAE_ANALYSIS_TIMEOUT_MS` for the browser request deadline (default 180000). A client timeout does not cancel in-flight CUDA computation. The backend waits for inference to finish before deleting the temporary file.

## Loading and limits

FastAPI lifespan loads CLIP and Moment-DETR once. A process-local lock serializes GPU inference; run one Uvicorn worker. The official loader calls `torch.load` without an explicit `weights_only` argument. During construction only, the adapter scopes `weights_only=False` to the configured, trusted checkpoint path, accommodating PyTorch 2.6+ without changing upstream files or globally disabling safer loading. See [official loader](https://github.com/jayleicn/moment_detr/blob/main/run_on_video/model_utils.py).

`ffprobe` runs before model inference. MP4 duration must be at most 150 seconds. The official model samples at approximately 2-second intervals and has a 75-clip positional limit. Failed startup leaves `/health` degraded and analysis returns `MODEL_UNAVAILABLE`; inspect server logs for the private traceback. Model metadata remains available at `/api/v1/model` even when the runtime is degraded.

## Validation

From the LUMAE repository directory:

```bash
python -m compileall backend/colab
python -m pytest -q backend/colab/test_lumae_api.py
```

Tests use a fake predictor for contract and HTTP coverage. They do not establish real GPU inference. The next real acceptance step is a Colab request with the mini-vacuum MP4 and Hook, Demo, Result, and custom queries, followed by browser interaction and a repeated request to confirm model reuse.
