# LUMAE

LUMAE is a research-to-product demo for query-conditioned short-form video understanding. The repository contains a Next.js frontend, a versioned mock/real API boundary, and a Colab FastAPI adapter for the existing Moment-DETR predictor. A real Colab runtime and checkpoint are required for inference.

## Run the mock flow

```bash
npm install
# In .env.local, set NEXT_PUBLIC_LUMAE_USE_MOCK=true
npm run dev
```

Open `http://localhost:3000`. The first screen is empty. Upload an MP4 or choose **Explore sample demo**, select a query, and press **Run Analysis**. Mock predictions pass through the same API v1 decoder as real HTTP responses. A visible **Demo Data Mode** badge identifies mock results. In development or explicit mock mode, the query panel offers standard, confident Hook, no predictions, `VIDEO_TOO_LONG`, and malformed-response scenarios. A failed real API request never falls back to mock data.

The bundled sample MP4 is a 30-second still-frame video generated from an illustrative product shot. Its prediction data is for interaction testing, not a real model evaluation. The image was generated with the built-in image tool using an editorial prompt of a hand holding a white LUMAE bottle in a modern workspace.

## API integration

Copy `.env.example` to `.env.local` and configure `NEXT_PUBLIC_LUMAE_API_BASE_URL` for the running backend. Keep `.env.local` out of version control. Restart Next.js after changing `NEXT_PUBLIC_` values. The frontend expects:

- `GET /api/v1/model` for model metadata and QVHighlights benchmark values.
- `POST /api/v1/analyze` with multipart `video` and JSON-string `queries` fields for current video inference.

The stable [semantics](docs/contracts/lumae-semantics-v1.md) and [API v1 contract](docs/contracts/lumae-api-v1.md) define provenance, named temporal windows, errors, versioning, and nullable Hook behavior. The [P0 audit](docs/contracts/p0-audit.md) records the starting architecture and refactor findings. Canonical domain types are in `src/lib/analysisTypes.ts`; `src/lib/lumaeApi.ts` owns transport, validation, and snake_case-to-camelCase adaptation.

Optional provenance links use `NEXT_PUBLIC_LUMAE_PROJECT_REPO_URL`, `NEXT_PUBLIC_LUMAE_COLAB_URL`, and `NEXT_PUBLIC_MOMENT_DETR_REPO_URL`.

The [Colab adapter guide](backend/colab/README.md) covers model paths, startup, CORS, and a temporary HTTPS tunnel. The [M0 integration record](docs/integration/m0-baseline-vertical-slice.md) tracks the real vertical slice gate.
## Research safeguards

Benchmark metrics are model-level data and never describe the uploaded video's accuracy. Window confidence is a ranking/matching signal. Raw saliency may be negative and is preserved; normalization is derived only for display. Hook Candidate is a nullable Lumae heuristic. The current CLIP-only baseline is primarily visual, samples at about two-second resolution, and accepts videos up to 150 seconds.

## Research Data Pipeline

Offline research data preparation and model fine-tuning are separated from product serving:

- `data_process/`: CPU-only offline dataset preparation, schema validation, leakage prevention, and manifest generation.
- `backend/colab/`: FastAPI model serving adapter for interactive inference.
- Future Colab notebook: GPU-side feature extraction and Moment-DETR fine-tuning.
- See the [operational boundary document](docs/research/local-colab-boundary.md) and [Stage A audit](docs/research/project-audit-stage-a.md).

## Verify

```bash
npm run lint
npm run typecheck
npm run test
npm run build
```
