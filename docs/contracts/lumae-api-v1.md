# LUMAE API contract — schema 1.0

Status: **frozen frontend boundary**. A backend can replace the current baseline with Stage A, B, or C by changing `model.id`, `model.stage`, and associated outputs. It must preserve the v1 endpoint names, wire shape, semantics, and error model. An incompatible shape requires a new API version.

Base URL: `NEXT_PUBLIC_LUMAE_API_BASE_URL`. JSON field names at the wire boundary use `snake_case`. The frontend decoder converts them once to the camelCase domain model in `src/lib/analysisTypes.ts`.

## GET /api/v1/model

Returns model-level metadata and benchmark results. It does not depend on a video or selected query.

```json
{
  "schema_version": "1.0",
  "model": {
    "id": "moment-detr-clip-baseline",
    "display_name": "Moment-DETR CLIP-only",
    "base_model": "Moment-DETR",
    "stage": "baseline",
    "temporal_resolution_seconds": 2,
    "max_video_duration_seconds": 150,
    "status": "research_demo"
  },
  "benchmark": {
    "dataset": "QVHighlights",
    "MR_R1_0_5": 53.23,
    "MR_R1_0_7": 34.00,
    "MR_mAP": 30.58,
    "MR_mAP_0_5": 54.80,
    "MR_mAP_0_75": 29.02,
    "MR_short_mAP": 3.11,
    "MR_middle_mAP": 29.42,
    "MR_long_mAP": 41.27,
    "HL_VeryGood_mAP": 35.51,
    "HL_VeryGood_HIT1": 55.87
  }
}
```

All benchmark values are dataset-level. They must not be copied into an analysis response or labeled as per-video accuracy.

## POST /api/v1/analyze

Content type: `multipart/form-data`. Fields:

- `video`: one MP4 file, at or below the current model's `max_video_duration_seconds`.
- `queries`: JSON string array of `{ "id": string, "query": string, "intent": "hook_candidate" | "demo_moment" | "benefit_moment" | "custom" }`.

Example successful response, shortened to one query:

```json
{
  "schema_version": "1.0",
  "analysis_id": "analysis_demo_001",
  "model": { "id": "moment-detr-clip-baseline", "stage": "baseline" },
  "video": { "filename": "demo_video.mp4", "duration_seconds": 19.133, "width": 576, "height": 1024, "codec": "h264" },
  "predictions": [
    {
      "query_id": "demo",
      "intent": "demo_moment",
      "query": "The creator demonstrates how the product works.",
      "windows": [
        { "start": 7.70, "end": 15.46, "confidence": 0.9275 },
        { "start": 3.26, "end": 7.40, "confidence": 0.0213 }
      ],
      "raw_saliency_scores": [-0.8359, -0.4944, 0.2634, -0.3079, 0.5986, 0.6387, 0.5044, 0.3943, -0.6152, -0.6299],
      "saliency_clip_duration_seconds": 2
    }
  ],
  "interpretation": {
    "hook": { "provenance": "HEURISTIC", "status": "not_confidently_detected", "candidate": null },
    "demo_moment": { "provenance": "MODEL_OUTPUT", "start": 7.70, "end": 15.46, "confidence": 0.9275 },
    "result_moment": { "provenance": "MODEL_OUTPUT", "start": 8.00, "end": 17.06, "confidence": 0.9990 },
    "core_action_region": { "provenance": "DERIVED", "start": 8.00, "end": 15.46 }
  }
}
```

`windows` is an array of named objects, never positional arrays. `raw_saliency_scores` is unnormalized and may contain negative values. An analysis may return empty `windows` and nullable interpretation moments. When Hook is absent, use `status: "not_confidently_detected"` and `candidate: null`. A detected Hook uses `status: "detected"` and a non-null candidate. Interpretation provenance values are `MODEL_OUTPUT`, `DERIVED`, and `HEURISTIC` as defined in the [semantics contract](lumae-semantics-v1.md).

## Error response

Any non-success response uses:

```json
{
  "schema_version": "1.0",
  "error": {
    "code": "VIDEO_TOO_LONG",
    "message": "Video duration exceeds the 150 second model limit.",
    "retryable": false
  }
}
```

Valid codes: `VIDEO_TOO_LONG`, `UNSUPPORTED_VIDEO`, `INVALID_QUERY`, `MODEL_UNAVAILABLE`, `ANALYSIS_TIMEOUT`, `MALFORMED_RESPONSE`, `UNKNOWN_ERROR`. The frontend maps network failures to `MODEL_UNAVAILABLE`, timeouts to `ANALYSIS_TIMEOUT`, and invalid JSON/schema to `MALFORMED_RESPONSE`. It never silently switches to mock after a real request fails.

## Versioning and transport

Both endpoints require `schema_version: "1.0"`. Unknown versions fail validation. Frontend presentation code receives only decoded domain objects. Mock transport returns the exact same wire response shape and passes through the same runtime decoder. Model internals such as checkpoint names, feature arrays, and raw Moment-DETR tuple formats are private to future backend adapters.
