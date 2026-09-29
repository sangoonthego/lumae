# LUMAE product and research semantics — v1

Status: **frozen for API schema 1.0**. A model adapter may change; these meanings and provenance boundaries may not change within v1.

## Three scopes

| Category | Scope | Examples | Source in API v1 |
| --- | --- | --- | --- |
| Model Benchmark | Model and held-out dataset | R1@0.5, MR mAP, HL mAP, HIT@1 | `GET /api/v1/model` |
| Current Analysis | One video and one or more queries | windows, model confidence, raw saliency | `POST /api/v1/analyze` |
| Lumae Interpretation | Product interpretation of current analysis | Hook Candidate, Demo Moment, Result Moment, Core Action Region | `analysis.interpretation` |

Benchmark metrics are never current-video accuracy, current-query accuracy, prediction confidence, or a video score. They must not change when a user changes query selection.

## Canonical terms

1. **Model Benchmark:** Results of a specific model version evaluated on a held-out dataset. The current reference is QVHighlights / Moment-DETR CLIP-only. It belongs to model metadata, not an individual analysis.
2. **Current Analysis:** The versioned result of processing one uploaded video with its submitted query definitions. It identifies the model used and contains current predictions plus Lumae interpretation.
3. **Query Prediction:** A query-conditioned collection of ranked temporal windows and raw saliency scores. Different queries may return different windows and saliency.
4. **Temporal Window:** Named `start` and `end` times in seconds, with `start < end`. Prediction windows also carry model confidence. API v1 never exposes positional tuples.
5. **Model Confidence:** A ranking or matching confidence attached to a predicted temporal window. It is neither accuracy nor a probability of correctness.
6. **Raw Saliency:** Query-conditioned temporal saliency produced by the model. Scores may be negative and are preserved exactly in `rawSaliencyScores`; they are not probabilities.
7. **Normalized Saliency:** A derived 0–1 display value, `(x - min) / (max - min)`. If all raw values match, display zeros. Normalized values never replace, overwrite, or enter the API in place of raw scores.
8. **Hook Candidate:** A nullable downstream Lumae **HEURISTIC**, not a native Moment-DETR class. `not_confidently_detected` requires `candidate: null`; a valid analysis may return no Hook Candidate.
9. **Demo Moment:** A temporal candidate associated with a query describing a product demonstration. Its temporal prediction has **MODEL_OUTPUT** provenance.
10. **Result / Benefit Moment:** A temporal candidate associated with the observable result or benefit of product usage. Its temporal prediction has **MODEL_OUTPUT** provenance.
11. **Core Action Region:** A temporal region derived from overlap or another documented relationship among relevant predictions. It has **DERIVED** provenance and may be null.
12. **MODEL_OUTPUT:** A value directly returned by the temporal model for the current video and query, including its window and confidence.
13. **DERIVED:** A deterministic downstream calculation from model outputs, such as the Core Action Region. It must not be presented as a native model class.
14. **HEURISTIC:** A downstream rule-based interpretation, currently Hook Candidate. It must not be presented as model ground truth.

## Baseline context

The current CLIP-only baseline samples at approximately 2 seconds, accepts videos up to 150 seconds, and is primarily visual. Audio and speech understanding are not guaranteed. Short temporal moments are a known benchmark weakness. These constraints describe the current model version and can change through model metadata while API v1 shape remains stable.
