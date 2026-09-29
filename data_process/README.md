# Lumae Offline Data Processing Subsystem (`data_process/`)

## 1. Purpose
The `data_process/` subsystem provides CPU-only, offline dataset preparation and validation for Lumae research. It establishes a rigorous boundary between offline research preparation and the online product serving layer (`src/`, `backend/colab/`), preventing training logic from polluting the Next.js frontend or FastAPI inference engine.

```
RAW ANNOTATIONS
      │
      ▼
SOURCE ADAPTER (QVHighlights, Ego4D NLQ)
      │
      ▼
CANONICAL SAMPLE (CanonicalTemporalSample)
      │
      ▼
VALIDATION (Boundary, Non-empty text, Duration)
      │
      ▼
SPLIT CHECK (Video-level disjoint split guarantee)
      │
      ▼
CANONICAL JSONL
      │
      ▼
FEATURE / GPU MANIFESTS (FeatureManifest, GpuJobManifest)
      │
      ▼
GOOGLE DRIVE (Persistent Artifact Storage)
      │
      ▼
COLAB T4 (Feature Extraction & Fine-Tuning)
```

---

## 2. What Runs Locally (CPU Only)
- Ingesting and parsing raw annotation files (JSON, JSONL).
- Schema mapping and normalization into canonical domain dataclasses.
- Strict boundary validation (temporal windows within video durations).
- Rejection logging for malformed records with explicit reason codes (`ReasonCode`).
- Video-level leakage prevention ($V_{train} \cap V_{val} = \emptyset$).
- Deterministic, seeded video splitting.
- Dataset distribution profiling and CLI reporting (`audit_dataset`).
- Lightweight planning manifests for video/text feature extraction and Colab GPU jobs.

---

## 3. What Does NOT Run Locally
- **No GPU inference or training**: No Moment-DETR optimizer or model checkpoint loading.
- **No CLIP feature extraction**: Frame-level video feature extraction and query text encoding happen on Colab T4.
- **No large dataset downloads**: Full raw Ego4D or QVHighlights video archives are stored externally in Google Drive.
- **No Google Drive API/SDK mounting**: Local pipelines operate strictly on standard filesystem paths.

---

## 4. Canonical Sample Schema
The research domain contract is defined in `data_process/schemas/temporal_sample.py` and is completely independent from frontend serving types:

```python
@dataclass
class CanonicalTemporalSample:
    qid: int | str                      # Unique query ID within dataset
    vid: str                            # Namespaced video identifier (e.g. ego4d__<uid>)
    query: str                          # Natural language query string
    duration: float                     # Total video duration in seconds (> 0)
    relevant_windows: list[list[float]] # [[start, end], ...] where 0 <= start < end <= duration
    relevant_clip_ids: list[int] | None = None
    saliency_scores: list[float] | None = None
    source: str | None = None           # e.g., "ego4d_nlq" or "qvhighlights"
    intent: str | None = None           # optional downstream intent tag
    original_id: str | int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
```

- **Required fields**: `qid`, `vid`, `query`, `duration`, `relevant_windows`.
- **Optional fields**: `relevant_clip_ids`, `saliency_scores`, `source`, `intent`, `original_id`, `metadata`.
- **Saliency policy**: Saliency is never fabricated for datasets that lack it (e.g. Ego4D NLQ).

---

## 5. Adapter Model
Adapters subclass `BaseDatasetAdapter` in `data_process/adapters/base.py`:
- `QVHighlightsAdapter`: Converts existing QVHighlights GT annotations, preserving ground-truth windows and saliency.
- `Ego4DNLQAdapter`: Normalizes Ego4D NLQ clip queries into the canonical format without inventing saliency.
- **Rejection Policy**: Bad data is never silently dropped. A `RejectedSampleRecord` is generated with reason codes:
  - `MISSING_QUERY`
  - `INVALID_DURATION`
  - `INVALID_WINDOW`
  - `WINDOW_OUT_OF_RANGE`
  - `MISSING_VIDEO_ID`
  - `DUPLICATE_QID`
  - `UNSUPPORTED_SOURCE_RECORD`
  - `MALFORMED_RECORD`

---

## 6. Validation Pipeline
Validation utilities live in `data_process/validation/`:
- `validate_duration(duration)`: Confirms finite, strictly positive duration.
- `validate_window(start, end, duration, tolerance)`: Enforces $0 \le start < end \le duration + tolerance$.
- `validate_canonical_dict(data)`: Validates full JSON/dict structures against required types and non-empty string constraints.

---

## 7. Leakage Rules
Queries from the same physical video must never leak across training and evaluation splits:
- `find_video_leakage(train, val, test)`: Detects any intersection of underlying `vid`s across split boundaries.
- `assert_no_video_leakage(...)`: Raises `VideoLeakageError` with conflicting video IDs.
- `deterministic_video_split(samples, ...)`: Seeds and partitions strictly by `vid`, never by sample index or `qid`.

---

## 8. Feature Manifest
`data_process/manifests/features.py` plans feature extraction offline without computing tensors:
```json
{
  "vid": "ego4d__clip123",
  "video_feature": {
    "ready": false,
    "path": "Features/Ego4D/video/ego4d__clip123.npz",
    "clip_duration_seconds": 2.0
  },
  "queries": [
    {
      "qid": "ego4d_nlq_42",
      "query_text": "Where is the hammer?",
      "ready": false,
      "path": "Features/Ego4D/text/qid_ego4d_nlq_42.npz"
    }
  ]
}
```

---

## 9. GPU-Job Manifest
`data_process/manifests/gpu_jobs.py` models future Colab GPU jobs as an executable directed acyclic graph (DAG):
- Job types: `VIDEO_FEATURE_EXTRACTION`, `TEXT_FEATURE_EXTRACTION`, `TRAIN_STAGE_A`, `EVAL_STAGE_A_EGO4D`, `EVAL_STAGE_A_QVHIGHLIGHTS`.
- Validates job dependencies and checks for cycles before Colab execution.

---

## 10. Local Storage
- Local data files live under `local_data/` (or paths defined in `data_process/config/paths.example.json`):
  - `local_data/raw/`: Raw annotation files.
  - `local_data/canonical/`: Clean, validated `.jsonl` files.
  - `local_data/reports/`: Audit summaries and rejection logs.
- `local_data/` is ignored by Git to prevent repository bloat.

---

## 11. Google Drive Handoff
Google Drive is an external artifact repository:
- Logical targets are documented in `data_process/config/drive_layout.json`.
- Validated canonical JSONLs and manifests generated locally are synced to Drive via user-controlled tooling.
- No Google Drive credentials or MCP connectors are embedded in the repository.

---

## 12. Colab Handoff
In Google Colab:
1. Mount Google Drive: `from google.colab import drive; drive.mount('/content/drive')`.
2. Copy canonical annotations and feature manifests from Drive to `/content/` SSD.
3. Run GPU feature extraction or Moment-DETR fine-tuning.
4. Copy updated checkpoints (`Checkpoints/M1/model_best.ckpt`) back to Drive.

---

## 13. Testing
All tests run locally on CPU without network or GPU access:
```bash
# Verify Python bytecode syntax
python -m compileall data_process

# Run data_process unit test suite
python -m pytest -q data_process/tests

# Run dataset audit CLI on synthetic fixture
python -m data_process.cli.audit_dataset --input data_process/tests/fixtures/synthetic_samples.jsonl
```

---

## 14. Stage A Roadmap
- **Stage A.0 (Completed)**: Scaffolding, canonical schema, validation, leakage checks, manifests, CLI auditor.
- **Stage A.1 (Next)**: Ingest real Ego4D NLQ annotations into `local_data/canonical/ego4d_nlq_train.jsonl` and `ego4d_nlq_val.jsonl`.
- **Stage A.2**: Generate feature manifest and prepare Colab T4 feature extraction notebook.
- **Stage A.3**: Run Moment-DETR fine-tuning on Colab T4 and record Stage A benchmark metrics.
