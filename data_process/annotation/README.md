# LUMAE Product Ads — Human Temporal Review Tool (Stage A.2.5)

This local application provides a strict, blind temporal annotation interface for human reviewers validating video advertisements for the LUMAE benchmark.

---

## Quickstart

From repository root, run:

```bash
streamlit run data_process/annotation/app.py
```

The application runs **100% locally** (no cloud services, no external APIs, no GPU required).

---

## Directory & File Structure

```
local_data/annotations/lumae_ads/
├── human_primary.csv                # Primary human annotations worklist (active)
├── human_secondary.csv              # Independent secondary annotations (>=20%)
├── human_adjudicated.csv            # Resolved consensus annotations for tIoU < 0.70
├── secondary_worklist.csv           # Deterministic secondary sample worklist with BLANK GT
├── human_review_log.jsonl           # Append-only audit trail with query edit provenance
├── human_multi_windows.jsonl        # Discontinuous multi-window intervals
├── backups/                         # Automatic timestamped backups created on save
└── draft_snapshot/                  # Preserved unverified agent draft snapshots
```

---

## Workflow Modes

### 1. Primary Review (`human_01`)
- Review each video in `local_data/raw/adsqa/videos/`.
- Review and verify the natural language query:
  - `VALID`: Query clearly describes a visible, localizable action.
  - `NEEDS_EDIT`: Query describes the wrong product or action. Edit to a descriptive, objective action sentence.
  - `INVALID_VIDEO`: No clear target action can be grounded. Click **Mark as Excluded**.
- Mark **START** and **END** boundaries with 0.1-second precision ($0 \le start < end \le duration$).
- Click **SAVE & NEXT**.

### 2. Secondary Review (`human_02`)
- **Blind Guarantee:** Secondary annotators see ONLY the verified final query and the video.
- Primary GT timestamps, agent draft suggestions, and boundary notes are strictly hidden.
- Annotator marks independent boundaries and clicks **SAVE & NEXT**.

### 3. Adjudication Mode (`human_adjudicator`)
- Activates when pairwise agreement $\text{tIoU} < 0.70$.
- Displays primary and secondary boundaries side-by-side.
- Adjudicator decides consensus GT without blind timestamp averaging.

---

## Freezing Ground Truth

Once all 48 pilot videos are reviewed or excluded, and secondary annotations meet the $\ge 20\%$ requirement:

```bash
python -m data_process.cli.freeze_lumae_ads_pilot
```

This validates all gates and outputs:
- `local_data/annotations/lumae_ads/frozen_gt.csv`
- `local_data/canonical/lumae_ads/pilot_v1_human_verified.jsonl`
- `local_data/manifests/lumae_ads_pilot_v1.json`
