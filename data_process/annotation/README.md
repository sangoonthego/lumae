# LUMAE Product Ads — Human Temporal Review Tool (Stage A.2.5)

## Stage A.2.6 assisted deployment

Stage A.2.6Q produced video-inspected AI query proposals for the current DRAFT samples. Review them in `local_data/reports/lumae_ads/batch_query_sanitation_review.csv` or in `QUERY_SANITATION` mode. The sampled-frame sheets are in `local_data/reports/lumae_ads/a26q_contact_sheets/`; reviewers should still watch the video before approving. The proposals remain `AI_SUGGESTED` until a human saves an Accept, Edit, Keep, or Exclude decision. The batch command reads the audited proposal input and writes only advisory artifacts:

```bash
python -m data_process.annotation.batch_query_sanitation
```

Prepare the current pilot worklist and readiness report from the source CSVs:

```bash
python -m data_process.annotation.assisted_deployment prepare
streamlit run data_process/annotation/app.py
```

Select `QUERY_SANITATION` for the single-screen fast lane. The panel opens at the first unresolved sample and shows the video, original query and quality, AI proposal, change type, visual evidence, duration, and progress. Enter your own `human_` reviewer ID, watch the video, confirm visibility and localizability, and click **Accept Suggestion · Lock & Predict**, **Edit Query · Lock & Predict**, or **Keep Original · Lock & Predict**. The click writes canonical `query_reviews.csv`, creates an immutable versioned query lock under `local_data/manifests/a26_fastlane_query_locks/`, and runs the unchanged frozen semantic_v3 ranker on video-derived frame-change proposals for that sample only. These proposals are advisory and their scores are uncalibrated. Review the predicted interval in the same screen; click **Accept AI Window**, **Edit Window**, **Reject AI Window · Save Manual Interval**, or **Exclude Sample from Temporal Dataset** to save human GT or an exclusion. Exclusion requires a reason and confirmation. Query revisions create new locks and predictions while preserving older versions. Every human action is saved immediately, and an interrupted temporal commit is recovered from its durable human audit. The app then advances to the next unresolved sample. After all 48 pilot samples are `REVIEWED` or `EXCLUDED`, the integrity gate automatically exports and freezes `local_data/datasets/lumae_ads_v1/` and creates `local_data/manifests/lumae_ads_v1_training_handoff.json`. Until that gate passes, no training dataset is frozen.

After human query approval, semantic_v3 needs a video-inspected candidate pool in `local_data/annotations/lumae_ads/semantic_v3_deployment_candidate_pools.json`. Each sample key maps to a list of frozen-ranker `SemanticCandidateEvent` objects, for example:

```json
{
  "sample_id": [
    {
      "candidate_id": "A",
      "start_coarse": 1.0,
      "end_coarse": 3.0,
      "visible_subject": "person",
      "visible_action": "turns a dial",
      "visible_object": "washing machine dial",
      "supporting_visual_evidence": "A hand rotates the dial in inspected video frames"
    }
  ]
}
```

Candidate windows must be within the video duration and grounded in inspection of the actual video. The frozen ranker has a generic fallback for unindexed samples; the deployment command refuses that fallback. Run:

```bash
python -m data_process.annotation.assisted_deployment precompute
```

Then select `ASSISTED_TEMPORAL_REVIEW` in the same Streamlit app. A human can `ACCEPT_AI`, `EDIT_AI`, `REJECT_AI`, or `EXCLUDE`. Only that decision writes `human_primary.csv` and `assisted_annotation_audit.csv`. The preannotation CSV remains advisory until review. Re-run `prepare` after reviews to refresh `pilot_dataset_readiness.json`. The official split is reserved for Stage A.2.7.

The A.2.5K selection manifest's exact frozen bytes were restored in Stage A.2.6.1. See `local_data/reports/lumae_ads/semantic_v3_clean_query_blind5_benchmark_provenance_addendum.md` for the recovery evidence and remaining timestamp caveat before dataset freeze.

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
