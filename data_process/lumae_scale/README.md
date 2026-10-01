# LUMAE Ads D200 builder

Run from the repository root:

```sh
python -m data_process.lumae_scale --target-videos 200 --seed 20260930 --resume
```

The command fingerprints the frozen D48 inputs, walks a seeded AdsQA source
order, resumes downloads and accepted records, creates timestamped contact
sheets and CLIP ViT-B/32 features, and stops at an explicit visual inspection
or interval verification gate. Source access may require a network-permitted
environment. A source video is never counted until a sample passes all gates.

For this D200 bootstrap, the agent views every contact sheet and writes
`local_data/intermediate/lumae_ads_d200/visual_index/<video_id>/visual_events.json`.
The file lists up to three specific visible events, each with a query,
timestamped frame references, contact-sheet references, a coarse phase, and
visible subject/action/object descriptions. Set
`inspection_method=agent_viewed_contact_sheets` and list **all** inspected
sheets. An empty `events` list with `rejection_reason` explicitly rejects a
visually unsuitable video, allowing replacement.

After localization, the command emits a before/inside/after verification image
and pauses. The agent views it and writes `verifier.json`, keyed by the SHA256
of the candidate query. Each verdict records `pass`, notes, and exactly the
displayed `evidence_timestamps`. Rerunning `--resume` uses the verifier and
persists an accepted sample under `accepted/<video_id>.json`. The frozen
semantic_v3 algorithm is checked before localization; boundary refinement is
a separate, versioned CLIP wrapper. CLIP scores are audit signals, not
calibrated probabilities.

The optional local visual provider uses a JSON stdin/stdout command protocol.
Configure it with `--provider-command`, `--model-id`, `--model-revision`,
`--device`, `--precision`, and `--batch-size`. The command receives only
contact-sheet paths and frame timestamps, never AdsQA QA text or titles. It
returns `{"video_id": "...", "events": [...]}`. For D200, an agent must still
inspect every sheet and record that fact in `agent_inspection.json` before the
local provider runs. No paid model API is implicit, and no local VLM has been
smoke-tested on this CPU-only host.

When and only when 152 accepted new samples exist, the builder validates 200
unique qids and videos, preserves the original D48 canonical JSONL lines,
assigns new qids `lumae_ads_d200_0049` through `_0200` by sorted source ID,
and freezes a seeded video split of 140/30/30. All 48 D48 rows are Train-only.
The Train/Val/Test split is LUMAE's split; AdsQA's source split remains
provenance. Val and Test contain AI pseudo labels, not human ground truth.

After a complete freeze, package the Colab inputs with:

```sh
python -m data_process.lumae_scale --package-colab
```

This creates `local_data/exports/lumae_ads_d200_colab.zip` with the frozen
dataset, handoff, runner, and all 200 videos. The D200 Colab runner is
`experiments/M1_lumae_ads/train_d200.py`. Training is outside this build.

## Local CPU compact review

Keep one session process alive so CLIP loads once:

```sh
python -u -m data_process.lumae_scale.cpu_session serve
python -m data_process.lumae_scale.cpu_session client next
```

The session prepares the next 16 sources using six download workers and three
decode slots. View the returned overview before submitting an `events` request;
view the returned verifier strip before submitting a `verify` request. The
`client events --request path.json` and `client verify --request path.json`
commands send agent-authored observations. Neither preparation nor CLIP decides
that an event was visually verified. Overview records include their SHA and the
complete cached-frame timestamp list. Existing accepted bytes are checked on
each request. CPU wall timing includes agent pauses and is saved separately
under `cpu_sessions/<session_id>/` immediately after atomic acceptance.

The 20261001T091033Z session completed the remaining 86 samples. The frozen D200
contains 200 samples, with splits 140/30/30 and D48 entirely in Train. Its master
audit CSV is `local_data/reports/lumae_ads_d200/lumae_ads_d200_annotation_master.csv`.
The final integrity/performance report is `d200_cpu_finish_report.json` in the
same directory. Historical end-to-end timings remain unknown. Persisted stage
cache checks describe the last pipeline pass, because earlier checks are
overwritten; they do not establish an all-session cache hit rate. Overlapped
background preparation times were not persisted independently.

Freeze exports normalize historical nested paths in copies while retaining the
original accepted files and D48 canonical bytes. The dataset manifest anchors
all 152 accepted-file hashes and original ledger/selection hashes. Repeating
freeze checks immutable output bytes. No training or final Test evaluation is
part of this local CPU task.
