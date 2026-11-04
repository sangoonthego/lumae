# Controlled P1 acquisition

Use `python -m data_process.lumae_scale.acquisition --help` for the actual CLI.
P1 adds a bounded collector around the P0 `candidate_plan` and `BuilderSession`.
Historical layouts, raw sources, annotations, datasets, cached training features
and P0/audit outputs remain read-only. New binary assets stay on the configured
external data root; no ZIP, upload, migration or training is performed.

## Operational commands

Run from the repository root. The reviewed pilot config is
`research_outputs/p1_20261103_01/pilot_config.json`.

```powershell
# No HTTP or external-root initialization in plan mode.
python -m data_process.lumae_scale.acquisition plan --config research_outputs/p1_20261103_01/pilot_config.json --output <new-plan.json>

# Requires network_execution_authorized=true, preflight gate and operator scope.
python -m data_process.lumae_scale.acquisition run --config research_outputs/p1_20261103_01/pilot_config.json --enable-live-network --max-wave 3

# Offline processing/triage/verification, no downloads.
python scripts/p1_processing.py index --config research_outputs/p1_20261103_01/pilot_config.json
python -m data_process.lumae_scale.acquisition triage --config research_outputs/p1_20261103_01/pilot_config.json
python -m data_process.lumae_scale.acquisition validate --config research_outputs/p1_20261103_01/pilot_config.json --output <new-validation.json>
python -m data_process.lumae_scale.acquisition export --config research_outputs/p1_20261103_01/pilot_config.json
```

The real pilot has exhausted its 30-candidate budget. Repeating that run does
not acquire another 30. It checks the pinned selection/config and completed
source hashes, preserves terminal rejections, and rebuilds derived exports.
The 1-hour budget clock persists across restarts. Increasing limits requires a
new reviewed configuration/run, not silently editing a completed run.

Windows OS file locks prevent simultaneous coordinators for the same run. A
crashed process releases its OS lock without manual stale-lock deletion.
Attempt JSON and source records use fsync/atomic replacement. Immutable event
segments avoid a half-written JSONL tail; the JSONL log is a derived export.
Missing derived exports can be rebuilt from authoritative records/attempts.
Malformed authoritative records fail closed. Incomplete atomic temporary files
are not treated as accepted records. Partial downloads use `.part` and an
ETag/Last-Modified identity sidecar. Range/If-Range, byte range, HTTP validators,
Content-Length and final video checks are validated before promotion.

Downloads are serial, rate limited, bounded by total/per-file bytes, elapsed
wall clock, request/candidate/retry counts, staging usage and free disk reserve.
Retries use bounded exponential delay and honor bounded Retry-After on 429.
401/403/404/410 are terminal. Redirects must stay on the HTTPS allowlist;
credentials and arbitrary request exceptions are not serialized to logs.
No default CLI mode makes a network request.

## Acceptance and review

Technical acquisition, source acceptance, temporal annotation acceptance and
final dataset readiness are separate states. Exact source IDs/URLs and parent
content hashes are excluded before/after acquisition. Three-frame dHash checks
against a content-bound 500-video parent baseline flag uncertain matches.
Sampling may miss edited duplicates; static frames/end cards may cause false
positives. Neither a different SHA nor a missing dHash match proves campaign
independence. New technically valid files enter quarantine until actual campaign
review resolves that risk. No reviewer identities or decisions are generated.

```powershell
python -m data_process.lumae_scale.acquisition review-source --config <config.json> --source-id <32hex> --evidence <actual-campaign-review.json>
```

Campaign evidence needs reviewer_id, reviewed_at, evidence_paths relative to
the pilot work directory, campaign_id, decision (DISTINCT/DUPLICATE/UNSURE), and
comparison_scope=D500_AND_PILOT. Referenced evidence must exist and is hashed.
DISTINCT additionally requires the full parent fingerprint baseline and a
fresh source/video check. This is an offline operator attestation, not identity
authentication; access permissions and actual review remain operational duties.

Frame/contact-sheet generation reuses `build_visual_index` with explicit
root/cache paths. Existing callers retain historical defaults. For a visually
inspected video, put an actual `AgentEvidenceProvider` event packet in
`<pilot-work>/visual_index/<source-id>/visual_events.json`. Required evidence
paths match the index, including the `contact_sheets/` directory prefix.

```powershell
python -m data_process.lumae_scale.pilot_annotation --config <config.json> --source-id <32hex>
python -m data_process.lumae_scale.acquisition accept-annotation --config <config.json> --evidence <complete-verified-p0-record.json>
```

Proposal uses local CPU CLIP weights and the unchanged frozen Semantic V3
ranker/boundary wrapper. It does not invent an unattended visual model. Raw
prediction, query/evidence hashes, model hash and algorithm version are saved.
Proposals remain AI pseudo labels pending visual interval verification.
Acceptance invokes the full P0 validator, including before/inside/after imagery,
semantic prediction identity and truthful AI provenance. The independent
`ReviewStore` registers original AI bounds and emits a blind review packet.
Human verification needs actual independent labels/adjudication through that
existing workflow. AI agents never become human reviewers.

## Scale gate

The prepared `scale_config.disabled.json` is planning-only. It binds the pilot
manifest by SHA, excludes all previously attempted source IDs, and references
existing pilot assets rather than redownloading/copying them. Prior accepted
sources count toward the 500 additional-source target; quarantined sources do
not. Any changed prior manifest requires a new reviewed binding.

Live scale mode requires a real passing `pilot_pass_evidence`, an operational
gate, `network_execution_authorized=true`, and both `--enable-live-network` and
`--enable-scale-budget`. The supplied pilot does not pass; scale stays disabled.
The future final builder still requires 1,000 accepted source/annotation pairs,
campaign-disjoint groups and the independent Test/review gates. This pilot does
not freeze a Test set or satisfy the planned 700/150/150 split, 300 verified
Test queries or >=10% independent double annotations.
