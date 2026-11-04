# D1000 incremental builder

`builder.py` is the configurable coordinator. The existing D200/D500 `BuildLayout`, `IncrementalSession`, CLI and frozen outputs retain their historical behavior. Their hardcoded200/300/500 counts are not silently relaxed. New configurable construction uses `BuildConfig`, `candidate_plan` and `BuilderSession`.

The only shared legacy change is optional `root=` on `freeze.validate_annotation`; callers without it retain repository-relative behavior. New annotations still require the same source SHA, real frame/contact-sheet and verifier evidence, sanitation, frozen Semantic V3 SHA and AI provenance. Human adjudication is a separate explicit review workflow; AI-only accepted records cannot become human ground truth implicitly.

## Safe discovery

```powershell
python -m data_process.lumae_scale.builder --config data_process/config/d1000_v1.json --repo-root . --source-manifest local_data/raw/adsqa/source_manifest.json --dry-run --output <new-plan.json>
```

The command performs no directory creation under E:, download, frame extraction, migration or acceptance. The target is1000 **unique accepted source videos**, not1000 URLs or filesystem names. Parent500 are read from protected D500 manifests/checksums. Seeded source ordering excludes parent IDs/URLs/hashes, duplicate URLs/IDs in the pool, and inherited terminal rejections. Raw116 outside D500 are inspected separately and never auto-accepted.

## Storage/path contract

`data_process/config/d1000_v1.json` proposes `data_root=E:/LUMAE`; model logic contains no E: constant. The parent directory stays repository-relative and read-only. New work is `<data_root>/work/<version>`, datasets `<data_root>/datasets/<version>`, sources `<data_root>/sources/adsqa/videos`, annotation cache `<data_root>/cache/<version>/visual_index`. E:\LUMAE already exists and has not been recreated or populated during this workstream. No legacy asset is copied/moved.

New accepted record paths must be relative to the configured data root, never `..` or arbitrary absolute paths. Parent canonical IDs,query IDs and source hashes remain unchanged. Old parent path references retain original provenance; consumers must use the parent/source-root mapping instead of rewriting old manifests. The existing D500 annotation review server stays historical; the generic coordinator accepts the established producer's verified record contract at the configured root. This phase does not launch a new annotation service or promise an unattended visual-review operator.

## Future collection API (not executed on live sources)

1. Instantiate `BuilderSession(config,pinned_source_pool)`. It pins parent checksums, config identity and selection order in new versioned work paths; repeated initialization resumes the append-only ledger.
2. `prepare_existing(source_id,path)` probes and hashes an already available video inside the new root. It does not adopt/migrate a legacy raw file implicitly.
3. A separately authorized collector may call `fetch(source_id,allow_network=True)`. Default is denied. It reuses `source_pool.download` (HTTP Range/.part and fsync/atomic finalize), classifies failures and records retries; no CLI here starts live crawling.
4. The annotation operator produces the unchanged query/temporal/visual evidence contract. `accept(record)` runs the full frozen validator, rejects duplicate IDs/content, invalid duration/timestamps and missing evidence. Accepted records are immutable. `reject` is terminal and preserves the reason. Downloaded/prefetched files never count as accepted.
5. Generate reviewed campaign assignments with `splitting.campaign`; pass both `assignments` and `split_plan` to `session.freeze`. Parent exposure cannot be erased. Quarantine prevents freezing a complete cohort prematurely. `historical_extension` is an explicit alternative retaining the parent evaluation assignment, not a fresh blind-test protocol.
6. Freeze rechecks parent, source/accepted-record hashes and all visual/semantic evidence; writes deterministic new all/train/val/test JSONL, source/split/campaign/feature manifests, checksums and COMPLETE marker. Old files are never rewritten. Training asset manifest includes only Train/Val qids/features. Existing parent feature entries are marked `BIND_EXISTING_CACHE_NO_REENCODE`; new entries are `EXTRACT_NEW_ONCE`, all readiness remains false until independently bound/verified.

Synthetic tests cover1000 target/determinism, HTTP interruption and Range resume, duplicate IDs/URLs/hashes, missing/corrupt videos, invalid times, acceptance idempotency, rejection, parent tamper, campaign conflict and Test/feature export boundaries. A synthetic provider/probe is clearly separated from production's ffprobe/full annotation validator. Dataset freeze is not evaluation readiness: the fresh Test still requires >=300 genuine human-verified queries and double-review checks.
