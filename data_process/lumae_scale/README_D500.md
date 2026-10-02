# LUMAE AdsQA D500 Phase A

## Reproduce the annotation build

The D200 canonical dataset is the immutable parent. The D500 builder selects 300 additional unique AdsQA source videos from the same pinned source manifest, retains every D200 canonical row, and appends each new row to Train. Val-30 and Test-30 retain their D200 JSONL bytes. D48 stays exclusively in Train. The annotation query, Semantic V3 ranker, CLIP boundary refinement, routing, and mandatory agent visual verdict follow the frozen D200 implementation.

From the repository root:

```powershell
python -m data_process.lumae_scale --shadow-d500
python -m data_process.lumae_scale.cpu_session serve --dataset-version lumae_ads_d500 --port 8767
```

If the annotation process cannot reach the pinned AdsQA video host directly, run the optional loopback source fetcher in a network-enabled terminal:

```powershell
python -m data_process.lumae_scale.network_fetcher serve --port 8770
```

Set `LUMAE_D500_DOWNLOAD_BRIDGE=http://127.0.0.1:8770` in the annotation server's environment before starting it. The fetcher accepts only HTTPS URLs at `video.adsoftheworld.com` with 32-character source IDs, writes to the normal AdsQA source directory, and listens only on loopback. Existing source files remain content checked by the annotation pipeline.

The persistent review server accepts `POST /next`, `POST /events`, `POST /verify`, and `POST /stats`. The `/next` response supplies an overview path, and `/verify` uses the generated before/inside/after strip. A human or visual agent must inspect actual frames and provide concrete event observations and an explicit verifier decision. The server never fabricates observations or treats the automatic verifier's abstention as a positive verdict.

`python -m data_process.lumae_scale.review_client next --port 8767` displays the active source. `review_client events` and `review_client verify` accept URL-encoded JSON payloads. Repeating `/next` after a process restart resumes the persisted active stage; it does not accept or relabel a video. Inspect both the overview and verification strip before sending the corresponding review payloads.

The annotation state is under `local_data/intermediate/lumae_ads_d500`. The source selection has a fixed seed of `20260930`; source content hashes are checked against D200 and previously accepted new sources. Retries use explicit failure classes and backoff. Preparation uses at most 16 queued sources, six downloads, and three frame decodes. Per-attempt telemetry is append-only in `work/telemetry`, separate from canonical accepted records.

To package a complete freeze:

```powershell
python -m data_process.lumae_scale --target-videos 500 --resume
python -m data_process.lumae_scale --target-videos 500 --package-colab
```

The freeze requires exactly 300 new accepted source videos and writes an immutable D500 canonical dataset, split manifest, annotation audit, 500-row master CSV, checksums, and Phase A training handoff. The package includes frozen dataset and source video files. A later Phase B Colab run must bind and verify the external D200 CLIP training feature cache, M0 checkpoint, and frozen training recipe; these are not prerequisites for Phase A annotation. It must encode only the 300 new video/text features, reuse the 200 old pairs when valid, and record a feature manifest before training. Test remains sealed until an explicitly requested final evaluation.

The audit at `local_data/reports/lumae_ads_d500/architecture_audit.json` records the physical workspace inventory and distinctions between local annotation caches and the external training NPZ cache. `shadow_replay.json` checks 25 compatible frozen D200 samples before new annotation; any historical verifier evidence discrepancy is disclosed there.
