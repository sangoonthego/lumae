# D1000 fresh evaluation policy v1

Planning targets:700 Train/150 Validation/150 Test unique videos; Test has at least300 independently annotated query moments and >=2 distinct queries per video. These are **targets**, not an existing D1000 dataset. When groups prevent exact counts, preserve complete groups and report deviations; never split a campaign to meet a percentage.

`data_process/splitting/campaign.py` builds deterministic connected components from confirmed source IDs, normalized URLs, exact hashes and reviewed campaign identities. Brand/product matches and dhash64 visual similarity only create uncertain relations. All members of affected confirmed components are quarantined until review resolves the match as SAME or DISTINCT with reviewer identity/evidence. Different hashes do not prove independent campaigns. Image fingerprints can be computed from existing frames with `image_dhash`; no GPU/video crawling is involved. Blank/similar-frame matches require human review.

Registry fields: vid,source_id,source_url,source_sha256,campaign_id,campaign_status,brand_id,product_id,visual_fingerprint,fingerprint_algorithm,historically_exposed. Default exposure is TRUE when unknown. Only unexposed CONFIRMED/SINGLETON_REVIEWED groups may enter fresh Test. Human/group review statuses must be supplied from real evidence; automatic string labels do not establish reviewer authenticity.

```powershell
python -m data_process.splitting.campaign plan --registry <approved-video-registry.json> --relations <reviewed-relations.json> --output <new-plan.json>
python -m data_process.splitting.campaign validate --splits <reviewed-splits.json> --plan <new-plan.json>
python -m data_process.splitting.campaign freeze --splits <reviewed-splits.json> --plan <new-plan.json> --output <new-version-directory>
```

Validation rejects cross-split campaign/video overlap, unresolved/quarantined groups, source ID/URL/hash collisions, duplicate qids, duplicate Test text within a video, invalid timestamps, false human states and historically exposed Test samples. Exact-content/source aliases must be canonicalized through review before final freeze; grouping them does not magically turn copies into independently unique videos. Parent D500 videos are historically exposed. In particular the historical Test-30 queries may never be described as new blind Test; the validator accepts an additional explicit exposed-qid denylist.

Train/Val inputs are stored under `training/`; fresh Test labels under `sealed_test/`. Frozen manifests use checksums and immutable writes. `freeze_splits` requires all Test annotations HUMAN_VERIFIED with independent human evidence and >=10% of the entire final query set double-annotated. It never selects alpha/checkpoint/architecture. No operator may tune from Test feedback; freeze alpha and model choice from Validation before unsealing Test. Store exposure history and never describe a previously inspected benchmark as freshly blind.

`nested_training_subsets` deterministically selects nested140/440/700 query-sample prefixes of the new Train cohort. All controlled runs share the exact newly frozen Validation/Test, CLIP feature extraction commit/precision protocol, M0 initialization, evaluator revision, seed and documented optimizer/training budget. Counts must be reported as query samples versus unique videos separately. Historical D200/D500 checkpoints on fresh Test are exploratory transfer results; they are not a controlled scaling comparison with retrained D1000 models.

Actual registry decisions, sufficient new accepted videos, supplemental distinct Test queries and real reviews are pending. The implementation's synthetic fixtures are never a benchmark dataset or human-agreement result.
