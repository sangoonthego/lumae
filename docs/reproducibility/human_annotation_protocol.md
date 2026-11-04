# Independent human temporal verification v1

Use `data_process/annotation/independent_review.py` for new versioned review records. Historical annotation CSV/JSON and GT remain untouched. Fields include original AI query/GT, frozen evaluation query, qid/vid/duration, actual annotator IDs, independent human intervals, submitted timestamps, comments, review status, separate adjudicator and adjudicated reference, and a hash-chained append-only revision history. AI-only annotations cannot be HUMAN_VERIFIED.

Operator workflow:

1. Register approved query/video/duration/AI provenance from a new input JSON. Query edits after registration need a new qid/version. Do not show AI boundaries to independent temporal annotators.
2. Give annotators only `packet --qid ...`: qid,vid,query,duration. It excludes AI GT and previous reviewers' boundaries. Actual video access is provided separately. Keep full storage/operator exports inaccessible to blind annotators through workspace/account permissions.
3. Record an independently chosen interval with `annotate --human-id <actual-id> --start ... --end ...`. Do not invent reviewer identities. This command prints only submission status, not other labels. Identity authentication is an external operational responsibility; this offline library validates attestations and provenance, not whether a person is truthful.
4. At least10% of the full final query cohort receives two **different independent** annotators; schedule representative short events and Test cases. `double_review_schedule` proposes a deterministic cohort, not a completed review. A person revising their annotation is not a second independent reviewer. Case variants of the same ID do not count twice.
5. A separate reviewer/adjudicator, distinct from the annotators, inspects the evidence and records an explicit final interval and rationale with `adjudicate`. It never auto-averages two intervals into GT. New annotation revisions invalidate prior HUMAN_VERIFIED status until adjudication again. Exclusions/rejections retain their revision history.
6. `agreement --qid ...` computes pairwise temporal IoU, absolute start/end disagreement and means only from actual independent annotations. Missing double review yields null agreement, not1.0. Report sample/pair counts and review selection, including short-event strata. Do not report synthetic unit-test scores as real agreement.
7. Export reviewed records as metadata.human_review with metadata.review_provenance=HUMAN_VERIFIED and relevant_windows=[adjudicated_reference]. Every fresh Test query must pass this workflow before final benchmark freeze. Historical human-reviewed Train/Val may retain their original protected provenance through an explicit parent-derived allowlist; this never qualifies old material as fresh blind Test.

CLI:

```powershell
python -m data_process.annotation.independent_review --root <new-review-root> --version d1000_human_v1 register --input <new-query.json>
python -m data_process.annotation.independent_review --root <new-review-root> --version d1000_human_v1 packet --qid <qid>
python -m data_process.annotation.independent_review --root <new-review-root> --version d1000_human_v1 annotate --qid <qid> --human-id <annotator-id> --start <seconds> --end <seconds>
python -m data_process.annotation.independent_review --root <new-review-root> --version d1000_human_v1 adjudicate --qid <qid> --human-id <reviewer-id> --start <seconds> --end <seconds> --comments <rationale>
python -m data_process.annotation.independent_review --root <new-review-root> --version d1000_human_v1 agreement --qid <qid>
```

No real human review or agreement scores were created in the P0 engineering phase. Synthetic test identities and intervals are isolated test fixtures.
