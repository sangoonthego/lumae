"""Read-only reconstruction of P2.1 inputs; writes only its new audit namespace."""
import csv
import hashlib
import json
import sys
from collections import Counter,defaultdict
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from data_process.lumae_scale.models import ROOT,read_json,atomic_json,digest
from data_process.lumae_scale.incremental import immutable_write,json_bytes

VERSION='p21_20261103_01'
O=ROOT/'research_outputs'/VERSION
E=Path('E:/LUMAE/work/d1000')/VERSION
R=Path('E:/LUMAE')

def csvout(path,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)

def main():
    state=read_json(O/'audit/audit_before_change.json');previous=Path(state['latest_p2_directory'])
    rows=read_json(O/'audit/recovery_pool_raw.json')
    pilot={r['source_id']:r for r in csv.DictReader((ROOT/'research_outputs/p14_p15_20261103_02/pilot30_inventory.csv').open(encoding='utf-8-sig'))}
    p2={read_json(f)['source_video_id']:read_json(f) for f in (R/'work/d1000/pilot_p2_20261103_01/records').glob('*.json')}
    prior_reviews={r['source_id']:r for r in [json.loads(l) for l in (previous/'pilot30_temporal_reviews.jsonl').read_text(encoding='utf-8').splitlines()]}
    p2_revisions=[json.loads(l) for l in (previous/'p2_query_revision_inventory.jsonl').read_text(encoding='utf-8').splitlines()]
    p2revs=defaultdict(list)
    for q in p2_revisions:p2revs[q['source_id']].append(q)
    entries=[];tax=[]
    for j,r in enumerate(rows,1):
        sid=r['source_id'];row=dict(r,recovery_id=f'recovery_{j:02d}')
        if r['origin']=='pilot':
            old=pilot[sid];review=prior_reviews[sid]
            row.update(duration=float(old['duration']),source_url=old['source_url'],candidate_id=old['candidate_id'],
                overview_path=str(R/'work/d1000/p14_p15_20261103_01'/f"pilot30_{int(old['candidate_id']):02d}_overview.jpg") if str(old['candidate_id']).isdigit() else '',
                prior_query=review['query'],prior_window=review['raw_semantic_proposal'],prior_observed=review['observed_event_interval'],
                prior_reason=review['reason'],prior_proposal=review['raw_artifact_path'],prior_review=str(previous/'pilot30_temporal_reviews.jsonl'),
                duplicate_candidates=[],campaign_status='UNKNOWN_SPLIT_RESTRICTED')
            # Candidate IDs may be prefixed; inventory row ordering is authoritative.
            order=list(pilot).index(sid)+1
            row['overview_path']=str(R/'work/d1000/p14_p15_20261103_01'/f'pilot30_{order:02d}_overview.jpg')
        else:
            old=p2[sid];review=read_json(R/'work/d1000/pilot_p2_20261103_01/annotation_decisions'/(sid+'.json'))
            revision=next((q for q in p2revs[sid] if q['selection_status']!='SUPERSEDED_NOT_ACCEPTED'),None)
            row.update(duration=float(old['technical']['probe']['duration']),candidate_id=sid,overview_path=str(R/'work/d1000/pilot_p2_20261103_01/evidence'/sid/'overview.jpg'),
                prior_query=revision['query'] if revision else None,prior_window=[revision['start'],revision['end']] if revision else None,
                prior_observed=review.get('observed_event_interval'),prior_reason=review['reason'],prior_proposal=revision['path'] if revision else None,
                prior_review=str(R/'work/d1000/pilot_p2_20261103_01/annotation_decisions'/(sid+'.json')),
                duplicate_candidates=old.get('duplicate_candidates',[]),campaign_status=old.get('campaign_status','UNKNOWN'))
        assert Path(row['overview_path']).is_file()
        row['overview_sha256']=digest(Path(row['overview_path']))
        row['technical_pass']=True
        # This is ledger-based diagnosis, explicitly not new visual ground truth.
        text=(row.get('prior_reason') or '').lower();causes=[]
        if row['duplicate_candidates']:causes.append('F7')
        if any(x in text for x in ('unsupported','incorrect','identity','not established','does not establish','static','scene description','not sufficiently','not distinguish','does not distinguish','only as','generic')):causes.append('F1')
        if any(x in text for x in ('miss','truncat','final reveal','only captures')):causes.append('F3')
        if row.get('prior_window') and any(x in text for x in ('raw','boundary','starts','includes','spans','interval','ends')):causes.append('F4')
        observed=row.get('prior_observed')
        if observed and observed[1]-observed[0]<=2:causes.append('F5')
        if any(x in text for x in ('repeated','repeat','several armored','multiple competing','different performer','three identities')):causes.append('F6')
        if any(x in text for x in ('no distinct observable','no clearly supported','static child','no unique bounded')):causes.append('F8')
        if not causes:causes=['F2']
        row['initial_failure_causes']=sorted(set(causes));entries.append(row)
        tax.append({'source_id':sid,'causes':'|'.join(row['initial_failure_causes']),'evidence_level':'PRIOR_LEDGER_DIAGNOSIS_NOT_FRESH_VISUAL_GT','reason':row.get('prior_reason')})
    # Conservative leakage grouping includes uncertain relations through shared
    # matched parents. It does not turn those relations into confirmed campaigns.
    parent={r['source_id']:r['source_id'] for r in entries};references=defaultdict(list)
    def find(x):
        parent.setdefault(x,x)
        while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
        return x
    def union(a,b):
        a,b=find(a),find(b);parent[max(a,b)]=min(a,b)
    for r in entries:
        for match in r['duplicate_candidates']:
            references[match['source_video_id']].append(r['source_id'])
    for linked in references.values():
        for sid in linked[1:]:union(linked[0],sid)
    groups=defaultdict(list)
    for r in entries:groups[find(r['source_id'])].append(r['source_id'])
    ordered=sorted(groups.values(),key=lambda g:hashlib.sha256(('20261103|'+'|'.join(sorted(g))).encode()).hexdigest())
    dev=[];reserved=[]
    for g in ordered:
        (dev if len(dev)+len(g)<=20 else reserved).extend(g)
    assert set(dev).isdisjoint(reserved) and len(dev)+len(reserved)==61
    split={'seed':20261103,'development_source_ids':sorted(dev),'reserved_offline_source_ids':sorted(reserved),
        'uncertain_relation_groups':[sorted(g) for g in groups.values()],
        'offline_holdout_limitation':'All61 sources were exposed in earlier tasks. Reserved subset is a prospective optimization holdout only, NOT genuinely unseen research data. Fresh25-source wave after config freeze is the source-held-out check; campaign independence remains unverified.',
        'no_d500_test_labels_used':True,'source_split_metadata_not_used_for_scoring':True,
        'frozen_before_optimization':True,'fresh_evaluation_max_candidates':25,'fresh_pipeline_changes_after_freeze_prohibited':True}
    immutable_write(O/'manifests/evaluation_protocol.json',json_bytes(split))
    immutable_write(O/'audit/recovery_inventory.json',json_bytes(entries))
    csvout(O/'recovery/recovery61_inventory.csv',[{**r,'prior_window':json.dumps(r['prior_window']),'prior_observed':json.dumps(r['prior_observed']),'duplicate_candidates':json.dumps(r['duplicate_candidates']),'initial_failure_causes':'|'.join(r['initial_failure_causes'])} for r in entries])
    csvout(O/'recovery/annotation_failure_taxonomy.csv',tax)
    distribution=Counter(c for r in entries for c in r['initial_failure_causes'])
    atomic_json(O/'recovery/failure_distribution.json',{'source_count':61,'multi_label_counts':dict(distribution),'basis':'Existing visually adjudicated ledgers; new recovery reviews may add F2/F9 or revise diagnosis without rewriting this baseline.'})
    stages=[
        ('Source video','acquisition.Pilot / builder','MP4 URL,allowlist,limits','MP4/probe/SHA','requests + ffmpeg','retry/network/container failures','Existing fault injection;50 previous real videos','Reuse; one fresh wave only'),
        ('Content inspection','visual_index + actual Codex multimodal interface','Full-source contact sheets','Observed events','Model revision NOT_EXPOSED','Sparse context; unsupported identities','Actual images and hashes; no independent calibrated judge','Reuse full coverage; abstain; stage timing'),
        ('Event/query generation','query_generator.AgentEvidenceProvider','Actual inspected events','Specific query + coarse phase','Actual Codex interface','Scene-only descriptions; repetition; semantic hallucination','query_errors + evidence annotations','Bounded max2 distinct candidates, visible onset/offset ranking'),
        ('Temporal retrieval','temporal_labeler.localize + frozen V3','Query/events + CLIP512','Selected phase/peak','Pinned CLIP ViT-B32 CPU + V3','Global peak or late-only phase','Frozen code/SHA; uncalibrated similarities','Keep ranker; uniform query-only reviewer before generator'),
        ('Boundary estimation','temporal_labeler._refine','Selected CLIP peak','Raw window','Same CLIP, wrapper version separate','4s floor; +/-1s crosses cuts; start=max can omit early action','Dense QC; short-event sensitivity not proven sole cause','New V4 wrapper coexistence; shorter evidence resolution, never unconditional widening'),
        ('Verification','actual AI visual QC + freeze.validate_annotation','Before/inside/after + query/proposal','Accept/abstain decision','Actual Codex interface, unblinded QC','Generator-conditioned sampling; self-review dependence','No human claims; partial blinding disclosed','Separate full-uniform query-only protocol; explicit limitations'),
        ('Acceptance/manifest','BuilderSession + freeze.validate_annotation','Source/query/window/provenance/evidence','Immutable accepted record','No independent semantic oracle','Structurally valid does not imply correct','Strict real-source validation + actual visual verdict','Same contract; separate raw B0 and V4 artifacts'),
        ('Training readiness','cached_runner / campaign grouping','Accepted source/annotations + group restrictions','Future features/splits','M0 + frozen D500 features','Unknown campaign/human Test0','Historical314 tests + protectedSHA','Do not train or freeze new Test; no D500 re-encode')]
    csvout(O/'audit/pipeline_dependency_map.csv',[dict(zip(('stage','implementation','inputs','outputs','model','failure','current_validation','proposed_improvement'),v)) for v in stages])
    report=f'''# Audit before change — P2.1

Verified latest checkpoint: `{previous}`. Execution manifest SHA `{state['p2_execution_manifest_sha256']}`;{state['previous_bound_hashes_checked']} bound hashes PASS. Reconciled500+18=518 accepted;61 unique technical-valid unaccepted sources =22 pilot +39 P2. No raw MP4 copies. Full historical SHA sweep is separately recorded before execution.

## Pipeline and dependencies

Source → actual content inspection → event/query → CLIP/V3 retrieval → boundary wrapper → actual AI QC → strict contract → immutable builder → future training features. Per-stage modules, dependencies, models, failure modes and proposed improvements are in `pipeline_dependency_map.csv`; prior pipeline timings were not fully instrumented, so stage costs remain NOT_MEASURED until benchmark.

## Initial root causes

Ledger-derived multi-label distribution: {dict(distribution)}. This is not fresh visual GT. `_refine` has a4s minimum around peak and1s padding. Its `start=max(start,evidence_first-1)` can retain a late start, so early evidence may remain outside the output. These are concrete mechanisms to test, not proof every failure has that cause. Scene descriptions, repeated identities/actions, semantic hallucinations and near-duplicate uncertainty are separate causes. Existing18 accepted remain unchanged; all are AI-pseudo, not human GT.

## Available evidence and runtime

Only CPU torch{state['runtime']['torch']};CUDA={state['runtime']['cuda_available']};8 logical CPUs;available RAM{state['runtime']['memory_available']/2**30:.2f}GiB. Use one runtime,threads2,batch≤8 and a1GiB available-RAM guard. E free{state['disks']['E']['free']/2**30:.2f}GiB;retain15GiB free plus bounded processing reserve. Actual Codex image inspection is available; model revision not exposed. No independent calibrated local VLM has been demonstrated. CLIP/V3 scores are not calibrated correctness probabilities.

## Candidate metadata finding

All1707 pool rows expose only source_video_id,target_name,source_url,source_split. There is no title,description,industry,duration,brand or campaign field. `source_split` is provenance and must not become a quality score. Do not infer action suitability from opaque URL/ID strings. Implement transparent eligibility + optional real metadata scoring with equal-prior seeded fallback. Ranking superiority is not supportable if metadata is absent. Freeze policy and selection before one fresh25-source wave.

## Leakage and protocol

Development{len(dev)} /reserved offline{len(reserved)} source assignments are frozen in `evaluation_protocol.json`, keeping shared fingerprint-uncertainty components together. Because all61 were exposed previously, reserved offline sources are NOT genuinely unseen. They must not choose V4 thresholds after freeze; fresh25 new sources provide the actual held-out-source check. Unknown campaign links remain unknown. No historical Test30 labels/metrics or source_split labels are optimization inputs.

## Protection

Preserve existing P0/P1/P14/P16/P2 directories, old E ledgers/raw/cache and13 code files pinned by the prior P2 execution manifest. Do not edit them even to add test guards. New modules/configs/scripts and a separate pytest plugin will provide P2.1 behavior. New cache and fresh downloads stay under a new E namespace. No GPU training, D500 feature re-encoding, ZIP, migration or cloud writes.
'''
    (O/'audit/audit_before_change.md').write_text(report,encoding='utf-8')
    (O/'recovery/annotation_root_cause_analysis.md').write_text(report.split('## Initial root causes')[1].split('## Available evidence')[0],encoding='utf-8')
    print({'inventory':len(entries),'initial_causes':dict(distribution),'dev':len(dev),'reserved':len(reserved)})

if __name__=='__main__':main()
