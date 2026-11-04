"""Evidence-based grouping and deterministic fresh evaluation splits.

Brand/product/visual similarity are candidates for review, never proof of a
campaign. Uncertain links quarantine whole connected confirmed groups.
"""
from __future__ import annotations
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path

def fingerprint_distance(a,b):
    if len(a)!=16 or len(b)!=16:raise ValueError('Expected 64-bit hexadecimal fingerprints')
    return (int(a,16)^int(b,16)).bit_count()

def image_dhash(path):
    from PIL import Image
    with Image.open(path) as im:
        values=list(im.convert('L').resize((9,8)).getdata())
    bits=0
    for y in range(8):
        for x in range(8):bits=(bits<<1)|int(values[y*9+x]>values[y*9+x+1])
    return f'{bits:016x}'

def group_campaigns(videos,relations=(),near_threshold=4):
    from ..lumae_scale.builder import normalized_url
    if not 0<=near_threshold<=64:raise ValueError('Invalid fingerprint threshold')
    by={v['vid']:dict(v) for v in videos}
    if len(by)!=len(videos):raise ValueError('Video registry must contain unique vid records')
    parent={v:v for v in by}
    def find(x):
        while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
        return x
    def union(a,b):
        a,b=find(a),find(b);parent[max(a,b)]=min(a,b)
    evidence=[];uncertain=[];distinct=set()
    for r in relations:
        a,b=r['a'],r['b']
        if a not in by or b not in by or a==b:raise ValueError('Unknown/self campaign relation')
        relation=r['relation']
        if relation not in ('same','uncertain','distinct'):raise ValueError('Unknown relation')
        if relation!='uncertain' and (not r.get('reviewer_id') or not r.get('evidence')):raise ValueError('Confirmed campaign decisions require review evidence')
        if relation=='distinct':distinct.add(frozenset((a,b)))
        elif relation=='same':union(a,b);evidence.append(dict(r))
        else:uncertain.append((a,b,'unresolved_relation'))
    indexes={k:{} for k in ('source_id','source_sha256','source_url','campaign_id')}
    for vid,v in sorted(by.items()):
        for key,index in indexes.items():
            value=v.get(key)
            if not value or (key=='campaign_id' and v.get('campaign_status')!='CONFIRMED'):continue
            if key=='source_url':value=normalized_url(value)
            if value in index:
                other=index[value];union(vid,other);evidence.append({'a':vid,'b':other,'relation':'same','evidence':key})
            else:index[value]=vid
    ids=sorted(by)
    for i,a in enumerate(ids):
        for b in ids[i+1:]:
            if find(a)==find(b) or frozenset((a,b)) in distinct:continue
            x,y=by[a],by[b]
            if x.get('visual_fingerprint') and y.get('visual_fingerprint') and x.get('fingerprint_algorithm')==y.get('fingerprint_algorithm')=='dhash64_v1':
                if fingerprint_distance(x['visual_fingerprint'],y['visual_fingerprint'])<=near_threshold:uncertain.append((a,b,'near_visual_match'))
            if x.get('brand_id') and x.get('product_id') and (x['brand_id'],x['product_id'])==(y.get('brand_id'),y.get('product_id')):
                uncertain.append((a,b,'same_brand_product_campaign_unconfirmed'))
    for pair in distinct:
        a,b=tuple(pair)
        if find(a)==find(b):raise ValueError('Contradictory distinct/same campaign evidence')
    groups=defaultdict(list)
    for vid in ids:groups[find(vid)].append(vid)
    group_ids={root:'campaign_'+hashlib.sha256('|'.join(members).encode()).hexdigest()[:20] for root,members in groups.items()}
    quarantine_roots={find(x) for a,b,_ in uncertain if find(a)!=find(b) for x in (a,b)}
    mapping={vid:group_ids[find(vid)] for vid in ids}
    return {'video_to_group':mapping,'groups':{group_ids[k]:v for k,v in sorted(groups.items())},
        'quarantined_videos':sorted(v for v in ids if find(v) in quarantine_roots),
        'uncertain_matches':[{'a':a,'b':b,'reason':why} for a,b,why in sorted(set(uncertain))],
        'confirmed_evidence':sorted(evidence,key=lambda r:json.dumps(r,sort_keys=True)),'registry':by}

def deterministic_campaign_split(grouping,targets=None,seed=20260930):
    targets=targets or {'train':700,'val':150,'test':150}
    if set(targets)!={'train','val','test'} or any(not isinstance(v,int) or v<0 for v in targets.values()):raise ValueError('Invalid targets')
    quarantine=set(grouping['quarantined_videos']);registry=grouping['registry'];groups=grouping['groups']
    active={k:v for k,v in groups.items() if not set(v)&quarantine};assign={};counts={s:0 for s in targets}
    def rank(g):return hashlib.sha256(f'{seed}:{g}'.encode()).hexdigest()
    remaining=sorted(active,key=rank)
    for split in ('test','val'):
        for g in list(remaining):
            vids=active[g]
            if split=='test' and any(registry[v].get('historically_exposed',True) or registry[v].get('campaign_status') not in ('CONFIRMED','SINGLETON_REVIEWED') for v in vids):continue
            if counts[split]<targets[split] and abs(counts[split]+len(vids)-targets[split])<=abs(counts[split]-targets[split]):
                for vid in vids:assign[vid]=split
                counts[split]+=len(vids);remaining.remove(g)
    for g in remaining:
        for vid in active[g]:assign[vid]='train'
        counts['train']+=len(active[g])
    return {'seed':seed,'targets':targets,'video_counts':counts,'deviation':{s:counts[s]-targets[s] for s in targets},
        'assignments':assign,'grouping':grouping,'status':'PLANNED_NOT_FROZEN','blind_test_ready':False}

def rows_for_plan(rows,plan):
    result={s:[] for s in ('train','val','test')}
    for row in rows:
        vid=row['vid']
        if vid not in plan['assignments']:continue
        meta={**row.get('metadata',{}),'campaign_group':plan['grouping']['video_to_group'][vid],
              'historically_exposed':plan['grouping']['registry'][vid].get('historically_exposed',True)}
        result[plan['assignments'][vid]].append({**row,'metadata':meta})
    return result

def validate_splits(splits,*,grouping=None,require_human_test=True,min_test_queries=300,exposed_qids=(),double_fraction=.1,legacy_review_qids=()):
    from ..validation.temporal import validate_duration,validate_windows
    from ..annotation.independent_review import validate_review
    if set(splits)!={'train','val','test'}:raise ValueError('Three explicit splits required')
    owner={};qids=set();source_owner={};hash_owner={};vid_owner={};url_owner={};campaign_owner={};double=0;test_queries=defaultdict(set)
    allrows=[r for s in splits.values() for r in s]
    for split,rows in splits.items():
        for r in rows:
            qid=str(r['qid']);vid=r['vid'];m=r.get('metadata',{});group=m.get('campaign_group')
            if not group:raise ValueError('Missing campaign group')
            if grouping and (grouping['video_to_group'].get(vid)!=group or vid in grouping['quarantined_videos']):raise ValueError('Unreviewed/quarantined campaign assignment')
            if m.get('campaign_status')=='CONFIRMED' and m.get('campaign_id'):
                if m['campaign_id'] in campaign_owner and campaign_owner[m['campaign_id']]!=split:raise ValueError('Confirmed campaign crosses splits')
                campaign_owner[m['campaign_id']]=split
            if group in owner and owner[group]!=split:raise ValueError('Campaign crosses splits')
            owner[group]=split
            if qid in qids:raise ValueError('Duplicate query ID')
            qids.add(qid)
            if vid in vid_owner and vid_owner[vid]!=split:raise ValueError('Video crosses splits')
            vid_owner[vid]=split
            if not validate_duration(r['duration'])[0] or not validate_windows(r['relevant_windows'],r['duration'])[0]:raise ValueError('Invalid timestamp/duration')
            for key,seen in [('source_video_id',source_owner),('source_video_sha256',hash_owner)]:
                value=m.get(key)
                if not value:raise ValueError('Missing source identity')
                if value in seen and seen[value]!=vid:raise ValueError('Source/hash collision requires deduplication')
                seen[value]=vid
            if m.get('source_url'):
                from ..lumae_scale.builder import normalized_url
                u=normalized_url(m['source_url'])
                if u in url_owner and url_owner[u]!=vid:raise ValueError('Source URL collision requires deduplication')
                url_owner[u]=vid
            review=m.get('human_review')
            if review:
                validate_review(review)
                if review['qid']!=qid or review['vid']!=vid or review['query']!=r['query'] or float(review['duration'])!=float(r['duration']):raise ValueError('Human review identity mismatch')
                if review['status']=='HUMAN_VERIFIED' and r['relevant_windows']!=[review['adjudicated_reference']]:raise ValueError('GT differs from adjudicated reference')
                if len(review['annotations'])>=2:double+=1
            legacy_allowed=split!='test' and qid in set(map(str,legacy_review_qids)) and m.get('historically_exposed',True)
            if m.get('review_provenance')=='HUMAN_VERIFIED' and not legacy_allowed and (not review or review['status']!='HUMAN_VERIFIED'):raise ValueError('AI-only cannot be marked human verified')
            if split=='test':
                if qid in set(map(str,exposed_qids)) or m.get('historically_exposed',True):raise ValueError('Historically exposed sample in fresh Test')
                query=' '.join(r['query'].casefold().split())
                if query in test_queries[vid]:raise ValueError('Duplicate query within Test video')
                test_queries[vid].add(query)
                if require_human_test and (not review or review['status']!='HUMAN_VERIFIED'):raise ValueError('Every Test query requires genuine human verification')
    if len(splits['test'])<min_test_queries:raise ValueError('Test requires at least 300 independently annotated query moments')
    if require_human_test:
        if any(len(q)<2 for q in test_queries.values()):raise ValueError('At least two distinct Test queries per video')
        if double<math.ceil(len(allrows)*double_fraction):raise ValueError('At least 10% independent double annotation required')
    return {'status':'PASS','query_counts':{s:len(r) for s,r in splits.items()},'double_annotated_queries':double,
            'limitations':'Schema/evidence validation cannot authenticate human identities without an external review authority'}

def nested_training_subsets(train_rows,sizes=(140,440,700),seed=20260930):
    if len({r['qid'] for r in train_rows})!=len(train_rows):raise ValueError('Duplicate training query')
    ordered=sorted(train_rows,key=lambda r:hashlib.sha256(f'{seed}:{r["qid"]}'.encode()).hexdigest())
    if any(n<1 or n>len(ordered) for n in sizes):raise ValueError('Insufficient approved Train samples')
    return {str(n):ordered[:n] for n in sorted(sizes)}

def double_review_schedule(rows,seed=20260930,fraction=.1):
    if not 0<fraction<=1:raise ValueError('Invalid double review fraction')
    n=math.ceil(len(rows)*fraction)
    # Reserve a representative short-event stratum, without manufacturing reviews.
    rank=lambda r:hashlib.sha256(f'{seed}:{r["qid"]}'.encode()).hexdigest()
    short=sorted([r for r in rows if min(b-a for a,b in r['relevant_windows'])<=2],key=rank)
    chosen=short[:min(len(short),math.ceil(n/2))];ids={r['qid'] for r in chosen}
    chosen+=sorted([r for r in rows if r['qid'] not in ids],key=rank)[:n-len(chosen)]
    return [r['qid'] for r in chosen]

def freeze_splits(output,splits,*,grouping,exposed_qids=(),legacy_review_qids=()):
    from ..lumae_scale.incremental import immutable_write,json_bytes
    from ..lumae_scale.freeze import _jsonl_bytes
    validation=validate_splits(splits,grouping=grouping,exposed_qids=exposed_qids,legacy_review_qids=legacy_review_qids)
    output=Path(output)
    if output.is_dir() and any(output.iterdir()):
        if not (output/'COMPLETE.json').is_file():raise FileExistsError('Use a new output directory, never append into historical/partial artifacts')
        previous=json.loads((output/'COMPLETE.json').read_text())
        for rel,expected in previous['checksums'].items():
            p=(output/rel).resolve()
            if not p.is_relative_to(output.resolve()) or hashlib.sha256(p.read_bytes()).hexdigest()!=expected:raise ValueError('Frozen split checksum changed')
    for split,rows in splits.items():
        folder='sealed_test' if split=='test' else 'training'
        immutable_write(output/folder/(split+'.jsonl'),_jsonl_bytes(sorted(rows,key=lambda r:(str(r['qid']),r['vid']))))
    immutable_write(output/'split_validation.json',json_bytes(validation))
    checks={str(p.relative_to(output)):hashlib.sha256(p.read_bytes()).hexdigest() for p in output.rglob('*.jsonl')}
    immutable_write(output/'COMPLETE.json',json_bytes({'checksums':checks,'test_selection_feedback_allowed':False}))
    return validation

def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('plan','validate','freeze'))
    p.add_argument('--registry',type=Path);p.add_argument('--relations',type=Path);p.add_argument('--splits',type=Path);p.add_argument('--plan',type=Path);p.add_argument('--output',type=Path);p.add_argument('--seed',type=int,default=20260930)
    a=p.parse_args()
    if a.action=='plan':
        if not a.registry or not a.output:p.error('--registry and --output required')
        videos=json.loads(a.registry.read_text(encoding='utf-8'));relations=json.loads(a.relations.read_text(encoding='utf-8')) if a.relations else []
        result=deterministic_campaign_split(group_campaigns(videos,relations),seed=a.seed)
        with a.output.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
        print(json.dumps({k:v for k,v in result.items() if k not in ('grouping','assignments')}))
    else:
        if not a.splits or not a.plan:p.error('--splits and --plan required')
        splits=json.loads(a.splits.read_text(encoding='utf-8'));grouping=json.loads(a.plan.read_text(encoding='utf-8'))['grouping']
        if a.action=='freeze':
            if not a.output:p.error('--output required')
            result=freeze_splits(a.output,splits,grouping=grouping)
        else:result=validate_splits(splits,grouping=grouping)
        print(json.dumps(result))

if __name__=='__main__':main()
