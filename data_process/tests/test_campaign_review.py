"""Synthetic protocol fixtures only; no real human scores/labels are produced."""
import copy
import hashlib
import json
import math
from pathlib import Path
import pytest
from data_process.splitting.campaign import (group_campaigns,deterministic_campaign_split,rows_for_plan,
    validate_splits,nested_training_subsets,double_review_schedule,freeze_splits,fingerprint_distance)
from data_process.annotation.independent_review import ReviewStore,validate_review,agreement


def registry(n):
    return [{'vid':f'v{i}','source_id':f's{i}','source_sha256':hashlib.sha256(f'video{i}'.encode()).hexdigest(),
        'source_url':f'https://example.org/{i}.mp4','historically_exposed':i<500,'campaign_status':'SINGLETON_REVIEWED'} for i in range(n)]

def test_group_evidence_quarantine_and_determinism():
    vids=registry(6);vids[0].update(campaign_id='campaignX',campaign_status='CONFIRMED');vids[1].update(campaign_id='campaignX',campaign_status='CONFIRMED')
    vids[2].update(visual_fingerprint='0000000000000000',fingerprint_algorithm='dhash64_v1');vids[3].update(visual_fingerprint='0000000000000001',fingerprint_algorithm='dhash64_v1')
    vids[4].update(brand_id='brand',product_id='bottle');vids[5].update(brand_id='brand',product_id='bottle')
    g=group_campaigns(vids);assert g['video_to_group']['v0']==g['video_to_group']['v1']
    assert set(g['quarantined_videos'])=={'v2','v3','v4','v5'}
    assert g==group_campaigns(list(reversed(vids)))
    relations=[{'a':'v2','b':'v3','relation':'distinct','reviewer_id':'fixture_reviewer','evidence':'separate reviewed campaigns'},
               {'a':'v4','b':'v5','relation':'same','reviewer_id':'fixture_reviewer','evidence':'same campaign source metadata'}]
    reviewed=group_campaigns(vids,relations);assert reviewed['quarantined_videos']==[]
    assert reviewed['video_to_group']['v4']==reviewed['video_to_group']['v5']
    assert fingerprint_distance('0000000000000000','0000000000000001')==1
    with pytest.raises(ValueError):group_campaigns(vids,[{'a':'v0','b':'v1','relation':'distinct','reviewer_id':'fixture_reviewer','evidence':'contradictory'}])


def fixture_review(row,double=False):
    # In-memory synthetic event chain, never a real annotation or IAA claim.
    original={'kind':'REGISTER','qid':str(row['qid']),'vid':row['vid'],'query':row['query'],'duration':row['duration'],
        'original_ai_gt':row['relevant_windows'][0],'original_ai_query':row['query']}
    events=[];previous=None;annotations=[]
    def event(e):
        nonlocal previous
        item={**e,'revision':len(events)+1,'previous_sha256':previous,'recorded_at':'2026-11-03T00:00:00+00:00'}
        events.append(item);previous=hashlib.sha256(json.dumps(item,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()
    event(original)
    for who in (['fixture_A','fixture_B'] if double else ['fixture_A']):
        annotation={'annotator_id':who,'interval':row['relevant_windows'][0],'source':'HUMAN','blind':True,'submitted_at':'2026-11-03T00:00:00+00:00','comments':'synthetic test fixture'}
        annotations.append(annotation);event({'kind':'ANNOTATE',**annotation})
    event({'kind':'ADJUDICATE','reviewer_id':'fixture_C','interval':row['relevant_windows'][0],'comments':'synthetic adjudication'})
    return {k:v for k,v in original.items() if k!='kind'}|{'status':'HUMAN_VERIFIED','annotations':annotations,'reviewer_id':'fixture_C',
        'adjudicated_reference':row['relevant_windows'][0],'revision_history':events}


def full_protocol_fixture():
    videos=registry(1000);g=group_campaigns(videos);plan=deterministic_campaign_split(g);rows=[]
    for v in videos:
        for i in range(2 if plan['assignments'][v['vid']]=='test' else 1):
            rows.append({'qid':v['vid']+f'_q{i}','vid':v['vid'],'query':f'Distinct event {i} for {v["vid"]}',
                'duration':20,'relevant_windows':[[2+i,6+i]],'metadata':{'source_video_id':v['source_id'],
                'source_video_sha256':v['source_sha256'],'source_url':v['source_url'],'review_provenance':'AI_PSEUDO_LABELED'}})
    splits=rows_for_plan(rows,plan)
    for i,r in enumerate(splits['test']):r['metadata'].update(review_provenance='HUMAN_VERIFIED',human_review=fixture_review(r,i<115))
    return plan,splits


def test_fresh_1000_video_300_query_protocol_and_immutable_test(tmp_path):
    plan,splits=full_protocol_fixture()
    assert plan['video_counts']=={'train':700,'val':150,'test':150}
    assert len(splits['test'])==300
    result=validate_splits(splits,grouping=plan['grouping']);assert result['double_annotated_queries']==115
    assert deterministic_campaign_split(group_campaigns(list(reversed(registry(1000)))))==plan
    nested=nested_training_subsets(splits['train']);assert {r['qid'] for r in nested['140']}<={r['qid'] for r in nested['440']}<={r['qid'] for r in nested['700']}
    schedule=double_review_schedule([r for rs in splits.values() for r in rs]);assert len(schedule)==115
    freeze_splits(tmp_path/'frozen',splits,grouping=plan['grouping']);before=(tmp_path/'frozen/sealed_test/test.jsonl').read_bytes()
    freeze_splits(tmp_path/'frozen',splits,grouping=plan['grouping']);assert (tmp_path/'frozen/sealed_test/test.jsonl').read_bytes()==before
    training=(tmp_path/'frozen/training/train.jsonl').read_text()+(tmp_path/'frozen/training/val.jsonl').read_text()
    assert all(r['qid'] not in training for r in splits['test'])
    changed=copy.deepcopy(splits);changed['test'][0]['query']='rewritten test'
    with pytest.raises(ValueError):freeze_splits(tmp_path/'frozen',changed,grouping=plan['grouping'])


@pytest.mark.parametrize('failure',['campaign','source','hash','duplicate_query','timestamp','AI_human','exposure','double','minimum_queries','forged_history'])
def test_protocol_rejects_leakage_and_false_review(failure):
    plan,s=full_protocol_fixture()
    if failure=='campaign':s['val'][0]['metadata']['campaign_group']=s['train'][0]['metadata']['campaign_group']
    elif failure=='source':s['test'][0]['metadata']['source_video_id']=s['train'][0]['metadata']['source_video_id']
    elif failure=='hash':s['test'][0]['metadata']['source_video_sha256']=s['train'][0]['metadata']['source_video_sha256']
    elif failure=='duplicate_query':s['test'][1]['query']=s['test'][0]['query']
    elif failure=='timestamp':s['test'][0]['relevant_windows']=[[-1,4]]
    elif failure=='AI_human':s['test'][0]['metadata'].pop('human_review')
    elif failure=='exposure':s['test'][0]['metadata']['historically_exposed']=True
    elif failure=='double':
        for r in s['test']:r['metadata']['human_review']=fixture_review(r,False)
    elif failure=='minimum_queries':s['test']=s['test'][:299]
    else:s['test'][0]['metadata']['human_review']['revision_history']=[{}]
    with pytest.raises(ValueError):validate_splits(s,grouping=plan['grouping'])


def test_real_workflow_blinding_revisions_and_agreement(tmp_path):
    store=ReviewStore(tmp_path,'fixture_reviews_v1');r=store.register('q1','v1','Person lifts bottle',20,[2,6])
    assert agreement(r)['mean_temporal_iou'] is None
    packet=store.blind_packet('q1');assert 'original_ai_gt' not in packet and 'annotations' not in packet
    with pytest.raises(ValueError):store.adjudicate('q1','fixture_C',2,6,'no annotation exists')
    with pytest.raises(ValueError):store.annotate('q1','AI_PIPELINE',2,6)
    store.annotate('q1','fixture_A',2,6);store.annotate('q1','fixture_B',3,7)
    r=store.adjudicate('q1','fixture_C',2.5,6.5,'Synthetic test rationale')
    assert r['status']=='HUMAN_VERIFIED';score=agreement(r)
    assert score['mean_temporal_iou']==pytest.approx(3/5);assert score['mean_start_disagreement_seconds']==1;assert score['mean_end_disagreement_seconds']==1
    revised=store.annotate('q1','fixture_A',2.2,6.2,comments='independent revised estimate')
    assert revised['status']=='IN_REVIEW' and len(revised['annotations'])==2 and revised['revision']==5
    with pytest.raises(ValueError):store.register('q1','v1','changed query',20,[2,6])
    with pytest.raises(ValueError):store.adjudicate('q1','fixture_A',2,6,'self review')
    with pytest.raises(ValueError):store.register('../unsafe','v1','query',20,[2,6])
