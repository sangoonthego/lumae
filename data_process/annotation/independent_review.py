"""Append-only independent temporal reviews; no AI-only human verification.

Operator identities are attestations supplied by actual reviewers, not invented
by this module. Blind packets omit AI GT and previous reviewers' boundaries.
"""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
from datetime import datetime,timezone

from ..validation.temporal import validate_duration,validate_windows

def interval(value,duration):
    if not validate_duration(duration)[0] or not validate_windows([value],duration)[0]:raise ValueError('Invalid temporal interval')

def human_id(value):
    if not isinstance(value,str) or not value.strip() or value.upper().startswith(('AI','MODEL','PIPELINE','SYNTHETIC')):raise ValueError('Actual human annotator/reviewer identity required')

def validate_review(r):
    if not r.get('qid') or not r.get('vid') or not r.get('query','').strip():raise ValueError('Missing review identity')
    interval(r['original_ai_gt'],r['duration'])
    if not r.get('original_ai_query','').strip():raise ValueError('Original AI query provenance required')
    ids=[]
    for a in r['annotations']:
        human_id(a['annotator_id']);interval(a['interval'],r['duration']);ids.append(a['annotator_id'])
        if a.get('source')!='HUMAN' or a.get('blind') is not True or not a.get('submitted_at'):raise ValueError('Independent human evidence required')
        if datetime.fromisoformat(a['submitted_at']).tzinfo is None:raise ValueError('Annotation timestamp requires timezone')
    if len(ids)!=len({x.casefold().strip() for x in ids}):raise ValueError('Double annotation requires distinct independent reviewers')
    if r['status'] not in ('DRAFT','IN_REVIEW','HUMAN_VERIFIED','REJECTED'):raise ValueError('Unknown review status')
    if r['status']=='HUMAN_VERIFIED':
        if not ids:raise ValueError('AI-only cannot be human verified')
        human_id(r.get('reviewer_id'))
        if r['reviewer_id'].casefold().strip() in {x.casefold().strip() for x in ids}:raise ValueError('Adjudicator must be independent from annotators')
        interval(r['adjudicated_reference'],r['duration'])
        if not r.get('revision_history'):raise ValueError('Missing provenance/revision history')
    history=r.get('revision_history',[])
    if not history:raise ValueError('Missing append-only review provenance')
    previous=None;active={};status='DRAFT';last_adjudication=None
    for i,e in enumerate(history,1):
        if e.get('revision')!=i or e.get('previous_sha256')!=previous:raise ValueError('Invalid review history chain')
        previous=hashlib.sha256(json.dumps(e,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()
        if i==1:
            if e.get('kind')!='REGISTER' or any(e.get(k)!=r[k] for k in ('qid','vid','query','duration','original_ai_gt','original_ai_query')):raise ValueError('Original review identity changed')
        elif e['kind']=='ANNOTATE':
            active[e['annotator_id']]={k:e[k] for k in ('annotator_id','interval','source','blind','submitted_at','comments')};status='IN_REVIEW';last_adjudication=None
        elif e['kind']=='ADJUDICATE':status='HUMAN_VERIFIED';last_adjudication=e
        elif e['kind']=='REJECT':status='REJECTED'
        else:raise ValueError('Unknown review event')
    if r['annotations']!=[active[k] for k in sorted(active)] or r['status']!=status:raise ValueError('Review state does not match event history')
    if status=='HUMAN_VERIFIED' and (last_adjudication['reviewer_id']!=r['reviewer_id'] or last_adjudication['interval']!=r['adjudicated_reference']):raise ValueError('Adjudication evidence mismatch')
    return True

class ReviewStore:
    def __init__(self,root,version):
        if not re.fullmatch(r'[A-Za-z0-9_]+',version) or version in ('lumae_ads_v1','lumae_ads_d200','lumae_ads_d500'):raise ValueError('A new review version is required')
        self.root=Path(root)/version

    def _events(self,qid):
        if not re.fullmatch(r'[A-Za-z0-9_-]+',qid):raise ValueError('Unsafe query ID')
        paths=sorted((self.root/qid).glob('*.json'));events=[];previous=None
        for index,p in enumerate(paths,1):
            data=p.read_bytes();e=json.loads(data)
            if e['revision']!=index or e['previous_sha256']!=previous:raise ValueError('Review revision chain corrupted')
            previous=hashlib.sha256(data).hexdigest();events.append(e)
        return events,previous

    def _append(self,qid,event):
        events,previous=self._events(qid);event={**event,'revision':len(events)+1,'previous_sha256':previous,'recorded_at':datetime.now(timezone.utc).isoformat()}
        path=self.root/qid/f'{len(events)+1:08d}.json';path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('x',encoding='utf-8',newline='\n') as f:json.dump(event,f,sort_keys=True,ensure_ascii=False,allow_nan=False)

    def register(self,qid,vid,query,duration,original_ai_gt,original_ai_query=None):
        interval(original_ai_gt,duration)
        initial={'kind':'REGISTER','qid':qid,'vid':vid,'query':query,'duration':duration,'original_ai_gt':original_ai_gt,'original_ai_query':original_ai_query or query}
        events,_=self._events(qid)
        if events:
            if any(events[0].get(k)!=v for k,v in initial.items()):raise ValueError('Original review input is immutable')
            return self.get(qid)
        if not query.strip():raise ValueError('Empty query')
        self._append(qid,initial);return self.get(qid)

    def get(self,qid):
        events,_=self._events(qid)
        if not events or events[0]['kind']!='REGISTER':raise FileNotFoundError(qid)
        r={k:v for k,v in events[0].items() if k not in ('kind','previous_sha256','recorded_at')};r.update(status='DRAFT',annotations=[],revision_history=events)
        annotations={}
        for e in events[1:]:
            if e['kind']=='ANNOTATE':annotations[e['annotator_id']]={k:e[k] for k in ('annotator_id','interval','source','blind','submitted_at','comments')};r['status']='IN_REVIEW';r.pop('adjudicated_reference',None);r.pop('reviewer_id',None)
            elif e['kind']=='ADJUDICATE':r.update(status='HUMAN_VERIFIED',reviewer_id=e['reviewer_id'],adjudicated_reference=e['interval'],reviewer_comments=e['comments'])
            elif e['kind']=='REJECT':r.update(status='REJECTED',reviewer_comments=e['comments'])
            else:raise ValueError('Unknown review event')
        r['annotations']=[annotations[k] for k in sorted(annotations)];r['revision']=len(events);validate_review(r);return r

    def blind_packet(self,qid):
        r=self.get(qid);return {k:r[k] for k in ('qid','vid','query','duration')}

    def annotate(self,qid,annotator_id,start,end,*,comments='',blind=True):
        r=self.get(qid);human_id(annotator_id);interval([start,end],r['duration'])
        if blind is not True:raise ValueError('Nonblind annotation cannot satisfy independent review')
        if any(a['annotator_id']!=annotator_id and a['annotator_id'].casefold().strip()==annotator_id.casefold().strip() for a in r['annotations']):raise ValueError('Use the same canonical annotator ID for revisions')
        self._append(qid,{'kind':'ANNOTATE','annotator_id':annotator_id,'interval':[start,end],'source':'HUMAN','blind':True,'comments':comments,'submitted_at':datetime.now(timezone.utc).isoformat()})
        return self.get(qid)

    def adjudicate(self,qid,reviewer_id,start,end,comments):
        r=self.get(qid);human_id(reviewer_id);interval([start,end],r['duration'])
        if not r['annotations'] or reviewer_id.casefold().strip() in {a['annotator_id'].casefold().strip() for a in r['annotations']}:raise ValueError('Independent adjudicator and actual annotations required')
        if not comments.strip():raise ValueError('Adjudication rationale required')
        self._append(qid,{'kind':'ADJUDICATE','reviewer_id':reviewer_id,'interval':[start,end],'comments':comments});return self.get(qid)

    def reject(self,qid,reviewer_id,comments):
        self.get(qid);human_id(reviewer_id)
        if not comments.strip():raise ValueError('Rejection rationale required')
        self._append(qid,{'kind':'REJECT','reviewer_id':reviewer_id,'comments':comments});return self.get(qid)

def agreement(review):
    validate_review(review);pairs=[]
    for a,b in itertools.combinations(review['annotations'],2):
        x,y=a['interval'],b['interval'];intersection=max(0,min(x[1],y[1])-max(x[0],y[0]));union=max(x[1],y[1])-min(x[0],y[0])
        pairs.append({'annotators':[a['annotator_id'],b['annotator_id']],'temporal_iou':intersection/union if union else 0.,
            'start_disagreement_seconds':abs(x[0]-y[0]),'end_disagreement_seconds':abs(x[1]-y[1])})
    return {'qid':review['qid'],'independent_pairs':len(pairs),'pairs':pairs,
        'mean_temporal_iou':sum(x['temporal_iou'] for x in pairs)/len(pairs) if pairs else None,
        'mean_start_disagreement_seconds':sum(x['start_disagreement_seconds'] for x in pairs)/len(pairs) if pairs else None,
        'mean_end_disagreement_seconds':sum(x['end_disagreement_seconds'] for x in pairs)/len(pairs) if pairs else None}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--version',required=True)
    sub=p.add_subparsers(dest='action',required=True)
    r=sub.add_parser('register');r.add_argument('--input',type=Path,required=True)
    for name in ('packet','annotate','adjudicate','agreement','export'):
        q=sub.add_parser(name);q.add_argument('--qid',required=True)
        if name in ('annotate','adjudicate'):
            q.add_argument('--human-id',required=True);q.add_argument('--start',type=float,required=True);q.add_argument('--end',type=float,required=True);q.add_argument('--comments',default='')
        if name=='export':q.add_argument('--output',type=Path,required=True)
    a=p.parse_args();store=ReviewStore(a.root,a.version)
    if a.action=='register':result=store.register(**json.loads(a.input.read_text(encoding='utf-8')))
    elif a.action=='packet':result=store.blind_packet(a.qid)
    elif a.action=='annotate':result=store.annotate(a.qid,a.human_id,a.start,a.end,comments=a.comments)
    elif a.action=='adjudicate':result=store.adjudicate(a.qid,a.human_id,a.start,a.end,a.comments)
    elif a.action=='agreement':result=agreement(store.get(a.qid))
    else:
        result=store.get(a.qid)
        with a.output.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    if a.action=='annotate':result={k:result[k] for k in ('qid','status','revision')}
    print(json.dumps(result,ensure_ascii=True,indent=2))

if __name__=='__main__':main()
