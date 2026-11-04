"""Configurable incremental construction, separate from immutable legacy D500.

CLI is offline: planning/verification only. Download API requires a future explicit
authorization; source_pool.download retains HTTP Range resume. Acceptance uses
the established annotation validator with an explicit storage root.
"""
from __future__ import annotations
from dataclasses import dataclass
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
import re
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode

from .models import atomic_json,digest,read_json,load_jsonl
from .incremental import immutable_write,json_bytes
from .freeze import canonical_row,validate_annotation,_jsonl_bytes
from .preparation import metadata_preflight,classify_failure,FailureKind
from .source_pool import download
from .state_machine import validate_transition
from ..adsqa.video_validation import probe_video_file
from ..validation.temporal import validate_duration,validate_windows
from ..annotation.assisted_deployment import ALGORITHM_SHA256

HISTORICAL={'lumae_ads_v1','lumae_ads_d200','lumae_ads_d500'}

def normalized_url(url):
    p=urlsplit(url.strip())
    if p.scheme.lower() not in ('http','https') or not p.hostname or p.username or p.password:raise ValueError('Invalid source URL')
    host=p.hostname.lower()
    if p.port and p.port not in (80,443):host+=':'+str(p.port)
    return urlunsplit((p.scheme.lower(),host,p.path or '/',urlencode(sorted(parse_qsl(p.query,keep_blank_values=True))),''))

def safe_under(root,relative):
    p=Path(relative)
    if p.is_absolute() or '..' in p.parts:raise ValueError('Only storage-root relative paths are allowed')
    p=(Path(root)/p).resolve()
    if not p.is_relative_to(Path(root).resolve()):raise ValueError('Path escapes configured root')
    return p

@dataclass(frozen=True)
class BuildConfig:
    version:str
    parent_version:str
    parent_dir:Path
    data_root:Path
    target_videos:int=1000
    seed:int=20260930
    max_duration:float=150
    max_download_retries:int=5
    split_policy:str='explicit_campaign_assignments'
    parent_asset_root:Path|None=None

    def __post_init__(self):
        if self.version in HISTORICAL or not re.fullmatch(r'[A-Za-z0-9_]+',self.version):raise ValueError('Use a new, safe version name')
        if self.target_videos<1 or self.max_download_retries<1:raise ValueError('Invalid target/retry budget')
        if self.split_policy not in ('explicit_campaign_assignments','historical_extension'):raise ValueError('Unknown split policy')
        if self.dataset.resolve().is_relative_to(self.parent_dir.resolve()) or self.parent_dir.resolve().is_relative_to(self.dataset.resolve()):raise ValueError('Output overlaps frozen parent')

    @property
    def work(self):return self.data_root/'work'/self.version
    @property
    def dataset(self):return self.data_root/'datasets'/self.version
    @property
    def videos(self):return self.data_root/'sources/adsqa/videos'
    @property
    def visual_cache(self):return self.data_root/'cache'/self.version/'visual_index'

    @classmethod
    def load(cls,path,repo_root):
        v=read_json(path);parent=Path(v['parent_dir'])
        return cls(v['version'],v['parent_version'],parent if parent.is_absolute() else Path(repo_root)/parent,
            Path(v['data_root']),v['target_videos'],v['seed'],v.get('max_duration',150),v.get('max_download_retries',5),v.get('split_policy','explicit_campaign_assignments'),
            Path(repo_root)/v.get('parent_asset_root','.'))

def parent_snapshot(config):
    d=config.parent_dir;m=read_json(d/'dataset_manifest.json');p=d/f'{config.parent_version}_all.jsonl'
    if digest(p)!=m['dataset_sha256']:raise ValueError('Parent canonical changed')
    for line in (d/'checksums.sha256').read_text().splitlines():
        h,n=line.split('  ',1)
        if digest(safe_under(d,n))!=h:raise ValueError('Parent checksum mismatch: '+n)
    rows=load_jsonl(p);n=len({r['vid'] for r in rows})
    if n!=m.get('target_videos',m['total_samples']) or len({r['qid'] for r in rows})!=len(rows):raise ValueError('Parent identity mismatch')
    if config.target_videos<=n:raise ValueError('Target must exceed parent unique videos')
    return m,rows,{str(p):digest(p) for p in d.iterdir() if p.is_file()}

def candidate_plan(config,pool):
    parent,rows,_=parent_snapshot(config);ids={r.get('metadata',{}).get('source_video_id',r['vid'].removeprefix('lumae_ads_')) for r in rows}
    hashes={r['sha256'] for r in parent['source_video_sha256'].values()};urls={normalized_url(r['metadata']['source_url']) for r in rows if r.get('metadata',{}).get('source_url')}
    prior_path=config.parent_dir/'selection_manifest.json';prior=read_json(prior_path) if prior_path.is_file() else {}
    rejected={**prior.get('inherited_terminal_rejections',{}),**{r['source_video_id']:r.get('rejection_reason') for r in prior.get('candidates',[]) if r.get('status')=='rejected'}}
    ordered=sorted(pool,key=lambda r:(r['source_video_id'],r['source_url']));seen_ids=set(ids);seen_urls=set(urls);selected=[];excluded=[]
    for r in ordered:
        try:
            metadata_preflight(r,set());u=normalized_url(r['source_url'])
            reason='inherited_terminal_rejection' if r['source_video_id'] in rejected else 'duplicate_source_id' if r['source_video_id'] in seen_ids else 'duplicate_source_url' if u in seen_urls else 'duplicate_content' if r.get('sha256') in hashes else None
            if reason:excluded.append({'source_video_id':r['source_video_id'],'reason':reason});continue
            seen_ids.add(r['source_video_id']);seen_urls.add(u);selected.append(dict(r))
        except ValueError as e:excluded.append({'source_video_id':r.get('source_video_id'),'reason':str(e)})
    random.Random(config.seed).shuffle(selected)
    return {'version':config.version,'target_videos':config.target_videos,'parent_videos':len(ids),'new_accepted_required':config.target_videos-len(ids),
        'seed':config.seed,'candidates':selected,'excluded':excluded,'download_started':False,'acceptance_count':0,
        'status':'DRY_RUN_METADATA_ONLY','warning':'Candidate URLs/raw files are not accepted or human-verified data'}

class BuilderSession:
    def __init__(self,config,pool):
        self.config=config;self.parent,self.base,self.protected=parent_snapshot(config)
        self.plan=candidate_plan(config,pool);self.path=config.work/'events.jsonl'
        identity={'version':config.version,'target':config.target_videos,'seed':config.seed,'parent':self.parent['dataset_sha256'],'policy':config.split_policy}
        immutable_write(config.work/'session_identity.json',json_bytes(identity))
        immutable_write(config.work/'source_selection.json',json_bytes(self.plan))
        self.sources={r['source_video_id']:r for r in self.plan['candidates']};self.states={vid:'unattempted' for vid in self.sources}
        self.prepared={};self.attempts=Counter();self.accepted={};self.accepted_digests={}
        self.parent_hashes={x['sha256'] for x in self.parent['source_video_sha256'].values()}
        self.parent_ids={r['vid'].removeprefix('lumae_ads_') for r in self.base}
        if self.path.is_file():
            for e in load_jsonl(self.path):
                if e['seq']!=getattr(self,'seq',0)+1:raise ValueError('Invalid append-only ledger sequence')
                self.seq=e['seq'];self._apply(e)
        else:self.seq=0
        for p in sorted((config.work/'accepted').glob('*.json')):
            r=read_json(p);self.accepted[r['source_video_id']]=r
            if self.states[r['source_video_id']]=='accepted':
                if digest(p)!=self.accepted_digests[r['source_video_id']]:raise ValueError('Accepted ledger checksum changed')
            else:
                validate_annotation(r,visual_cache=config.visual_cache,root=config.data_root)
                self._event(r['source_video_id'],'accepted',record_sha256=digest(p),recovered_atomic_acceptance=True)
        if {v for v,s in self.states.items() if s=='accepted'}!=set(self.accepted):raise ValueError('Missing accepted record')

    def guard(self):
        for p,h in self.protected.items():
            if digest(Path(p))!=h:raise RuntimeError('Frozen parent changed: '+p)

    def _apply(self,e):
        vid=e['source_video_id'];self.states[vid]=e['state']
        if e.get('prepared'):self.prepared[vid]=e['prepared']
        if e['state']=='download_deferred':self.attempts[vid]+=1
        if e['state']=='accepted':self.accepted_digests[vid]=e['record_sha256']

    def _event(self,vid,state,**details):
        validate_transition(self.states[vid],state,vid)
        self.seq+=1;e={'seq':self.seq,'source_video_id':vid,'state':state,**details}
        self.path.parent.mkdir(parents=True,exist_ok=True)
        with self.path.open('a',encoding='utf-8',newline='\n') as f:f.write(json.dumps(e,sort_keys=True,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
        self._apply(e)

    def prepare_existing(self,vid,path,*,probe=probe_video_file):
        self.guard()
        if vid not in self.sources:raise ValueError('Source not in pinned candidate selection')
        path=Path(path).resolve()
        if not path.is_relative_to(self.config.data_root.resolve()):raise ValueError('New assets must be inside configured data root; no implicit migration')
        if not path.is_file() or path.stat().st_size==0:raise ValueError('Missing/empty MP4')
        info=probe(path)
        if not info.get('is_model_compatible') or not 0<float(info['duration'])<=self.config.max_duration:raise ValueError('Corrupt/incompatible MP4')
        h=digest(path)
        if h in self.parent_hashes or any(r['source_video_sha256']==h for k,r in self.accepted.items() if k!=vid):raise ValueError('Duplicate exact content hash')
        prepared={'path':path.relative_to(self.config.data_root.resolve()).as_posix(),'sha256':h,'duration':float(info['duration']),'bytes':path.stat().st_size}
        if vid in self.prepared:
            if self.prepared[vid]!=prepared:raise ValueError('Prepared source changed')
            return prepared
        if self.states[vid] in ('accepted','rejected'):raise ValueError('Terminal source cannot be replaced')
        if self.states[vid]!='downloading':self._event(vid,'downloading')
        self._event(vid,'downloaded',prepared=prepared)
        return prepared

    def fetch(self,vid,*,allow_network=False,downloader=download,probe=probe_video_file):
        if not allow_network:raise PermissionError('Actual crawling requires a separate explicit authorization')
        if len(self.accepted)>=self.config.target_videos-len(self.parent_ids):raise ValueError('Acceptance target already reached')
        if self.states[vid] in ('accepted','rejected'):raise ValueError('Terminal candidate')
        if self.attempts[vid]>=self.config.max_download_retries:raise ValueError('Retry budget exhausted')
        if self.states[vid]!='downloading':self._event(vid,'downloading')
        try:
            path=downloader(self.sources[vid],video_dir=self.config.videos)
            return self.prepare_existing(vid,path,probe=probe)
        except Exception as e:
            kind=classify_failure(e)
            state='download_deferred' if kind in (FailureKind.RETRYABLE_IO_ERROR,FailureKind.RETRYABLE_NETWORK_ERROR) else 'rejected'
            self._event(vid,state,failure_kind=kind.value,reason=str(e),retry_after_seconds=min(300,2**(self.attempts[vid]+1)*5))
            raise

    def accept(self,record,*,annotation_validator=validate_annotation):
        self.guard();vid=record['source_video_id']
        if vid not in self.sources or vid in self.parent_ids:raise ValueError('Duplicate/unselected source')
        if vid in self.accepted:
            if json_bytes(self.accepted[vid])!=json_bytes(record):raise ValueError('Immutable accepted annotation changed')
            return
        if len(self.accepted)>=self.config.target_videos-len(self.parent_ids):raise ValueError('Target already reached')
        if self.states[vid] not in ('downloaded','visual_index_ready','annotation_candidate_ready') or vid not in self.prepared:raise ValueError('Source must pass video preparation first')
        p=self.prepared[vid]
        if record['source_video_sha256']!=p['sha256'] or record['source_video_path']!=p['path'] or float(record['duration'])!=p['duration']:raise ValueError('Annotation/source identity mismatch')
        if normalized_url(record['source_url'])!=normalized_url(self.sources[vid]['source_url']):raise ValueError('Wrong annotation URL')
        if not validate_windows([record['window']],record['duration'])[0]:raise ValueError('Invalid timestamps')
        if record['source_video_sha256'] in self.parent_hashes|{r['source_video_sha256'] for r in self.accepted.values()}:raise ValueError('Duplicate source hash')
        for key in ('source_video_path','semantic_artifact_path','verification_image_path'):
            if key in record:safe_under(self.config.data_root,record[key])
        annotation_validator(record,visual_cache=self.config.visual_cache,root=self.config.data_root)
        if self.states[vid]=='downloaded':self._event(vid,'visual_index_ready')
        if self.states[vid]=='visual_index_ready':self._event(vid,'annotation_candidate_ready')
        immutable_write(self.config.work/'accepted'/(vid+'.json'),json_bytes(record))
        self.accepted[vid]=record;self._event(vid,'accepted',record_sha256=digest(self.config.work/'accepted'/(vid+'.json')))

    def reject(self,vid,reason):
        if not reason.strip():raise ValueError('Rejection reason required')
        if self.states[vid]=='rejected':return
        if self.states[vid]=='unattempted':self._event(vid,'downloading')
        self._event(vid,'rejected',reason=reason)

    def freeze(self,assignments=None,*,split_plan=None,annotation_validator=validate_annotation):
        self.guard();cfg=self.config
        if len(self.accepted)+len(self.parent_ids)!=cfg.target_videos:raise ValueError('Insufficient approved accepted videos')
        # Re-validate sources, immutable acceptance bytes, and review evidence at freeze.
        for vid,r in self.accepted.items():
            p=safe_under(cfg.data_root,r['source_video_path'])
            accepted_path=cfg.work/'accepted'/(vid+'.json')
            if digest(p)!=r['source_video_sha256'] or read_json(accepted_path)!=r or digest(accepted_path)!=self.accepted_digests.get(vid):raise ValueError('Accepted source/record changed')
            annotation_validator(r,visual_cache=cfg.visual_cache,root=cfg.data_root)
        if len(self.parent_hashes|{r['source_video_sha256'] for r in self.accepted.values()})!=cfg.target_videos:raise ValueError('Duplicate source hash at freeze')
        new=[canonical_row(r,f'{cfg.version}_{vid}') for vid,r in sorted(self.accepted.items())]
        all_rows=self.base+new
        if len({r['qid'] for r in all_rows})!=len(all_rows) or len({r['vid'] for r in all_rows})!=cfg.target_videos:raise ValueError('Duplicate dataset identity')
        if assignments is None:
            if cfg.split_policy!='historical_extension':raise ValueError('Campaign-reviewed explicit assignments required')
            assignments={r['vid']:s for s in ('train','val','test') for r in load_jsonl(cfg.parent_dir/f'{cfg.parent_version}_{s}.jsonl')}
            assignments.update({r['vid']:'train' for r in new})
        if set(assignments)!={r['vid'] for r in all_rows} or any(s not in ('train','val','test') for s in assignments.values()):raise ValueError('Invalid split coverage')
        splits={s:[r for r in all_rows if assignments[r['vid']]==s] for s in ('train','val','test')}
        if cfg.split_policy=='explicit_campaign_assignments':
            from ..splitting.campaign import validate_splits,rows_for_plan
            if split_plan is None or split_plan['assignments']!=assignments:raise ValueError('Reviewed campaign split plan required')
            if any(not split_plan['grouping']['registry'][r['vid']].get('historically_exposed',True) for r in self.base):raise ValueError('Historical exposure cannot be erased')
            splits=rows_for_plan(all_rows,split_plan)
            legacy=[r['qid'] for r in self.base if r['metadata']['review_provenance']=='HUMAN_VERIFIED']
            validate_splits(splits,grouping=split_plan['grouping'],require_human_test=False,min_test_queries=0,legacy_review_qids=legacy)
            enriched={r['qid']:r for values in splits.values() for r in values}
            all_rows=[enriched[r['qid']] for r in all_rows]
        payloads={f'{cfg.version}_all.jsonl':_jsonl_bytes(all_rows)}
        payloads.update({f'{cfg.version}_{s}.jsonl':_jsonl_bytes(v) for s,v in splits.items()})
        features=[{'vid':r['vid'],'qid':r['qid'],'video_feature':f'features/video/{r["vid"]}.npz','text_feature':f'features/text/{r["qid"]}.npz','ready':False,
            'required_action':'BIND_EXISTING_CACHE_NO_REENCODE' if r['vid'] in {'lumae_ads_'+v for v in self.parent_ids} else 'EXTRACT_NEW_ONCE',
            'extractor_commit':'b7e553ac3b0c898ee6b85e03ee507c064eab89ca','video_dim':512,'text_dim':512,'tef_in_cache':False} for r in all_rows]
        payloads['feature_manifest.json']=json_bytes({'dataset_version':cfg.version,'features':features,'status':'PLANNED_NOT_EXTRACTED'})
        training={'dataset_version':cfg.version,'splits':{s:f'{cfg.version}_{s}.jsonl' for s in ('train','val')},
            'queries':[r['qid'] for s in ('train','val') for r in splits[s]],'test_labels_included':False}
        training['features']=[f for f in features if f['qid'] in set(training['queries'])]
        payloads['training_asset_manifest.json']=json_bytes(training)
        payloads['storage_binding.json']=json_bytes({'roots':{'parent_assets':str((cfg.parent_asset_root or cfg.parent_dir.parent).resolve()),'new_assets':str(cfg.data_root.resolve())},
            'source_locations':{r['vid']:{'root':'parent_assets' if r['vid'] in {'lumae_ads_'+v for v in self.parent_ids} else 'new_assets',
                'path':r['metadata'].get('source_video_path','local_data/raw/adsqa/videos/'+r['metadata'].get('video_filename',r['vid']+'.mp4'))} for r in all_rows},
            'note':'Environment-specific binding; frozen historical path strings/IDs remain unchanged. No copy/move was performed.'})
        if split_plan is not None:payloads['campaign_grouping.json']=json_bytes(split_plan['grouping'])
        payloads['split_manifest.json']=json_bytes({'assignments':assignments,'seed':cfg.seed,'policy':cfg.split_policy})
        manifest={'dataset_version':cfg.version,'parent_version':cfg.parent_version,'parent_sha256':self.parent['dataset_sha256'],
            'target_videos':cfg.target_videos,'total_samples':len(all_rows),'counts':{s:len(v) for s,v in splits.items()},
            'dataset_sha256':hashlib.sha256(payloads[f'{cfg.version}_all.jsonl']).hexdigest(),
            'seed':cfg.seed,'semantic_v3_algorithm_sha256':ALGORITHM_SHA256,
            'source_video_sha256':{**self.parent['source_video_sha256'],**{r['qid']:{'sha256':r['metadata']['source_video_sha256'],'video_filename':r['metadata']['video_filename']} for r in new}},
            'provenance_counts':dict(Counter(r['metadata']['review_provenance'] for r in all_rows)),
            'validation_status':'DATASET_FROZEN_EVALUATION_REVIEW_PENDING'}
        payloads['dataset_manifest.json']=json_bytes(manifest)
        payloads['checksums.sha256']=''.join(f'{hashlib.sha256(data).hexdigest()}  {name}\n' for name,data in sorted(payloads.items())).encode()
        for name,data in payloads.items():immutable_write(cfg.dataset/name,data)
        immutable_write(cfg.dataset/'COMPLETE.json',json_bytes({'manifest_sha256':digest(cfg.dataset/'dataset_manifest.json')}))
        return manifest

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--repo-root',type=Path,default=Path.cwd());p.add_argument('--source-manifest',type=Path,required=True);p.add_argument('--dry-run',action='store_true',required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();cfg=BuildConfig.load(a.config,a.repo_root);pool=read_json(a.source_manifest)['videos'];result=candidate_plan(cfg,pool)
    with a.output.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2,ensure_ascii=False)
    print(json.dumps({k:v for k,v in result.items() if k not in ('candidates','excluded')}))

if __name__=='__main__':main()
