"""Bounded, restartable P1 acquisition. No network in planning or validation.

Source-video, annotation and dataset readiness are deliberately separate states.
No legacy downloader, dataset freezer or annotation algorithm is modified here.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import csv
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import time
from urllib.parse import urlsplit, urljoin

import requests

from .builder import BuildConfig, candidate_plan, parent_snapshot, safe_under
from .models import ROOT, atomic_json, read_json, digest
from .incremental import immutable_write, json_bytes
from .preparation import metadata_preflight
from .pilot_quality import validate_video, fingerprint_matches

AGENT='LumaeResearchPilot/1.0'
HOST='video.adsoftheworld.com'
TERMINAL={'REJECTED','QUARANTINED','SOURCE_ACCEPTED','ANNOTATION_ACCEPTED'}

class BudgetStop(RuntimeError):pass
class PermanentSource(ValueError):pass

def utc():return datetime.now(timezone.utc).isoformat()

def allowed_url(url):
    u=urlsplit(url)
    if u.scheme!='https' or u.hostname!=HOST or u.port not in (None,443) or u.username or u.password or u.fragment:
        raise PermanentSource('URL outside HTTPS source allowlist')
    return url

def validate_config(c):
    if not re.fullmatch(r'[A-Za-z0-9_-]+',c['run_id']):raise ValueError('Unsafe run ID')
    positive=('target_accepted','max_candidates','max_http_requests','max_download_bytes','max_file_bytes','max_runtime_seconds','max_staging_bytes','min_free_bytes','max_consecutive_failures','retries_per_source','timeout_seconds','request_interval_seconds')
    if any(not isinstance(c.get(k),(int,float)) or not math.isfinite(c[k]) or c[k]<=0 for k in positive):raise ValueError('Invalid resource limits')
    if c.get('concurrency')!=1:raise ValueError('Pilot supports conservative concurrency=1 only')
    if not 0<c['max_domain_error_rate']<=1:raise ValueError('Invalid error rate')
    if c['mode'] not in ('pilot','scale'):raise ValueError('Unknown mode')
    if c['mode']=='pilot' and (not 20<=c['target_accepted']<=30 or c['max_candidates']>60):raise ValueError('Pilot outside reviewed bounds')
    root=Path(c['data_root']).resolve()
    if not root.is_dir():raise ValueError('Configured data root must already exist')
    if root.is_relative_to(ROOT) or ROOT.is_relative_to(root):raise ValueError('Acquisition root overlaps research repository')
    if c.get('network_execution_authorized') is not True:raise PermissionError('Config network_execution_authorized must be true for live mode')
    permission=c.get('source_permission',{})
    if permission.get('status')!='OPERATOR_ATTESTED' or not permission.get('evidence') or permission.get('host')!=HOST:
        raise PermissionError('Explicit scoped source-permission evidence required')
    return root

def storage_check(c,root):
    free=shutil.disk_usage(root).free
    if free-c['max_staging_bytes']-c.get('processing_reserve_bytes',0)<c['min_free_bytes']:
        raise BudgetStop('Projected processing/staging violates free-space reserve')
    return free

def validate_scale_gate(c):
    """P1.6 supersedes only the count gate, never evidence/integrity gates."""
    binding=c.get('engineering_gate')
    if not binding:
        if not c.get('pilot_pass_evidence') or not read_json(Path(c['pilot_pass_evidence'])).get('pilot_pass'):
            raise PermissionError('Scale requires passing pilot evidence')
        return
    path=Path(binding['path'])
    if digest(path)!=binding['sha256']:raise PermissionError('Engineering gate identity changed')
    gate=read_json(path)
    if gate.get('schema')!='LUMAE_P16_ENGINEERING_V1' or gate.get('engineering_pass') is not True or gate.get('critical_failure') is not False:
        raise PermissionError('P1.6 engineering gate failed')
    if (gate.get('reconciled_candidates'),gate.get('prior_annotations_reaudited'),gate.get('failed_sources_recovery_attempted'))!=(30,16,13):
        raise PermissionError('Incomplete P1.6 audit')
    freeze=Path(gate['freeze_path'])
    if digest(freeze)!=gate['freeze_sha256']:raise PermissionError('Recovery freeze changed')
    manifest=read_json(freeze)
    for name,h in manifest.get('artifact_sha256',{}).items():
        if digest(freeze.parent/name)!=h:raise PermissionError('Recovery artifact changed')
    for field in ('evidence_sha256','processing_code_sha256'):
        for name,h in manifest.get(field,{}).items():
            if digest(Path(name))!=h:raise PermissionError('Recovery evidence/code changed')
    if not manifest.get('artifact_sha256') or manifest.get('final_d1000_freeze') is not False:
        raise PermissionError('Invalid pilot freeze scope')

@contextmanager
def run_lock(path):
    """OS-released lock; a process crash does not leave a stale exclusive flag."""
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a+b') as f:
        if f.tell()==0:f.write(b'0');f.flush()
        f.seek(0)
        if os.name=='nt':
            import msvcrt
            msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:yield
        finally:
            f.seek(0)
            if os.name=='nt':msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)
            else:fcntl.flock(f.fileno(),fcntl.LOCK_UN)

class Pilot:
    def __init__(self,c,build,pool,*,live=False,session=None,validator=validate_video):
        self.c=c;self.build=build;self.pool=pool;self.live=live;self.validator=validator
        self.plan=candidate_plan(build,pool);self.parent,self.parent_rows,_=parent_snapshot(build)
        self.root=Path(c['data_root']).resolve();self.work=self.root/'work/d1000'/('pilot_'+c['run_id'])
        self.session=session
        self.started=time.monotonic();self.last_request=0
        self.parent_hashes={r['sha256'] for r in self.parent['source_video_sha256'].values()}
        self.parent_files={p.name:digest(p) for p in build.parent_dir.iterdir() if p.is_file()}
        self.baseline=read_json(Path(c['baseline_fingerprints'])) if c.get('baseline_fingerprints') else {'videos':[],'complete':False}
        if self.baseline.get('complete'):
            baseline_hashes={v.get('sha256') for v in self.baseline['videos']}
            if self.baseline.get('dataset_sha256')!=self.parent['dataset_sha256'] or baseline_hashes!=self.parent_hashes or len(self.baseline['videos'])!=len(self.parent_hashes):
                raise ValueError('Complete fingerprint baseline does not bind the full frozen parent')
        self.previous=[]
        for binding in c.get('previous_runs',[]):
            previous=safe_under(self.root,binding['manifest'])
            if digest(previous)!=binding['sha256']:raise ValueError('Prior pilot manifest changed')
            prior=read_json(previous)
            if prior['parent_dataset_sha256']!=self.parent['dataset_sha256']:raise ValueError('Prior run parent mismatch')
            for r in prior['records']:
                if r.get('sha256') and digest(safe_under(self.root,r['path']))!=r['sha256']:raise ValueError('Prior pilot asset changed')
                self.previous.append(r)
        previous_ids={r['source_video_id'] for r in self.previous}
        if self.previous:
            self.plan['candidates']=[r for r in self.plan['candidates'] if r['source_video_id'] not in previous_ids]
            self.plan['previous_run_source_ids']=sorted(previous_ids)
            self.plan['previous_accepted_sources']=sum(r['status'] in ('SOURCE_ACCEPTED','ANNOTATION_ACCEPTED') for r in self.previous)
        self.parent_hashes.update(r['sha256'] for r in self.previous if r.get('sha256'))

    def guard(self):
        for name,h in self.parent_files.items():
            if digest(self.build.parent_dir/name)!=h:raise RuntimeError('Frozen parent changed')

    def initialize(self):
        if not self.live:raise PermissionError('Dry run cannot initialize live storage or HTTP')
        root=validate_config(self.c)
        if self.c['mode']=='scale':validate_scale_gate(self.c)
        storage_check(self.c,root);self.guard()
        gate=read_json(Path(self.c['operational_gate']))
        if not gate.get('protected_assets_pass') or not gate.get('storage_pass') or not gate.get('source_permissions_attested') or not gate.get('crawler_tests_pass'):
            raise PermissionError('Operational preflight gate has not passed')
        if gate.get('source_manifest_sha256')!=self.c['source_manifest_sha256']:raise ValueError('Gate source identity mismatch')
        self.work.mkdir(parents=True,exist_ok=True)
        immutable_write(self.work/'run_config.json',json_bytes(self.c))
        immutable_write(self.work/'source_selection.json',json_bytes(self.plan))
        immutable_write(self.work/'parent_binding.json',json_bytes({'parent_files':self.parent_files,'dataset_sha256':self.parent['dataset_sha256'],'source_manifest_sha256':self.c['source_manifest_sha256']}))
        for d in ('downloads','staging','logs/attempts','logs/events','manifests','annotations','reviews','reports','records'):(self.work/d).mkdir(parents=True,exist_ok=True)
        clock=self.work/'budget_clock.json'
        if not clock.exists():immutable_write(clock,json_bytes({'started_epoch':time.time()}))
        self.budget_started=read_json(clock)['started_epoch']

    def records(self):return [read_json(p) for p in sorted((self.work/'records').glob('*.json'))]
    def attempts(self):return [read_json(p) for p in sorted((self.work/'logs/attempts').glob('*.json'))]

    def event(self,kind,**fields):
        directory=self.work/'logs/events'
        seq=len(list(directory.glob('*.json')))+1
        immutable_write(directory/f'{seq:08d}.json',json_bytes({'seq':seq,'utc':utc(),'event':kind,**fields}))

    def checkpoint(self,record):
        atomic_json(self.work/'records'/(record['source_video_id']+'.json'),record)

    def check_limits(self,additional=0):
        attempts=self.attempts()
        transferred=sum(a['bytes_received'] for a in attempts)
        if transferred+additional>self.c['max_download_bytes']:raise BudgetStop('Cumulative network byte limit')
        elapsed=sum(a.get('elapsed_seconds',0) for a in attempts)
        if max(time.time()-getattr(self,'budget_started',time.time()),time.monotonic()-self.started,elapsed)>self.c['max_runtime_seconds']:raise BudgetStop('Runtime limit')
        retained=sum(p.stat().st_size for folder in (self.work,self.root/'cache'/('p1_'+self.c['run_id'])) for p in folder.rglob('*') if p.is_file())
        if retained+additional>self.c['max_staging_bytes']:raise BudgetStop('Staging byte limit')
        if shutil.disk_usage(self.root).free-additional-self.c.get('processing_reserve_bytes',0)<self.c['min_free_bytes']:raise BudgetStop('Free-space reserve')
        state=self.work/'storage_high_water.json'
        previous=read_json(state).get('observed_bytes',0) if state.is_file() else 0
        if retained+additional>previous:atomic_json(state,{'observed_bytes':retained+additional,'method':'Logical work/cache bytes observed at chunk/limit checkpoints; includes retained downloads, excludes unobserved atomic-write transients'})
        return transferred,retained

    def request(self,url,headers,attempt,attempt_path):
        for _ in range(4):
            allowed_url(url);self.check_limits()
            # Count every redirect request; do not silently spend outside global budget.
            count=sum(a.get('http_requests',0) for a in self.attempts())
            if count>=self.c['max_http_requests']:raise BudgetStop('HTTP request limit')
            wait=max(0,self.c['request_interval_seconds']-(time.monotonic()-self.last_request))
            if wait:time.sleep(wait)
            attempt['http_requests']+=1;atomic_json(attempt_path,attempt)
            self.last_request=time.monotonic()
            response=self.session.get(url,headers=headers,stream=True,timeout=(5,self.c['timeout_seconds']),allow_redirects=False)
            attempt['http_status']=response.status_code;atomic_json(attempt_path,attempt)
            if response.status_code in (301,302,303,307,308):
                target=urljoin(url,response.headers.get('Location',''));response.close();allowed_url(target);url=target;continue
            return response
        raise PermanentSource('Redirect limit')

    def fetch(self,row):
        if not self.live:raise PermissionError('Network disabled in dry run')
        metadata_preflight(row,set());allowed_url(row['source_url'])
        sid=row['source_video_id'];final=self.work/'downloads'/row['target_name'];part=final.with_suffix('.mp4.part')
        # Completed file, including one atomically promoted before a crash: validate locally.
        if final.is_file():return final
        offset=part.stat().st_size if part.is_file() else 0
        if offset>self.c['max_file_bytes']:raise BudgetStop('Existing partial exceeds per-file limit')
        aid=f'{len(self.attempts())+1:06d}_{sid}'
        attempt={'attempt_id':aid,'source_video_id':sid,'source_url':row['source_url'],'started_utc':utc(),'initial_offset':offset,'bytes_received':0,'http_requests':0,'status':'STARTED','elapsed_seconds':0}
        ap=self.work/'logs/attempts'/(aid+'.json');atomic_json(ap,attempt);start=time.monotonic()
        try:
            headers={'User-Agent':AGENT,'Accept-Encoding':'identity'}
            resume_meta=part.with_suffix('.identity.json')
            identity=read_json(resume_meta) if resume_meta.is_file() else {}
            if offset:
                # Appending without a stable HTTP validator can splice two versions.
                etag=identity.get('etag')
                validator=(etag if etag and not etag.startswith('W/') else None) or identity.get('last_modified')
                if not validator:offset=0
                else:headers.update({'Range':f'bytes={offset}-','If-Range':validator})
            response=self.request(row['source_url'],headers,attempt,ap)
            with response:
                status=response.status_code
                if status in (401,403,404,410):raise PermanentSource(f'HTTP_{status}; no access bypass or alternate host')
                if status==429:
                    attempt['retry_after_seconds']=min(300,max(1,int(response.headers.get('Retry-After','60'))) if response.headers.get('Retry-After','60').isdigit() else 60)
                    raise requests.ConnectionError('HTTP_429')
                if status>=500:raise requests.ConnectionError(f'HTTP_{status}')
                if status not in (200,206):raise PermanentSource(f'Unexpected HTTP_{status}')
                if status==200:offset=0
                if status==206:
                    m=re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)',response.headers.get('Content-Range',''))
                    if not m or int(m[1])!=offset or int(m[2])<offset or int(m[3])<=int(m[2]):raise PermanentSource('Invalid Content-Range')
                    expected=int(m[2])-offset+1;total=int(m[3])
                    if int(m[2])+1!=total:raise PermanentSource('Incomplete range response')
                else:
                    length=response.headers.get('Content-Length')
                    if length is not None and not length.isdigit():raise PermanentSource('Invalid Content-Length')
                    expected=int(length) if length is not None else None
                    total=expected
                if total and total>self.c['max_file_bytes']:raise PermanentSource('Advertised size exceeds per-file budget')
                content_type=response.headers.get('Content-Type','').lower()
                if 'text/' in content_type or 'json' in content_type or 'xml' in content_type:raise PermanentSource('Non-video content type')
                new_identity={'etag':response.headers.get('ETag'),'last_modified':response.headers.get('Last-Modified')}
                if status==206 and identity and any(identity.get(k) and new_identity.get(k)!=identity[k] for k in ('etag','last_modified')):raise PermanentSource('Resume HTTP validator changed')
                atomic_json(resume_meta,new_identity)
                attempt['response_headers']={k:response.headers[k] for k in ('Content-Length','Content-Range','Content-Type','ETag','Last-Modified') if k in response.headers}
                atomic_json(ap,attempt)
                received=0
                with part.open('ab' if offset else 'wb') as stream:
                    for chunk in response.iter_content(256*1024):
                        if not chunk:continue
                        self.check_limits(len(chunk))
                        if offset+received+len(chunk)>self.c['max_file_bytes']:raise PermanentSource('Stream exceeds per-file budget')
                        # Count before writing: a crash can conservatively overcount, never undercount budget.
                        attempt['bytes_received']+=len(chunk);attempt['elapsed_seconds']=time.monotonic()-start;atomic_json(ap,attempt)
                        stream.write(chunk);stream.flush();os.fsync(stream.fileno());received+=len(chunk)
                if expected is not None and received!=expected:raise requests.ConnectionError('Truncated body')
                if not part.stat().st_size:raise PermanentSource('Empty body')
                os.replace(part,final)
                attempt['status']='DOWNLOADED';attempt['final_bytes']=final.stat().st_size
            return final
        except BaseException as e:
            attempt['status']='INTERRUPTED' if isinstance(e,(KeyboardInterrupt,SystemExit)) else 'FAILED'
            attempt['error_type']=type(e).__name__
            # No request object, arbitrary exception strings, credentials or signed redirects in logs.
            attempt['reason']=str(e)[:180] if isinstance(e,(PermanentSource,BudgetStop)) else type(e).__name__
            raise
        finally:
            attempt['elapsed_seconds']=time.monotonic()-start;atomic_json(ap,attempt)

    def process(self,row,path):
        sid=row['source_video_id'];result=self.validator(path)
        record={'source_video_id':sid,'source_url':row['source_url'],'selection_reason':'SEEDED_ELIGIBLE_POOL_EXCLUDES_PARENT_AND_TERMINAL_REJECTIONS','selection_seed':self.build.seed,
                'path':path.relative_to(self.root).as_posix(),'bytes':path.stat().st_size,'sha256':digest(path),'technical':result,
                'annotation_status':'PENDING_VISUAL_QUERY_AND_SEMANTIC_VERIFICATION','human_review_status':'PENDING','dataset_freeze_ready':False,'observed_utc':utc()}
        if not result['pass']:record.update(status='REJECTED',reason=result['reason'])
        elif record['sha256'] in self.parent_hashes or any(r.get('sha256')==record['sha256'] and r['source_video_id']!=sid for r in self.records()):record.update(status='REJECTED',reason='EXACT_CONTENT_DUPLICATE')
        else:
            references=self.baseline['videos']+[r for r in self.previous+self.records() if r['source_video_id']!=sid and r.get('technical',{}).get('pass')]
            matches=fingerprint_matches(result['fingerprint'],references)
            # Missing brand/campaign evidence must not be silently called independent.
            record.update(status='QUARANTINED',reason='NEAR_DUPLICATE_REVIEW' if matches else 'CAMPAIGN_REVIEW_REQUIRED',duplicate_candidates=matches,
                          campaign_status='UNREVIEWED',baseline_comparison_complete=self.baseline.get('complete',False))
        self.checkpoint(record);self.event('VIDEO_VALIDATED',source_video_id=sid,status=record['status'],sha256=record['sha256'])
        return record

    def review_source(self,sid,evidence):
        """Promote only a real reviewed source; does not accept its temporal label."""
        self.guard();record=read_json(self.work/'records'/(sid+'.json'))
        if record['status'] not in ('QUARANTINED','SOURCE_ACCEPTED'):raise ValueError('Source not eligible for review')
        required=('reviewer_id','reviewed_at','evidence_paths','campaign_id','decision','comparison_scope')
        if any(not evidence.get(k) for k in required):raise ValueError('Incomplete real source review evidence')
        if evidence['decision'] not in ('DISTINCT','DUPLICATE','UNSURE','RETAIN_SPLIT_RESTRICTED'):raise ValueError('Unknown decision')
        if evidence['comparison_scope']!='D500_AND_PILOT':raise ValueError('Must compare parent and pilot campaigns')
        # Evidence is an operator/agent assertion, not an invented automated identity.
        hashes={}
        for rel in evidence['evidence_paths']:
            p=safe_under(self.work,rel)
            if not p.is_file():raise ValueError('Missing review evidence')
            hashes[rel]=digest(p)
        if evidence['decision'] in ('DISTINCT','RETAIN_SPLIT_RESTRICTED'):
            if not self.baseline.get('complete'):raise ValueError('Complete baseline fingerprint index required before source acceptance')
            source_path=safe_under(self.root,record['path'])
            if digest(source_path)!=record['sha256'] or record['sha256'] in self.parent_hashes:raise ValueError('Changed/duplicate reviewed source')
            if not self.validator(source_path)['pass']:raise ValueError('Source failed independent revalidation')
            if evidence['decision']=='RETAIN_SPLIT_RESTRICTED':
                if not self.c.get('engineering_gate'):raise ValueError('Restricted retention requires P1.6 engineering gate')
                validate_scale_gate(self.c)
                if evidence.get('relationship') not in ('UNKNOWN','SAME_CAMPAIGN_DISTINCT_VIDEO') or evidence.get('fresh_test_eligible') is not False:
                    raise ValueError('Restricted source must preserve uncertain/shared campaign and Test exclusion')
                record.update(split_eligibility='RESTRICTED_PENDING_CAMPAIGN_GROUP_REVIEW',fresh_test_eligible=False,campaign_group_id=evidence['campaign_id'])
            record.update(status='SOURCE_ACCEPTED',campaign_status='REVIEWED_DISTINCT' if evidence['decision']=='DISTINCT' else 'RETAINED_SPLIT_RESTRICTED',campaign_review_evidence={**evidence,'evidence_sha256':hashes})
        elif evidence['decision']=='DUPLICATE':record.update(status='REJECTED',reason='REVIEWED_DUPLICATE')
        immutable_write(self.work/'reviews'/(sid+'_campaign.json'),json_bytes({**evidence,'evidence_sha256':hashes}))
        self.checkpoint(record);self.event('SOURCE_REVIEW',source_video_id=sid,decision=evidence['decision'])

    def accept_annotation(self,record):
        """Reuse the P0 frozen evidence validator and independent review store."""
        from dataclasses import replace
        from .builder import BuilderSession
        from ..annotation.independent_review import ReviewStore
        sid=record['source_video_id'];source=read_json(self.work/'records'/(sid+'.json'))
        if source['status'] not in ('SOURCE_ACCEPTED','ANNOTATION_ACCEPTED'):raise ValueError('Campaign/source review must pass first')
        if record['source_video_sha256']!=source['sha256'] or record['source_video_path']!=source['path']:raise ValueError('Wrong source binding')
        config=replace(self.build,version='p1_'+self.c['run_id'],data_root=self.root)
        builder=BuilderSession(config,self.pool);builder.prepare_existing(sid,safe_under(self.root,source['path']));builder.accept(record)
        qid=record['qid'];store=ReviewStore(self.work/'reviews','p1_'+self.c['run_id'])
        store.register(qid,'lumae_ads_'+sid,record['query'],record['duration'],record['window'])
        atomic_json(self.work/'reviews'/(sid+'_blind_packet.json'),store.blind_packet(qid))
        source.update(status='ANNOTATION_ACCEPTED',annotation_status='AI_PSEUDO_LABELED',human_review_status='PENDING',accepted_annotation_path=str(config.work/'accepted'/(sid+'.json')))
        self.checkpoint(source);self.event('ANNOTATION_ACCEPTED',source_video_id=sid,human_verified=False)

    def run(self,*,max_wave=None,resume_connection_failures=False):
        if not self.live:raise PermissionError('Use plan for a network-denied dry run')
        self.initialize()
        with run_lock(self.work/'logs/run.lock'):
            return self._run_locked(max_wave,resume_connection_failures)

    @staticmethod
    def trailing_failures(attempts):
        failures=[]
        for attempt in reversed(attempts):
            if attempt['status']!='FAILED':break
            failures.append(attempt)
        return failures

    def connection_recovery_cutoff(self,requested):
        marker=self.work/'logs/connection_recovery.json'
        attempts=self.attempts()
        if marker.exists():
            if requested:raise ValueError('Connection recovery already recorded for this batch')
            recovery=read_json(marker)
            cutoff=recovery['attempt_count']
            if not 0<cutoff<=len(attempts) or attempts[cutoff-1]['attempt_id']!=recovery['last_attempt_id']:
                raise ValueError('Connection recovery does not match the attempt journal')
            return cutoff
        if not requested:return 0
        failures=self.trailing_failures(attempts)
        if len(failures)!=self.c['max_consecutive_failures'] or any(
            a.get('reason')!='ConnectionError' or a.get('http_status') is not None
            or a.get('retry_after_seconds') or a['http_requests']!=1 for a in failures
        ):
            raise ValueError('Recovery requires a threshold stop with only pre-response connection failures')
        self.check_limits()
        if sum(a['http_requests'] for a in attempts)>=self.c['max_http_requests']:
            raise ValueError('Connection recovery requires remaining HTTP budget')
        cutoff=len(attempts)
        immutable_write(marker,json_bytes({'utc':utc(),'attempt_count':cutoff,'last_attempt_id':attempts[-1]['attempt_id'],
                                          'reason':'OPERATOR_AUTHORIZED_CONNECTION_RECOVERY'}))
        return cutoff

    def _run_locked(self,max_wave,resume_connection_failures=False):
        cutoff=self.connection_recovery_cutoff(resume_connection_failures)
        self.session=self.session or requests.Session()
        self.event('RUN_STARTED',network_execution_authorized=True)
        stop='CANDIDATE_BUDGET';wave=0
        try:
            for row in self.plan['candidates'][:self.c['max_candidates']]:
                records={r['source_video_id']:r for r in self.records()};sid=row['source_video_id']
                if sid in records and records[sid]['status'] in TERMINAL:
                    if records[sid].get('sha256'):
                        p=safe_under(self.root,records[sid]['path'])
                        if not p.is_file() or digest(p)!=records[sid]['sha256']:raise RuntimeError('Completed source changed during resume')
                    continue
                if self.plan.get('previous_accepted_sources',0)+sum(r['status'] in ('SOURCE_ACCEPTED','ANNOTATION_ACCEPTED') for r in records.values())>=self.c['target_accepted']:stop='ACCEPTED_TARGET';break
                if max_wave is not None and wave>=max_wave:stop='BOUNDED_WAVE';break
                attempts=self.attempts();recent=[a for a in attempts if a['status'] in ('FAILED','DOWNLOADED')]
                current=[a for a in attempts[cutoff:] if a['status'] in ('FAILED','DOWNLOADED')]
                if len(self.trailing_failures(current))>=self.c['max_consecutive_failures']:raise BudgetStop('Consecutive failed request threshold')
                if len(recent)>=self.c.get('domain_error_min_requests',5) and sum(a['status']=='FAILED' for a in recent)/len(recent)>self.c['max_domain_error_rate']:raise BudgetStop('Domain error rate threshold')
                self.check_limits();wave+=1
                prior=[a for a in attempts if a['source_video_id']==sid]
                remaining=self.c['retries_per_source']-len(prior)
                if remaining<=0:
                    self.checkpoint({'source_video_id':sid,'source_url':row['source_url'],'status':'REJECTED','reason':'RETRY_BUDGET_EXHAUSTED','annotation_status':'NOT_STARTED','human_review_status':'NOT_STARTED','dataset_freeze_ready':False});continue
                for retry in range(remaining):
                    try:
                        path=self.fetch(row);self.process(row,path);break
                    except BudgetStop:raise
                    except PermanentSource as e:
                        self.checkpoint({'source_video_id':sid,'source_url':row['source_url'],'status':'REJECTED','reason':str(e),'annotation_status':'NOT_STARTED','human_review_status':'NOT_STARTED','dataset_freeze_ready':False});break
                    except (requests.RequestException,OSError) as e:
                        self.event('REQUEST_FAILURE',source_video_id=sid,error_type=type(e).__name__)
                        current=self.attempts()
                        if len(self.trailing_failures(current[cutoff:]))>=self.c['max_consecutive_failures']:raise BudgetStop('Consecutive failed request threshold')
                        delay=max(min(60,2**(len(prior)+retry)*5),current[-1].get('retry_after_seconds',0))
                        if retry+1<remaining:time.sleep(delay)
                else:
                    self.checkpoint({'source_video_id':sid,'source_url':row['source_url'],'status':'REJECTED','reason':'RETRY_BUDGET_EXHAUSTED','annotation_status':'NOT_STARTED','human_review_status':'NOT_STARTED','dataset_freeze_ready':False})
        except BudgetStop as e:stop=str(e)
        except KeyboardInterrupt:stop='OPERATOR_INTERRUPTED'
        except Exception as e:
            stop='INGESTION_ERROR_'+type(e).__name__;self.event('RUN_ERROR',error_type=type(e).__name__)
            raise
        finally:
            self.guard();self.event('RUN_STOPPED',reason=stop);result=self.export(stop)
        return result

    def export(self,stop='OFFLINE_REBUILD'):
        rows=self.records();attempts=self.attempts();valid=[r for r in rows if r.get('technical',{}).get('pass')];sizes=[r['bytes'] for r in valid]
        retained=sum(p.stat().st_size for folder in (self.work,self.root/'cache'/('p1_'+self.c['run_id'])) for p in folder.rglob('*') if p.is_file())
        observed=read_json(self.work/'storage_high_water.json').get('observed_bytes',0) if (self.work/'storage_high_water.json').is_file() else 0
        metrics={'stop_reason':stop,'attempted_candidates':len({a['source_video_id'] for a in attempts}),'http_requests':sum(a['http_requests'] for a in attempts),'download_attempts':len(attempts),
                 'successful_downloads':sum(a['status']=='DOWNLOADED' for a in attempts),'technically_valid_videos':len(valid),'accepted_unique_videos':sum(r['status'] in ('SOURCE_ACCEPTED','ANNOTATION_ACCEPTED') for r in rows),
                 'rejected_videos':sum(r['status']=='REJECTED' for r in rows),'quarantined_candidates':sum(r['status']=='QUARANTINED' for r in rows),'annotation_accepted':sum(r['status']=='ANNOTATION_ACCEPTED' for r in rows),
                 'review_pending':len(valid),'average_valid_file_bytes':statistics.mean(sizes) if sizes else None,'median_valid_file_bytes':statistics.median(sizes) if sizes else None,
                 'total_bytes_downloaded':sum(a['bytes_received'] for a in attempts),'total_bytes_retained':retained,'observed_storage_high_water_bytes':max(retained,observed),
                 'download_elapsed_seconds':sum(a['elapsed_seconds'] for a in attempts),'mean_seconds_per_accepted_video':None,'human_agreement':None,'annotation_acceptance_rate':None,
                 'current_free_bytes':shutil.disk_usage(self.root).free,'pilot_pass':False,'classification':'Observed records only; quarantined technically valid files are not accepted'}
        manifest={'run_id':self.c['run_id'],'records':rows,'metrics':metrics,'parent_dataset_sha256':self.parent['dataset_sha256'],'final_d1000_frozen':False}
        atomic_json(self.work/'manifests/pilot_download_manifest.json',manifest)
        atomic_json(self.work/'reports/pilot_storage_metrics.json',metrics)
        proposals=[{'path':str(p.relative_to(self.work)),'sha256':digest(p),'proposal_count':len(read_json(p)['proposals'])} for p in (self.work/'annotations').glob('*_proposals.json')]
        atomic_json(self.work/'annotations/pilot_annotation_ledger.json',{'records':[{'source_video_id':r['source_video_id'],'status':r['annotation_status'],'human_review_status':r['human_review_status']} for r in rows],'proposals':proposals,'human_verified':0})
        atomic_json(self.work/'reviews/pilot_review_queue.json',{'records':[{'source_video_id':r['source_video_id'],'path':r['path'],'required':['campaign independence review','actual visual event query','frozen Semantic V3 labeling','independent human review'],'status':'PENDING'} for r in valid]})
        source_ledger=[];retries={};by_id={r['source_video_id']:r for r in rows}
        for attempt in attempts:
            sid=attempt['source_video_id'];r=by_id.get(sid,{})
            source_ledger.append({**attempt,'retry_count':retries.get(sid,0),'final_local_path':r.get('path'),'sha256':r.get('sha256'),'video_status':r.get('status'),'probe_pass':r.get('technical',{}).get('pass'),'failure_or_rejection_reason':r.get('reason')})
            retries[sid]=retries.get(sid,0)+1
        for name,data in [('pilot_source_ledger.csv',source_ledger),('pilot_video_integrity.csv',[{k:r.get(k) for k in ('source_video_id','path','bytes','sha256','status','reason')} for r in rows]),('pilot_duplicate_candidates.csv',[{'source_video_id':r['source_video_id'],'candidate_source_video_id':m['source_video_id'],'distance':m['distance']} for r in rows for m in r.get('duplicate_candidates',[])])]:
            target=self.work/'reports'/name;fields=sorted({k for r in data for k in r}) or ['source_video_id','status']
            # Derived exports may be rebuilt; source records and event segments remain authoritative.
            temporary=target.with_suffix('.csv.tmp')
            with temporary.open('w',encoding='utf-8-sig',newline='') as f:
                w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(data)
            os.replace(temporary,target)
        events=[read_json(p) for p in sorted((self.work/'logs/events').glob('*.json'))]
        temp=self.work/'reports/pilot_execution_log.jsonl.tmp'
        temp.write_text(''.join(json.dumps(e,sort_keys=True)+'\n' for e in events),encoding='utf-8');os.replace(temp,self.work/'reports/pilot_execution_log.jsonl')
        hashes={str(p.relative_to(self.work)):digest(p) for directory in ('manifests','annotations','reviews','reports') for p in (self.work/directory).glob('*') if p.is_file() and p.name!='checksums.json'}
        atomic_json(self.work/'reports/checksums.json',hashes)
        return metrics

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('command',choices=['plan','run','export','validate','triage','review-source','accept-annotation']);ap.add_argument('--config',type=Path,required=True);ap.add_argument('--enable-live-network',action='store_true');ap.add_argument('--enable-scale-budget',action='store_true');ap.add_argument('--max-wave',type=int);ap.add_argument('--output',type=Path);ap.add_argument('--source-id');ap.add_argument('--evidence',type=Path)
    a=ap.parse_args();c=read_json(a.config);build=BuildConfig.load(ROOT/c['builder_config'],ROOT);pool=read_json(ROOT/c['source_manifest'])['videos']
    if digest(ROOT/c['source_manifest'])!=c['source_manifest_sha256']:raise ValueError('Source manifest changed')
    pilot=Pilot(c,build,pool,live=a.enable_live_network)
    if a.command=='plan':
        if a.enable_live_network:raise ValueError('Plan is network-denied')
        value={'candidate_count':len(pilot.plan['candidates']),'selected':pilot.plan['candidates'][:c['max_candidates']],'network_started':False,'data_root':c['data_root'],'limits':c,'acceptance_count':0}
        if a.output:immutable_write(a.output,json_bytes(value))
    elif a.command=='run':
        if c['mode']=='scale':
            if not a.enable_scale_budget:raise PermissionError('Scale requires --enable-scale-budget')
            validate_scale_gate(c)
        value=pilot.run(max_wave=a.max_wave)
    elif a.command=='validate':
        from .pilot_quality import validate_run
        value=validate_run(pilot.work,pilot.root,pilot.parent_hashes)
        if a.output:atomic_json(a.output,value)
    elif a.command=='triage':
        if a.enable_live_network:raise ValueError('Triage is network-denied')
        with run_lock(pilot.work/'logs/run.lock'):
            pilot.guard()
            sources={r['source_video_id']:r for r in pilot.plan['candidates']}
            for r in pilot.records():
                if r['status']=='QUARANTINED':pilot.process(sources[r['source_video_id']],safe_under(pilot.root,r['path']))
            value=pilot.export('OFFLINE_TRIAGE')
    elif a.command in ('review-source','accept-annotation'):
        if a.enable_live_network or not a.evidence:raise ValueError('Offline evidence import requires --evidence')
        with run_lock(pilot.work/'logs/run.lock'):
            if a.command=='review-source':pilot.review_source(a.source_id,read_json(a.evidence))
            else:pilot.accept_annotation(read_json(a.evidence))
            value=pilot.export()
    else:
        if a.enable_live_network:raise ValueError('Export is network-denied')
        value=pilot.export()
    print(json.dumps(value,ensure_ascii=True,indent=2))

if __name__=='__main__':main()
