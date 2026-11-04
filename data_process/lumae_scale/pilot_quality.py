"""Independent real-video checks and deliberately conservative duplicate triage."""
from __future__ import annotations
import math
from pathlib import Path
import subprocess
import numpy as np
from .models import read_json,digest
from .builder import safe_under
from ..adsqa.video_validation import probe_video_file

def frame_hash(path,timestamp):
    r=subprocess.run(['ffmpeg','-v','error','-ss',str(timestamp),'-i',str(path),'-frames:v','1','-vf','scale=9:8,format=gray','-f','rawvideo','pipe:1'],capture_output=True,timeout=30)
    if r.returncode or len(r.stdout)!=72:raise ValueError('Sample frame decoding failed')
    a=np.frombuffer(r.stdout,dtype=np.uint8).reshape(8,9)
    return f'{int("".join("1" if x else "0" for x in (a[:,1:]>a[:,:-1]).ravel()),2):016x}'

def validate_video(path):
    p=Path(path);info=probe_video_file(p);duration=float(info.get('duration',0))
    valid=info.get('is_model_compatible') and math.isfinite(duration) and 0<duration<=150 and info.get('width',0)>0 and info.get('height',0)>0 and info.get('codec')
    result={'pass':False,'probe':info,'reason':info.get('incompatibility_reason') or 'Invalid duration/codec/dimensions'}
    if not valid:return result
    try:
        times=[duration*x for x in (.15,.5,.85)]
        hashes=[frame_hash(p,t) for t in times]
        # Check decoder timestamps independently of ffprobe's container duration.
        import json
        r=subprocess.run(['ffprobe','-v','error','-select_streams','v:0','-read_intervals','%+3','-show_entries','frame=best_effort_timestamp_time:format=format_name','-of','json',str(p)],capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=30)
        decoded=json.loads(r.stdout);container=decoded.get('format',{}).get('format_name','')
        if not {'mov','mp4'}&set(container.split(',')):raise ValueError('Unexpected source container')
        stamps=[float(f['best_effort_timestamp_time']) for f in decoded.get('frames',[]) if 'best_effort_timestamp_time' in f]
        if r.returncode or not stamps or any(not math.isfinite(t) for t in stamps) or stamps!=sorted(stamps) or stamps[-1]>duration+1:raise ValueError('Inconsistent sample timestamps')
        result.update({'pass':True,'reason':None,'container_format':container,'fingerprint':{'algorithm':'three_frame_dhash64_v1','timestamps':times,'hashes':hashes,'duration':duration},'sample_frames_decoded':3,'initial_timestamps_consistent':True})
    except (ValueError,KeyError,subprocess.SubprocessError) as e:result['reason']=str(e)[:200]
    return result

def fingerprint_matches(fingerprint,references,threshold=4):
    matches=[]
    for r in references:
        f=r.get('fingerprint') or r.get('technical',{}).get('fingerprint')
        if not f or f.get('algorithm')!=fingerprint['algorithm']:continue
        distances=[(int(a,16)^int(b,16)).bit_count() for a in fingerprint['hashes'] for b in f['hashes']]
        if min(distances)<=threshold:matches.append({'source_video_id':r['source_video_id'],'distance':min(distances),'status':'UNCERTAIN_REVIEW_REQUIRED'})
    return matches

def validate_run(work,root,parent_hashes):
    import time
    work=Path(work);records=[read_json(p) for p in sorted((work/'records').glob('*.json'))];errors=[];ids=set();hashes=set();checked=[];accepted_annotations=0
    for r in records:
        sid=r['source_video_id']
        if sid in ids:errors.append('Duplicate source ID: '+sid)
        ids.add(sid)
        if not r.get('path'):continue
        try:
            p=safe_under(root,r['path'])
            if not p.resolve().is_relative_to(work.resolve()):raise ValueError('Source outside versioned pilot')
            if digest(p)!=r['sha256'] or p.stat().st_size!=r['bytes']:raise ValueError('Source checksum/size changed')
            start=time.monotonic();actual=validate_video(p);checked.append({'source_video_id':sid,'pass':actual['pass'],'container':actual.get('container_format'),'elapsed_seconds':time.monotonic()-start})
            if r['status'] in ('SOURCE_ACCEPTED','ANNOTATION_ACCEPTED'):
                if not actual['pass'] or r['sha256'] in parent_hashes|hashes:raise ValueError('Accepted corrupt/duplicate source')
                if not r.get('campaign_review_evidence'):raise ValueError('Accepted source lacks campaign evidence')
                if r.get('campaign_status')=='RETAINED_SPLIT_RESTRICTED':
                    config=read_json(work/'run_config.json')
                    from .acquisition import validate_scale_gate
                    if not config.get('engineering_gate'):raise ValueError('Restricted retention lacks P1.6 gate')
                    validate_scale_gate(config)
                    if r.get('fresh_test_eligible') is not False or r.get('split_eligibility')!='RESTRICTED_PENDING_CAMPAIGN_GROUP_REVIEW':raise ValueError('Restricted campaign lost split protection')
                elif r.get('campaign_status')!='REVIEWED_DISTINCT':raise ValueError('Accepted source lacks campaign evidence')
                hashes.add(r['sha256'])
            if r['status']=='ANNOTATION_ACCEPTED':
                from .freeze import validate_annotation
                annotation=read_json(Path(r['accepted_annotation_path']))
                config=read_json(work/'run_config.json')
                validate_annotation(annotation,visual_cache=Path(root)/'cache'/('p1_'+config['run_id'])/'visual_index',root=Path(root))
                accepted_annotations+=1
            if r.get('human_review_status')=='HUMAN_VERIFIED' and not r.get('review_provenance'):raise ValueError('False human verification')
        except (OSError,ValueError) as e:errors.append({'source_video_id':sid,'reason':str(e)})
    checks=read_json(work/'reports/checksums.json') if (work/'reports/checksums.json').is_file() else {}
    for rel,h in checks.items():
        try:
            if digest(safe_under(work,rel))!=h:errors.append('Export checksum changed: '+rel)
        except OSError:errors.append('Export missing: '+rel)
    config=read_json(work/'run_config.json');attempts=[read_json(p) for p in (work/'logs/attempts').glob('*.json')]
    budget_pass=sum(a['bytes_received'] for a in attempts)<=config['max_download_bytes'] and sum(a['http_requests'] for a in attempts)<=config['max_http_requests']
    external=read_json(work/'reports/operational_validation.json') if (work/'reports/operational_validation.json').is_file() else {}
    gates={'integrity':not errors,'accepted_target':len(hashes)>=20,'annotation_coverage':accepted_annotations==len(hashes) and bool(hashes),'resource_budgets':budget_pass,'protected_assets':external.get('protected_assets_pass',False),'resume_tests':external.get('resume_tests_pass',False),'source_permissions':config.get('source_permission',{}).get('status')=='OPERATOR_ATTESTED'}
    return {'integrity_pass':not errors,'pilot_pass':all(gates.values()),'gates':gates,'videos_reprobed':checked,'errors':errors,'accepted_count':len(hashes),'annotation_accepted_count':accepted_annotations,'reason':'Integrity alone does not satisfy pilot acceptance, campaign review, annotation and recovery gates'}
