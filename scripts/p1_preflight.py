"""Read-only P1 operational audit. Writes only a fresh repository evidence folder."""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data_process.lumae_scale.builder import BuildConfig, candidate_plan, parent_snapshot
from data_process.lumae_scale.models import atomic_json, read_json, digest

PROTECTED = [ROOT/'local_data'/n for n in ('raw','datasets','canonical','annotations','intermediate','manifests','exports','reports','cache')]
PROTECTED += [ROOT/'audit_outputs', ROOT/'research_outputs/p0_20261103_01']

def snapshot():
    assets=[];errors=[]
    for root in PROTECTED:
        for directory, dirs, files in os.walk(root, onerror=lambda e:errors.append(str(e))):
            dirs[:] = [d for d in dirs if not (Path(directory)/d).is_symlink() and not (Path(directory)/d).is_junction()]
            for name in sorted(files):
                p=Path(directory)/name
                try:
                    s=p.stat();assets.append({'path':str(p),'bytes':s.st_size,'mtime_ns':s.st_mtime_ns,'sha256':digest(p)})
                    if len(assets)%2000==0:print(json.dumps({'hashed_files':len(assets),'current_root':str(root)}),flush=True)
                except OSError as e:errors.append(str(e))
    return {'created_utc':datetime.now(timezone.utc).isoformat(),'assets':assets,'read_errors':errors,'scope':'All readable protected files; inaccessible directories explicitly reported'}

def compare(before,full=False):
    changed=[];errors=[]
    for item in before['assets']:
        p=Path(item['path'])
        try:
            s=p.stat()
            metadata_changed=s.st_size!=item['bytes'] or s.st_mtime_ns!=item['mtime_ns']
            if full or metadata_changed:
                content_changed=digest(p)!=item['sha256']
                if metadata_changed or content_changed:changed.append({'path':str(p),'content_changed':content_changed,'mtime_changed':s.st_mtime_ns!=item['mtime_ns']})
        except OSError as e:errors.append(str(e))
    known={i['path'] for i in before['assets']};current=set()
    for root in PROTECTED:
        for directory,dirs,files in os.walk(root):
            dirs[:]=[d for d in dirs if not (Path(directory)/d).is_junction() and not (Path(directory)/d).is_symlink()]
            current.update(str(Path(directory)/name) for name in files)
    added=sorted(current-known)
    return {'pass':not changed and not errors and not added,'checked':len(before['assets']),'changed':changed,'added_files':added,'errors':errors,'method':'Full SHA-256 of every snapshotted readable file plus metadata and file-inventory comparison' if full else 'Full initial SHA-256; final stat comparison and SHA on any metadata change; file inventory checked','initial_read_errors':before['read_errors']}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--verify',action='store_true');ap.add_argument('--full',action='store_true');ap.add_argument('--resume-preflight',action='store_true');a=ap.parse_args()
    out=a.output.resolve()
    if not out.is_relative_to(ROOT/'research_outputs') or any(out.is_relative_to(p) for p in PROTECTED):raise ValueError('Fresh research_outputs path required')
    if a.verify:
        result=compare(read_json(out/'protected_asset_snapshot.json'),full=a.full);atomic_json(out/'protected_asset_final_validation.json',result);print(json.dumps({k:v for k,v in result.items() if k!='initial_read_errors'}));return
    if a.resume_preflight:
        if not out.is_dir() or (out/'d500_integrity.json').exists():raise ValueError('Only an incomplete preflight can be resumed')
    else:out.mkdir(parents=True,exist_ok=False)
    if not (out/'git_before.json').exists():atomic_json(out/'git_before.json',{'revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True,cwd=ROOT).strip(),'status':subprocess.check_output(['git','status','--short'],text=True,cwd=ROOT),'diff':subprocess.check_output(['git','diff','--binary'],text=True,cwd=ROOT)})
    if a.resume_preflight and (out/'protected_asset_snapshot.json').exists():
        frozen=read_json(out/'protected_asset_snapshot.json')
        if not compare(frozen)['pass']:raise ValueError('Protected state changed while preflight was paused')
    else:
        frozen=snapshot();atomic_json(out/'protected_asset_snapshot.json',frozen)
    cfg=BuildConfig.load(ROOT/'data_process/config/d1000_v1.json',ROOT);manifest,rows,_=parent_snapshot(cfg)
    raw=[]
    for item in manifest['source_video_sha256'].values():
        p=ROOT/'local_data/raw/adsqa/videos'/item['video_filename'];h=digest(p);raw.append({'path':str(p),'sha256':h,'expected':item['sha256'],'pass':h==item['sha256'],'bytes':p.stat().st_size})
    zip_path=ROOT/'local_data/exports/lumae_ads_d500_colab.zip'
    p0=ROOT/'research_outputs/p0_20261103_01';final=p0/'freeze_final/research_freeze_manifest.json';f=read_json(final)
    integrity={'d500_raw':raw,'raw_pass':all(x['pass'] for x in raw),'raw_bytes':sum(x['bytes'] for x in raw),'unique_hashes':len({x['sha256'] for x in raw}),
        'zip_sha256':digest(zip_path),'expected_zip_sha256':'132621746f9d2538d199e8c5cba1ddec0f0586854ae26c3628478de83ef19345',
        'p0_final_freeze_sha256':digest(final),'p0_final_freeze_expected':(final.with_suffix('.sha256')).read_text().strip(),
        'p0_base_freeze_matches':digest(Path(f['base_freeze']['path']))==f['base_freeze']['sha256'],
        'splits':{s:{'rows':len([json.loads(l) for l in (cfg.parent_dir/f'lumae_ads_d500_{s}.jsonl').read_text().splitlines() if l]),'sha256':digest(cfg.parent_dir/f'lumae_ads_d500_{s}.jsonl')} for s in ('all','train','val','test')},
        'checkpoint_m0_sha256':digest(p0/'assets/M0_pretrained.ckpt'),'cloud_zip_content':'UNVERIFIED_PROVIDER_256_MIB_CAP','historical_d48_access':'See protected snapshot read_errors; never infer empty directory'}
    integrity['pass']=integrity['raw_pass'] and integrity['unique_hashes']==500 and integrity['zip_sha256']==integrity['expected_zip_sha256'] and integrity['p0_base_freeze_matches'] and integrity['p0_final_freeze_sha256']==integrity['p0_final_freeze_expected']
    atomic_json(out/'d500_integrity.json',integrity)
    source=read_json(ROOT/'local_data/raw/adsqa/source_manifest.json');plan=candidate_plan(cfg,source['videos']);atomic_json(out/'candidate_plan.json',plan)
    selected={r['source_video_id']:i for i,r in enumerate(plan['candidates'])};excluded={r['source_video_id']:r['reason'] for r in plan['excluded']}
    selection=read_json(cfg.parent_dir/'selection_manifest.json');states={r['source_video_id']:r for r in selection['candidates']};inherited=selection.get('inherited_terminal_rejections',{})
    with (out/'source_candidate_audit.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        w=csv.DictWriter(stream,fieldnames=['source_video_id','source_url','selection_rank','eligibility','raw_exists','historical_state','historical_rejection']);w.writeheader()
        for r in source['videos']:
            sid=r['source_video_id'];old=states.get(sid,{})
            w.writerow({'source_video_id':sid,'source_url':r['source_url'],'selection_rank':selected.get(sid),'eligibility':'METADATA_ELIGIBLE_NOT_ACCEPTED' if sid in selected else excluded.get(sid),'raw_exists':(ROOT/'local_data/raw/adsqa/videos'/r['target_name']).is_file(),'historical_state':old.get('status','INHERITED_REJECTED' if sid in inherited else 'NOT_ATTEMPTED'),'historical_rejection':old.get('rejection_reason',inherited.get(sid))})
    reserve=15*2**30;projected=2*2**30+512*2**20+256*2**20
    disks={d:dict(zip(('total','used','free'),shutil.disk_usage(d+':/'))) for d in 'CDE'}
    atomic_json(out/'storage_preflight.json',{'evidence':'VERIFIED_CURRENT','disks':disks,'data_root':str(cfg.data_root),'root_exists':cfg.data_root.is_dir(),'same_physical_disk':'INFERRED prior verified IOCTL inventory; E is not independent backup','reserve_bytes':reserve,'pilot_max_download_bytes':2*2**30,'staging_additional_bytes':512*2**20,'annotation_frames_budget_bytes':256*2**20,'projected_peak_additional_bytes':projected,'projected_e_free_bytes':disks['E']['free']-projected,'pass':cfg.data_root.is_dir() and disks['E']['free']-projected>=reserve,'classification':'Budget limits ASSUMED, disk bytes VERIFIED'})
    modules={m:bool(importlib.util.find_spec(m)) for m in ('torch','clip','transformers','cv2','PIL','requests')}
    annotation={'modules':modules,'ffmpeg':shutil.which('ffmpeg'),'ffprobe':shutil.which('ffprobe'),'pipeline':['build_visual_index','extract_clip_features','AgentEvidenceProvider or LocalCommandProvider','temporal_labeler.localize','freeze.validate_annotation','BuilderSession.accept','ReviewStore'],
        'semantic_ranker':'Pinned source check required via verify_frozen_ranker','clip_weights':[{'path':str(p),'sha256':digest(p),'bytes':p.stat().st_size} for p in (ROOT/'local_data/cache/clip_weights').glob('*.pt')],
        'query_generation':'Existing agent-viewed contact sheets supported; no unattended approved visual model configured','semantic_verification':'visual_verifier abstains: compact CPU VLM failed negative control; actual agent/operator evidence required','human_review':'No reviewer fabricated; pending until independent real labels supplied','full_unattended_pipeline_ready':False,'cpu_smoke':'TODO separate execution','storage_binding_gap':'Legacy visual_index defaults historical cache; P1 must pass explicit isolated cache/root'}
    atomic_json(out/'annotation_readiness.json',annotation)
    atomic_json(out/'crawler_readiness.json',{'evidence':'VERIFIED_SOURCE_INSPECTION','candidate_count':len(plan['candidates']),'legacy_downloader':{'https_host_allowlist':True,'range_resume':True,'atomic_part_promotion':True,'redirect_host_enforced':False,'stream_byte_budget':False,'rate_limit':False,'backoff_execution':False},'required_p1_controls':['deny-default network','source permission attestation','no redirects outside allowlist','bounded bytes/time/disk','durable attempt ledger','retry/backoff','technical video validation','campaign quarantine','isolated root'],'live_gate':'PENDING_P1_IMPLEMENTATION_AND_NETWORK_PROBE'})
    print(json.dumps({'output':str(out),'protected_files':len(frozen['assets']),'read_errors':len(frozen['read_errors']),'d500_pass':integrity['pass'],'candidates':len(plan['candidates']),'storage_pass':disks['E']['free']-projected>=reserve}))

if __name__=='__main__':main()
