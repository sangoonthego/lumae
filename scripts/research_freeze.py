"""Metadata-first research freeze; inputs read-only, output must be a new version."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

EXPECTED_ZIP = '132621746f9d2538d199e8c5cba1ddec0f0586854ae26c3628478de83ef19345'

def checksum(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def create(root, output, audit):
    root, output, audit = map(lambda p: Path(p).resolve(), (root, output, audit))
    if (output / 'research_freeze_manifest.json').exists():
        raise FileExistsError('Use a new freeze version; existing manifest is immutable')
    output.mkdir(parents=True, exist_ok=True)
    paths = set()
    errors = []
    for rel in ('raw', 'datasets', 'canonical', 'annotations', 'intermediate', 'manifests', 'exports', 'reports', 'cache'):
        for parent, dirs, files in os.walk(root / 'local_data' / rel, onerror=lambda e: errors.append(str(e)), followlinks=False):
            dirs[:] = [d for d in dirs if not (Path(parent)/d).is_junction() and not (Path(parent)/d).is_symlink()]
            paths.update(Path(parent)/f for f in files)
    tracked = subprocess.check_output(['git', 'ls-files'], cwd=root, text=True).splitlines()
    paths.update(root/p for p in tracked if (root/p).is_file())
    paths.update(p for p in audit.rglob('*') if p.is_file())
    assets = []
    for p in sorted(paths):
        try:
            stat = p.stat()
            assets.append({'path': str(p), 'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns,
                           'sha256': checksum(p), 'classification': 'KEEP — PROTECTED', 'evidence': 'VERIFIED_CURRENT_CONTENT'})
        except OSError as e:
            errors.append(f'{p}: {e}')
    dataset = root/'local_data/datasets/lumae_ads_d500'
    rows = {s: [json.loads(l) for l in (dataset/f'lumae_ads_d500_{s}.jsonl').read_text(encoding='utf-8').splitlines() if l]
            for s in ('all', 'train', 'val', 'test')}
    raw = root/'local_data/raw/adsqa/videos'
    filenames = {r['metadata']['video_filename'] for r in rows['all']}
    actual_zip = next(a for a in assets if a['path'] == str(root/'local_data/exports/lumae_ads_d500_colab.zip'))
    cloud = read(audit/'cloud_inventory.json')['files']
    feature_snapshot = read(audit/'cloud_feature_validation.json')
    environment = {'python': platform.python_version(), 'platform': platform.platform(),
                   'historical_training_environment': 'TODO exact historical CUDA/package lock unavailable'}
    experiments = {'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
                   'upstream_commit': 'b7e553ac3b0c898ee6b85e03ee507c064eab89ca',
                   'd500_e0_config': read(audit/'training_config.json'),
                   'cloud_research_assets': [{**f, 'classification': 'KEEP — PROTECTED',
                       'evidence': 'PRIOR_AUDIT_LISTING_NOT_CURRENT_CONTENT_HASH'} for f in cloud if f['file_or_folder']=='file'],
                   'notebook_source': {'path':str(audit/'e0_notebook_sources.txt'),'sha256':checksum(audit/'e0_notebook_sources.txt')},
                   'environment':environment}
    result = {'schema_version': 1, 'created_utc': datetime.now(timezone.utc).isoformat(), 'version': output.name,
              'dataset_version': 'lumae_ads_d500', 'git_commit': experiments['git_commit'],
              'preexisting_git_status': subprocess.check_output(['git','status','--short'],cwd=root,text=True),
              'dataset': {'counts': {s:len(rows[s]) for s in ('train','val','test')},
                  'unique_videos':len({r['vid'] for r in rows['all']}),'raw_bytes':sum((raw/n).stat().st_size for n in filenames),
                  'samples':[{'qid':r['qid'],'vid':r['vid'],'split':s,'query':r['query'],
                      'provenance':r.get('metadata',{}).get('review_provenance'),
                      'review_status':r.get('metadata',{}).get('query_review_status'),
                      'temporal_human_status':'HUMAN_VERIFIED' if r.get('metadata',{}).get('review_provenance')=='HUMAN_VERIFIED' else 'AI_ONLY_NOT_HUMAN_GT',
                      'source_sha256':r.get('metadata',{}).get('source_video_sha256')} for s in ('train','val','test') for r in rows[s]]},
              'local_zip':{**actual_zip,'expected_sha256':EXPECTED_ZIP,'pass':actual_zip['sha256']==EXPECTED_ZIP},
              'cloud_backup':{'status':'UNVERIFIED','reason':'Matching metadata is not content proof; stream result recorded separately if available'},
              'feature_cache':{'evidence':'PRIOR_AUDIT_CONTENT_SNAPSHOT; preserve bytes, current checks recorded separately',
                   'snapshot_file':str(audit/'cloud_feature_validation.json'),'assets':feature_snapshot,
                   'dtype_exceptions':[f for f in feature_snapshot if any(a['dtype']!='<f4' for a in f.get('arrays',[]))]},
              'assets':assets,'read_errors':errors,'completeness':'PARTIAL_ACCESS' if errors else 'READABLE_SCOPE_COMPLETE',
              'cleanup_allowed':False,'historical_training_reproduction_claimed':False}
    for name, value in [('research_freeze_manifest.json',result),('historical_experiments_manifest.json',experiments)]:
        with (output/name).open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2)
    with (output/'research_asset_checksums.csv').open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(assets[0]));w.writeheader();w.writerows(assets)
    (output/'research_freeze_manifest.sha256').write_text(checksum(output/'research_freeze_manifest.json')+'\n',encoding='ascii')
    (output/'backup_integrity_report.md').write_text(
        f'# P0.0 — Research freeze\n\nVERIFIED local D500 ZIP SHA256: {actual_zip["sha256"]}; PASS={result["local_zip"]["pass"]}. '
        f'{len(assets)} readable files hashed; {len(errors)} read errors, explicitly retained in manifest. '
        'Cloud backup UNVERIFIED until actual stream SHA256 comparison. No cleanup or destructive migration authorized. '
        'Feature/checkpoint cloud identities from prior audit are timestamped evidence, not fresh content verification. '
        'All discovered local and listed cloud research assets classified protected. New metadata freeze does not rewrite historical manifests.\n',encoding='utf-8')
    print(json.dumps({'files':len(assets),'read_errors':len(errors),'zip_pass':result['local_zip']['pass']}))
    return result

def verify(manifest):
    value=read(manifest);changed=[];errors=[]
    for a in value['assets']:
        try:
            p=Path(a['path']);s=p.stat()
            if s.st_size!=a['bytes'] or s.st_mtime_ns!=a['mtime_ns']:
                if checksum(p)!=a['sha256']:changed.append(str(p))
        except OSError as e:errors.append(str(e))
    return {'mode':'stat check; SHA256 on changed metadata','changed':changed,'errors':errors,'pass':not changed and not errors}

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path.cwd());parser.add_argument('--output',type=Path);parser.add_argument('--audit',type=Path);parser.add_argument('--verify',type=Path)
    a=parser.parse_args()
    if a.verify:print(json.dumps(verify(a.verify)))
    elif a.output and a.audit:create(a.root,a.output,a.audit)
    else:parser.error('--output and --audit required for a new freeze')
