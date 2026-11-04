"""Isolated real-video indexing and read-only parent fingerprint preparation."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from data_process.lumae_scale.models import ROOT,read_json,atomic_json,digest
from data_process.lumae_scale.visual_index import build_visual_index
from data_process.lumae_scale.pilot_quality import frame_hash,validate_video

def baseline(output,limit=None):
    manifest=read_json(ROOT/'local_data/datasets/lumae_ads_d500/dataset_manifest.json')
    rows=read_json(output)['videos'] if output.exists() else [];done={r['source_video_id'] for r in rows}
    items=sorted(manifest['source_video_sha256'].values(),key=lambda x:x['video_filename'])
    new=0
    for item in items:
        sid=Path(item['video_filename']).stem
        if sid in done:continue
        if limit and new>=limit:break
        path=ROOT/'local_data/raw/adsqa/videos'/item['video_filename']
        if digest(path)!=item['sha256']:raise ValueError('Frozen parent video hash mismatch')
        info=validate_video(path)
        if not info['pass']:raise ValueError('Parent fingerprint failed: '+sid)
        rows.append({'source_video_id':sid,'sha256':item['sha256'],'fingerprint':info['fingerprint']});new+=1
        atomic_json(output,{'dataset_sha256':manifest['dataset_sha256'],'videos':rows,'complete':len(rows)==500,'scope':'Real original MP4, three decoded frames per video; candidate triage only, no campaign independence proof'})
        if new%25==0:print('Fingerprint baseline:',len(rows),flush=True)
    print('Fingerprint baseline:',len(rows),flush=True)

def index(config):
    c=read_json(config);root=Path(c['data_root']).resolve();work=root/'work/d1000'/('pilot_'+c['run_id'])
    cache=root/'cache'/('p1_'+c['run_id'])/'visual_index';results=[]
    for file in sorted((work/'records').glob('*.json')):
        record=read_json(file)
        if not record.get('technical',{}).get('pass'):continue
        from data_process.lumae_scale.builder import safe_under
        source=safe_under(root,record['path'])
        if digest(source)!=record['sha256']:raise ValueError('Pilot source changed')
        import shutil
        usage=sum(p.stat().st_size for folder in (work,cache) for p in folder.rglob('*') if p.is_file())
        if usage+64*2**20>c['max_staging_bytes'] or shutil.disk_usage(root).free-64*2**20<c['min_free_bytes']:raise ValueError('Processing storage limit')
        value=build_visual_index(source,record['technical']['probe']['duration'],record['sha256'],work=work,strict=True,visual_cache=cache,root=root)
        results.append({'source_video_id':record['source_video_id'],'frames':len(value['frames']),'contact_sheets':len(value['contact_sheets']),'metadata_sha256':digest(cache/record['source_video_id']/'metadata.json'),'query_status':'INSPECTED' if (work/'visual_index'/record['source_video_id']/'visual_events.json').exists() else 'PENDING_ACTUAL_VISUAL_INSPECTION','annotation_status':'PENDING','clip_extracted':value.get('clip_features_status')=='complete'})
        atomic_json(work/'reports/visual_index_results.json',results)
        print(record['source_video_id'],len(value['frames']),flush=True)

ap=argparse.ArgumentParser();ap.add_argument('command',choices=['baseline','index']);ap.add_argument('--output',type=Path);ap.add_argument('--config',type=Path);ap.add_argument('--limit',type=int);a=ap.parse_args()
if a.command=='baseline':baseline(a.output,a.limit)
else:index(a.config)
