"""Real fresh-wave overview preparation and batch end-to-end clock closure."""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from p21_audit import O,E,R
from p21_process import obtain_frame,make_sheet,guard
from data_process.lumae_scale.models import read_json,digest
from data_process.lumae_scale.incremental import immutable_write,json_bytes

def overviews():
 start=time.perf_counter();rows=[];profiles=[]
 for record in sorted((E/'fresh_wave/records').glob('*.json')):
  r=read_json(record)
  if not r.get('technical',{}).get('pass') or r.get('reason')=='EXACT_CONTENT_DUPLICATE':continue
  sid=r['source_video_id'];row={'source_id':sid,'duration':r['technical']['probe']['duration'],'path':str(R/r['path']),'sha256':r['sha256'],
     'source_url':r['source_url'],'duplicate_candidates':r.get('duplicate_candidates',[]),'origin':'P21_FRESH_SINGLE_WAVE'}
  guard();started=time.perf_counter();stats={'frame_sampling_seconds':0,'frames_decoded':0,'frame_cache_hits':0}
  frames=[obtain_frame(row,t,{},stats) for t in np.arange(0,row['duration']-.05,2)]
  image=E/'fresh_wave/evidence'/sid/'overview.jpg';make_sheet(frames,image)
  immutable_write(image.with_suffix('.json'),json_bytes({'frames':frames,'source_sha256':row['sha256'],'sampling_seconds':2,'overview_sha256':digest(image)}))
  row.update(overview_path=str(image),overview_sha256=digest(image));rows.append(row)
  profiles.append({'source_id':sid,**stats,'total_overview_wall_seconds':time.perf_counter()-started})
  print(sid,str(image),flush=True)
 immutable_write(O/'benchmarks/fresh_source_inventory.json',json_bytes(rows))
 immutable_write(O/'benchmarks/fresh_overview_profile.json',json_bytes({'sources':profiles,'wall_seconds':time.perf_counter()-start}))

def finish():
 start=read_json(O/'benchmarks/fresh_pipeline_clock.json');end=time.time()
 phases={}
 for path in (O/'benchmarks/interactive_timing').glob('*_start.json'):
  s=read_json(path);e=read_json(path.with_name(path.name.replace('_start.json','_end.json')))
  phases[s['phase']]={'wall_seconds':e['epoch']-s['epoch'],'start_epoch':s['epoch'],'end_epoch':e['epoch'],'scope':'Batch actual AI interface wall time including inspection, reasoning and source-specific transcription; no invented per-source allocation'}
 immutable_write(O/'benchmarks/fresh_pipeline_clock_complete.json',json_bytes({**start,'ended_epoch':end,'actual_experimental_end_to_end_seconds':end-start['started_epoch'],'interactive_phases':phases,'includes_baseline_ablation_overhead':True,'setup_and_report_writing_included':False,'human_verified':False}))
 print('ACTUAL_END_TO_END_SECONDS',end-start['started_epoch'])

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('command',choices=['overviews','finish']);a=ap.parse_args()
 overviews() if a.command=='overviews' else finish()
