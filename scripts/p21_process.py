"""P2.1 offline runner. All writes use new namespaces; no network here."""
import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
import threading
from dataclasses import dataclass
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import psutil
from p21_audit import O,E,R,VERSION
from data_process.lumae_scale.models import ROOT,read_json,digest
from data_process.lumae_scale.cache_validation import atomic_npz
from data_process.lumae_scale.incremental import immutable_write,json_bytes
from data_process.lumae_scale.builder import BuildConfig,BuilderSession
from data_process.lumae_scale.annotation_v4 import DEFAULTS,CachedRuntime,key_for,adaptive_times,rank_events,localize_v4,uniform_query_only
from data_process.lumae_scale.clip_runtime import ClipRuntime
from data_process.lumae_scale.temporal_labeler import localize
from data_process.lumae_scale.freeze import validate_annotation
from data_process.lumae_scale.quality_gate import query_errors
from data_process.lumae_scale.pilot_quality import validate_video
from data_process.annotation.independent_review import ReviewStore

CONFIG=O/'annotation_v4/config.json'
WEIGHTS=ROOT/'local_data/cache/clip_weights/ViT-B-32.pt'

@dataclass(frozen=True)
class P21BuildConfig(BuildConfig):
    index_root:Path|None=None
    @property
    def visual_cache(self):return self.index_root or super().visual_cache

def config():return read_json(CONFIG) if CONFIG.exists() else dict(DEFAULTS)

def guard():
    c=config()
    if psutil.virtual_memory().available<c['min_available_memory_bytes']:raise MemoryError('Bounded P21 RAM reserve')
    import shutil
    if shutil.disk_usage(R).free<c['min_free_bytes']+256*2**20:raise OSError('Bounded P21 disk reserve')

def source_rows(stage):
    if stage.startswith('fresh'):
        return read_json(O/'benchmarks/fresh_source_inventory.json')
    return read_json(O/'audit/recovery_inventory.json')

def old_catalog():
    """Externally pinned prior CLIP weight/preprocessing identity is unchanged.

    Old vectors are indexed by exact source/timestamp AND exact JPEG hash, so
    different extractions at the same timestamp cannot silently reuse a vector.
    """
    frames={};vectors={}
    roots=[R/'work/d1000/p14_p15_20261103_01/visual_index',R/'work/d1000/p14_p15_20261103_02/visual_index',
        R/'work/d1000/p16_p2_20261103_01/visual_index',R/'cache/p1_p2_20261103_01/visual_index']
    for root in roots:
        for path in root.glob('*/metadata.json'):
            m=read_json(path);sid=path.parent.name
            features=None
            if (path.parent/'clip_features.npz').exists():
                with np.load(path.parent/'clip_features.npz') as z:features=z['features'].copy();times=z['timestamps'].copy()
                assert features.shape==(len(m['frames']),512) and np.allclose(times,[f['timestamp'] for f in m['frames']])
            for j,f in enumerate(m['frames']):
                image=path.parent/f['path'];stamp=round(float(f['timestamp']),6)
                if not image.is_file():continue
                frames[(sid,stamp)]={'timestamp':stamp,'path':str(image),'sha256':f['sha256'],'original_metadata':str(path)}
                if features is not None:vectors[(sid,stamp,f['sha256'])]=features[j]
    return frames,vectors

def make_sheet(frames,path,columns=6):
    from PIL import Image,ImageDraw
    canvas=Image.new('RGB',(220*columns,150*math.ceil(len(frames)/columns)));draw=ImageDraw.Draw(canvas)
    for j,f in enumerate(frames):
        with Image.open(f['path']) as im:
            im=im.convert('RGB');im.thumbnail((218,122));x=j%columns*220;y=j//columns*150
            canvas.paste(im,(x,y));draw.text((x+2,y+125),f"{f['timestamp']:.3f}s",fill='white')
    path.parent.mkdir(parents=True,exist_ok=True);canvas.save(path,quality=90)

def obtain_frame(row,stamp,catalog,stats):
    sid=row['source_id'];stamp=round(float(stamp),6);known=catalog.get((sid,stamp))
    if known:
        if digest(Path(known['path']))!=known['sha256']:raise ValueError('Changed historical frame')
        stats['frame_cache_hits']+=1;return known
    target=E/'frame_evidence'/sid/(f'{stamp:.6f}'+'.jpg');target.parent.mkdir(parents=True,exist_ok=True)
    if not target.exists():
        guard();start=time.perf_counter()
        subprocess.run(['ffmpeg','-v','error','-ss',str(stamp),'-i',row['path'],'-frames:v','1','-vf','scale=320:-2','-q:v','3',str(target)],check=True,capture_output=True,timeout=30)
        stats['frame_sampling_seconds']+=time.perf_counter()-start;stats['frames_decoded']+=1
    else:stats['frame_cache_hits']+=1
    value={'timestamp':stamp,'path':str(target),'sha256':digest(target),'extraction':'ffmpeg_exact_seek_requested_timestamp'}
    catalog[(sid,stamp)]=value
    return value

def prepare_index(row,times,index_root,runtime,catalog,vectors,stats,c):
    sid=row['source_id'];target=index_root/sid;metadata_path=target/'metadata.json'
    if metadata_path.exists():
        m=read_json(metadata_path)
        if m['video_sha256']!=row['sha256'] or [f['timestamp'] for f in m['frames']]!=times:raise ValueError('Index resume input changed; use new stage/version')
        stats['index_cache_hits']+=1;return m
    frames=[obtain_frame(row,t,catalog,stats) for t in times]
    key=key_for('images',video_sha=row['sha256'],encoder_sha=digest(WEIGHTS),config=c,timestamps=times)
    store=E/'annotation_cache/features'/key;feature_path=store/'clip_features.npz'
    target.mkdir(parents=True,exist_ok=True)
    if feature_path.exists() and (store/'binding.json').exists():
        binding=read_json(store/'binding.json')
        if binding['feature_sha256']!=digest(feature_path):raise ValueError('Feature cache corruption')
        stats['feature_cache_hits']+=len(frames)
    else:
        values=[None]*len(frames);missing=[]
        for j,f in enumerate(frames):
            v=vectors.get((sid,f['timestamp'],f['sha256']))
            if v is None:missing.append(j)
            else:values[j]=v;stats['feature_cache_hits']+=1
        if missing:
            guard();start=time.perf_counter()
            encoded=runtime.encode_images([frames[j]['path'] for j in missing],batch_size=c['max_image_batch'],normalize=True)
            stats['clip_extraction_seconds']+=time.perf_counter()-start;stats['image_encoder_batches']+=math.ceil(len(missing)/c['max_image_batch'])
            for j,v in zip(missing,encoded):values[j]=v;vectors[(sid,frames[j]['timestamp'],frames[j]['sha256'])]=v
            stats['feature_cache_misses']+=len(missing)
        store.mkdir(parents=True,exist_ok=True)
        features=np.asarray(values,dtype=np.float32)
        if features.shape!=(len(frames),512) or not np.isfinite(features).all():raise ValueError('Invalid real feature cache')
        if feature_path.exists():
            # Recovery from a feature-file-before-binding interruption: prove
            # equality with recomputed expected arrays, never bless by name.
            with np.load(feature_path) as z:
                if not np.array_equal(z['features'],features) or not np.array_equal(z['timestamps'],times):raise ValueError('Unbound feature artifact differs from recomputed expected content')
        else:atomic_npz(feature_path,features=features,timestamps=np.asarray(times,dtype=float))
        immutable_write(store/'binding.json',json_bytes({'cache_key':key,'video_sha256':row['sha256'],'encoder_sha256':digest(WEIGHTS),'preprocess':c['preprocess'],'schema':c['feature_schema'],'timestamps':times,'feature_sha256':digest(feature_path),'old_vector_reuse_basis':'Pinned previous pipeline uses the same CLIP weights and official preprocessing; exact JPEG SHA required'}))
    linked=target/'clip_features.npz'
    if linked.exists():
        if digest(linked)!=digest(feature_path):raise ValueError('Changed interrupted index link')
    else:os.link(feature_path,linked)
    overview=target/'overview.jpg';make_sheet(frames,overview)
    m={'video_id':sid,'video_path':row['path'],'video_sha256':row['sha256'],'duration':row['duration'],'frames':frames,
        'contact_sheets':[str(overview)],'clip_features_status':'complete','clip_features_sha256':digest(feature_path),'clip_model':str(WEIGHTS),
        'clip_weights_sha256':digest(WEIGHTS),'clip_device':'cpu','feature_schema':c['feature_schema'],'preprocess':c['preprocess'],'cache_key':key}
    immutable_write(metadata_path,json_bytes(m));return m

def event_payload(e,index):
    return {**e,'evidence_timestamps':[f['timestamp'] for f in index['frames'] if e['coarse_start']<=f['timestamp']<e['coarse_end']]}

def process(stage,input_path):
    c=config();data=read_json(input_path);sources={r['source_id']:r for r in source_rows(stage)}
    if not read_json(O/'validation/historical_preflight.json')['pass']:raise PermissionError('Historical audit gate')
    run=E/'runs'/stage;run.mkdir(parents=True,exist_ok=True)
    immutable_write(run/'input_binding.json',json_bytes({'events_sha256':digest(input_path),'config':c,'input_path':str(input_path),
        'script_sha256':digest(Path(__file__)),'annotation_module_sha256':digest(ROOT/'data_process/lumae_scale/annotation_v4.py')}))
    if (run/'process_profile.json').exists():
        print('RESUME_NO_OP_COMPLETE_STAGE',stage);return
    import torch
    torch.set_num_threads(2);guard();started=time.perf_counter()
    peak={'rss':psutil.Process().memory_info().rss};stop=threading.Event()
    def sample_memory():
        while not stop.wait(.1):peak['rss']=max(peak['rss'],psutil.Process().memory_info().rss)
    threading.Thread(target=sample_memory,daemon=True).start()
    raw=ClipRuntime(str(WEIGHTS),device='cpu',download_root=E/'model_cache')
    warm=time.perf_counter();raw.encode_text('a moving object');warm_seconds=time.perf_counter()-warm
    runtime=CachedRuntime(raw,E/'annotation_cache/text',digest(WEIGHTS),c)
    catalog,vectors=old_catalog();profiles=[]
    for observations in data:
        sid=observations['source_id'];row=sources[sid];folder=run/'sources'/sid
        if (folder/'complete.json').exists():continue
        start=time.perf_counter();guard()
        if digest(Path(row['path']))!=row['sha256']:raise ValueError('Changed original MP4')
        stats={k:0 for k in ('frame_cache_hits','frames_decoded','frame_sampling_seconds','feature_cache_hits','feature_cache_misses','clip_extraction_seconds','image_encoder_batches','index_cache_hits')}
        folder.mkdir(parents=True,exist_ok=True)
        immutable_write(folder/'observations.json',json_bytes(observations))
        probe_start=time.perf_counter();technical=validate_video(Path(row['path']));probe_seconds=time.perf_counter()-probe_start
        if not technical['pass']:raise ValueError('Previously valid source failed revalidation')
        ranked=rank_events(observations.get('events',[]),c)
        if not ranked:
            immutable_write(folder/'complete.json',json_bytes({'source_id':sid,'status':'ABSTAIN_NO_DEFENSIBLE_EVENT','reason':observations.get('abstention_reason'),'human_verified':False,'seconds':time.perf_counter()-start}))
            print(sid,'ABSTAIN',flush=True);continue
        sparse_times=np.arange(0,row['duration']-.05,2).tolist()
        sparse=prepare_index(row,sparse_times,run/'sparse_index',runtime,catalog,vectors,stats,c)
        dense_times=adaptive_times(row['duration'],ranked,c)
        dense=prepare_index(row,dense_times,run/'visual_index',runtime,catalog,vectors,stats,c)
        # Uniform query-only procedure is computed BEFORE any generator proposal.
        blind=[]
        for e in ranked:blind.append(uniform_query_only(sid,e['query'],cache=run/'visual_index',runtime=runtime,duration=row['duration']))
        immutable_write(folder/'blind_first.json',json_bytes(blind))
        outputs=[]
        first=observations['events'][0];best=ranked[0]
        # The existing event provider already requires actual visual inspection.
        # B0/B1 therefore share the first genuine event: query-awareness alone
        # is not an independently identified improvement and must not be sold as
        # one. This replay never receives prior adjudicated intervals as inputs.
        variants=[('B0',first,run/'sparse_index',sparse),('B1',first,run/'sparse_index',sparse),
            ('B2',best,run/'sparse_index',sparse),('B3',best,run/'visual_index',dense)]
        for variant,e,cache,index in variants:
            t=time.perf_counter()
            try:
                pred=localize(sid,e['query'],[event_payload(e,index)],row['duration'],0,clip_runtime=runtime,visual_cache=cache)
                outputs.append({'variant':variant,'source_id':sid,'query':e['query'],'prediction':pred,'seconds':time.perf_counter()-t,
                    'status':'PROPOSED_NOT_ACCEPTED','baseline_protocol':'Frozen V3 code; controlled uniform2s replay, not a claimed byte-exact replay of historical mixed sampling' if variant=='B0' else None})
            except ValueError as ex:outputs.append({'variant':variant,'source_id':sid,'status':'ABSTAIN','reason':str(ex),'seconds':time.perf_counter()-t})
        t=time.perf_counter()
        try:
            old,pred=localize_v4(sid,best['query'],[event_payload(best,dense)],row['duration'],visual_cache=run/'visual_index',runtime=runtime,config=c)
            outputs.append({'variant':'B4','source_id':sid,'query':best['query'],'prediction':pred,'seconds':time.perf_counter()-t,'status':'PROPOSED_NOT_ACCEPTED'})
            repeat=time.perf_counter();_,again=localize_v4(sid,best['query'],[event_payload(best,dense)],row['duration'],visual_cache=run/'visual_index',runtime=runtime,config=c)
            assert pred==again
            outputs.append({'variant':'B5','source_id':sid,'query':best['query'],'prediction':again,'seconds':time.perf_counter()-repeat,'status':'PROPOSED_NOT_ACCEPTED','same_output_as_B4':True,'efficiency_control':'Exact text/frame feature cache; no repeated image encoding'})
        except ValueError as ex:outputs.append({'variant':'B4','source_id':sid,'status':'ABSTAIN','reason':str(ex),'seconds':time.perf_counter()-t})
        immutable_write(folder/'proposals.json',json_bytes(outputs))
        for result in outputs:
            if 'prediction' not in result:continue
            immutable_write(folder/(result['variant']+'_prediction.json'),json_bytes(result['prediction']))
        # One coherent dense context supports review of every variant; raw
        # bounds are in JSON, never silently replaced with the observed phase.
        bestpred=next((v['prediction'] for v in outputs if v['variant']=='B4' and 'prediction' in v),None)
        if bestpred:
            s=max(0,min(best['coarse_start'],bestpred['start'])-1);end=min(row['duration'],max(best['coarse_end'],bestpred['end'])+1)
            qc=[f for f in dense['frames'] if s<=f['timestamp']<=end]
            make_sheet(qc,folder/'qc.jpg');immutable_write(folder/'qc_frames.json',json_bytes(qc))
        profile={'source_id':sid,'probe_decode_seconds':probe_seconds,**stats,'total_computation_wall_seconds':time.perf_counter()-start,
            'event_discovery_seconds':None,'query_generation_seconds':None,'ai_visual_review_seconds':None,
            'missing_stage_reason':'Interactive stages separately timed in session logs; development images were viewed before timing, so not measured here',
            'rss_bytes':psutil.Process().memory_info().rss,'local_text_calls_cumulative':runtime.text_calls,'text_hits_cumulative':runtime.text_hits}
        profiles.append(profile);immutable_write(folder/'profile.json',json_bytes(profile))
        immutable_write(folder/'complete.json',json_bytes({'source_id':sid,'status':'PENDING_ACTUAL_VISUAL_REVIEW','input_sha256':digest(input_path)}))
        print(sid,[(v['variant'],[v['prediction']['start'],v['prediction']['end']] if 'prediction' in v else v['status']) for v in outputs],flush=True)
    stop.set()
    immutable_write(run/'process_profile.json',json_bytes({'stage':stage,'runtime_warmup_seconds':warm_seconds,'total_process_wall_seconds':time.perf_counter()-started,'peak_rss_sampled_bytes':peak['rss'],'rss_sample_interval_seconds':.1,'sources_processed_this_invocation':len(profiles),'profiles':profiles,'text_encoder_calls':runtime.text_calls,'text_cache_hits':runtime.text_hits,'image_api_calls':runtime.image_calls,'image_frames_encoded':runtime.image_frames,'human_verified':False,'pipeline_end_to_end_seconds':None,'not_end_to_end_reason':'This command excludes separately measured interactive event discovery/query and review sessions'}))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--stage',required=True);ap.add_argument('--events',type=Path,required=True);a=ap.parse_args();process(a.stage,a.events)
