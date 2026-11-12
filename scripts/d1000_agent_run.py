"""Resume D500-compatible collection. Observations/verdicts must come from actual image inspection."""
import argparse, hashlib, json, os, shutil, sys, time
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from data_process.lumae_scale.models import ROOT,read_json,atomic_json,digest,load_jsonl
from data_process.lumae_scale.quality_gate import query_errors,interval_errors
from data_process.lumae_scale.cache_validation import visual_index_valid,clip_npz_valid
from data_process.lumae_scale.source_inventory_guard_v2 import inventory_mp4
from data_process.lumae_scale.query_generator import validate_events
from data_process.lumae_scale.freeze import validate_annotation,canonical_row
from data_process.lumae_scale.visual_index import build_visual_index,extract_clip_features
from data_process.lumae_scale.temporal_labeler import localize,WRAPPER_VERSION
from data_process.lumae_scale.router import route_candidate
from data_process.lumae_scale.visual_verifier import verify
from data_process.adsqa.video_validation import probe_video_file

PREVIOUS=Path('E:/LUMAE/work/d1000/d1000_fasttrack_20261104_01')
BASE=Path('E:/LUMAE/work/d1000/d1000_agent_20261104_01')
OUT=ROOT/'research_outputs/d1000_agent_20261104_01'
P21=ROOT/'research_outputs/p21_20261103_01'
POLICY='D500_COMPAT_20261104_v1'
_ACCEPTANCE_CHECKS={}
_PROTECTED_CHECKS={}
NETWORK_OPTIONS={}

def observe(rows):
 """Persist the active agent's actual image observations through the existing file contract."""
 queue={r['source_id']:r for r in read_json(BASE/'review_queue.json')}
 for sid,query,start,end,times,subject,action,obj in rows:
  if query_errors(query):raise ValueError('Original D500 query sanitation: '+str(query_errors(query)))
  r=queue[sid];index=read_json(Path(r['cache'])/sid/'metadata.json')
  event={'query':query,'coarse_start':start,'coarse_end':end,'evidence_timestamps':times,'visible_subject':subject,'visible_action':action,'visible_object':obj,
   'evidence_sheets':sorted({index['contact_sheets'][n//10] for n,f in enumerate(index['frames']) if f['timestamp'] in times})}
  validate_events(sid,[event],r['duration'],visual_cache=Path(r['cache']))
  path=BASE/'observations'/(sid+'.json')
  value={'inspection_method':'agent_viewed_overview','overview_path':r['overview'],'overview_sha256':digest(Path(r['overview'])),'events':[event],'human_verified':False}
  if path.exists() and read_json(path)!=value:
   assert not (BASE/'accepted'/(sid+'.json')).exists(),'Accepted observations immutable'
   previous=read_json(path);oldquery=previous['events'][0]['query'];tag=hashlib.sha256(oldquery.encode()).hexdigest()
   if not query_errors(oldquery):assert read_json(BASE/'verdicts'/sid/(tag+'.json'))['pass'] is False,'Alternative requires actual negative verdict'
   history=BASE/'observations/history'/sid
   assert len(list(history.glob('*.json')))<2,'Bounded3 query attempts'
   atomic_json(history/(digest(path)+'.json'),previous)
   state=BASE/'states'/(sid+'.json')
   if state.exists() and read_json(state).get('status')=='REJECTED_ANNOTATION':
    atomic_json(BASE/'states/history'/sid/(digest(state)+'.json'),read_json(state))
    atomic_json(state,{'status':'PENDING_REVIEW','reason':'Bounded alternative after preserved prior rejection'})
  atomic_json(path,value)

def verdict(rows):
 """Persist genuine agent strip judgments; no automatic positive verdict."""
 for sid,passed,reason in rows:
  state=read_json(BASE/'states'/(sid+'.json'));prediction=Path(state['prediction'])
  assert digest(Path(state['image']))==state['image_sha256']
  value={'pass':passed,'notes':reason,'image_sha256':state['image_sha256'],'prediction_sha256':digest(prediction),'evidence_timestamps':state['evidence_timestamps'],'human_verified':False,'reviewer':'Codex actual image inspection; model revision NOT_EXPOSED'}
  path=BASE/'verdicts'/sid/(prediction.stem+'.json')
  if path.exists():assert read_json(path)==value,'Do not overwrite prior verdict'
  else:atomic_json(path,value)

def jsonl(path,rows):
 path.parent.mkdir(parents=True,exist_ok=True)
 data=''.join(json.dumps(r,sort_keys=True,ensure_ascii=False,allow_nan=False)+'\n' for r in rows)
 tmp=path.with_suffix('.tmp');tmp.write_text(data,encoding='utf-8');os.replace(tmp,path)

def init():
 BASE.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
 parent=ROOT/'local_data/datasets/lumae_ads_d500';manifest=read_json(parent/'dataset_manifest.json')
 canonical=parent/'lumae_ads_d500_all.jsonl'
 assert digest(canonical)==manifest['dataset_sha256']
 prior=load_jsonl(ROOT/'research_outputs/d1000_fasttrack_20261104_01/accepted_annotations.jsonl')
 assert len(prior)==len({r['source_video_id'] for r in prior})
 seal=read_json(P21/'manifests/p21_execution_manifest.json')
 for name,h in seal['artifact_sha256'].items():assert digest(P21/name)==h,name
 protected={str(p):digest(p) for p in parent.iterdir() if p.is_file()}
 protected.update({str(ROOT/name):digest(ROOT/name) for name in ['data_process/lumae_scale/temporal_labeler.py','data_process/lumae_scale/freeze.py','data_process/lumae_scale/quality_gate.py','data_process/lumae_scale/router.py','data_process/lumae_scale/visual_index.py']})
 identity={'policy':POLICY,'parent':str(canonical),'parent_sha256':digest(canonical),'prior':str(ROOT/'research_outputs/d1000_fasttrack_20261104_01/accepted_annotations.jsonl'),'prior_sha256':digest(ROOT/'research_outputs/d1000_fasttrack_20261104_01/accepted_annotations.jsonl'),'protected':protected,'target_new':500,'min_free_bytes':15*2**30,'batch_size':25,'query_attempts':3,'sampling_step':2,'wrapper':WRAPPER_VERSION,'research_gold':False}
 pin=BASE/'config.json'
 if pin.exists():assert read_json(pin)==identity,'Resume identity changed'
 else:atomic_json(pin,identity)
 if not (BASE/'inventory.json').exists():
  roots=read_json(P21/'candidate_ranking/next_pool_cache_audit.json')['registered_roots']
  raw=inventory_mp4(roots)
  pool=read_json(ROOT/'local_data/raw/adsqa/source_manifest.json')['videos']
  excluded={r['vid'].removeprefix('lumae_ads_') for r in load_jsonl(canonical)}|{r['source_video_id'] for r in prior}
  rows=[{**r,'existing_paths':raw.get(r['source_video_id'],[])} for r in pool if r['source_video_id'] not in excluded]
  rows.sort(key=lambda r:(not bool(r['existing_paths']),r['source_video_id']))
  atomic_json(BASE/'inventory.json',rows)
  for p in (PREVIOUS/'states').glob('*.json'):
   record=read_json(p)
   if record.get('status','').startswith('REJECTED'):atomic_json(BASE/'states'/p.name,{**record,'inherited_from':str(p)})
 return identity

def accepted():
 records=[]
 for p in sorted((BASE/'accepted').glob('*.json')):
  r=read_json(p)
  assert r['annotation_policy']==POLICY and r['human_verified'] is False
  state=BASE/'states'/(r['source_video_id']+'.json')
  if state.exists() and read_json(state).get('record_sha256'):
   assert read_json(state)['record_sha256']==digest(p),'Accepted record changed'
  # Full original validation once per process/input identity, then reuse while
  # the immutable record and bound inputs retain their filesystem identities.
  inputs=[p]+[Path(r[k]) for k in ('source_video_path','semantic_artifact_path','verification_image_path') if r.get(k)]
  index=Path(r['cache_root'])/r['source_video_id']/'metadata.json'
  if index.exists():inputs.append(index)
  signature=(digest(p),tuple((str(x),x.stat().st_size,x.stat().st_mtime_ns) for x in inputs))
  if _ACCEPTANCE_CHECKS.get(str(p))!=signature:
   validate_annotation(r,visual_cache=Path(r['cache_root']))
   _ACCEPTANCE_CHECKS[str(p)]=signature
  records.append(r)
 return records

def network_batch(*,resume_connection_batch=None):
 """New bounded batches, same existing P1 crawler; no sealed pilot is resumed."""
 from data_process.lumae_scale.acquisition import Pilot,run_lock
 from data_process.lumae_scale.builder import BuildConfig
 inventory=read_json(BASE/'inventory.json')
 roots=NETWORK_OPTIONS.get('registered_roots') or read_json(P21/'candidate_ranking/next_pool_cache_audit.json')['registered_roots']
 current=inventory_mp4(roots)
 for r in inventory:
  if not r['existing_paths'] and current.get(r['source_video_id']):r['existing_paths']=current[r['source_video_id']]
 # Cached sources have priority, including externally added local files.
 if any(r['existing_paths'] and not (BASE/'states'/(r['source_video_id']+'.json')).exists() for r in inventory):
  inventory.sort(key=lambda r:(not bool(r['existing_paths']),r['source_video_id']));atomic_json(BASE/'inventory.json',inventory);return
 pending=[r for r in inventory if not r['existing_paths'] and not (BASE/'states'/(r['source_video_id']+'.json')).exists()][:25]
 if not pending:raise RuntimeError('No eligible sources remain')
 number=len(list((BASE/'network').glob('batch_*')))+1
 folder=BASE/'network'/f'batch_{number:04d}'
 # A partially completed batch is resumed with the same selection and budgets.
 incomplete=[p for p in (BASE/'network').glob('batch_*') if not (p/'bulk_complete.json').exists()]
 if incomplete:folder=sorted(incomplete)[0];number=int(folder.name.split('_')[1]);pending=read_json(folder/'source_selection.json')['candidates']
 if resume_connection_batch is not None and (folder not in incomplete or folder!=resume_connection_batch):
  raise RuntimeError('Authorized connection recovery must match the active incomplete batch')
 config=read_json(P21/'benchmarks/fresh_wave_config.json')
 config.update(run_id=f'agent_20261104_b{number:04d}',mode='pilot',target_accepted=25,max_candidates=25,previous_runs=[],engineering_gate=None,operational_gate=str(OUT/'network_gate.json'))
 config['source_permission']={**config['source_permission'],'scope':'Current authorized agent-driven D1000 collection with bounded25 internal batches and unchanged resource/access limits','authorization_attachment':'74fc69fa-7d2b-437d-a102-4558129d93bb/Pasted text.txt sections8,9'}
 if NETWORK_OPTIONS:
  config['run_id']=f'{BASE.name}_b{number:04d}'
  config['source_permission'].update(NETWORK_OPTIONS['source_permission'])
  saved_config=folder/'run_config.json'
  if saved_config.exists():
   config['previous_runs']=read_json(saved_config)['previous_runs']
  else:
   data_root=Path(config['data_root']).resolve()
   previous_manifests=[p.parent/'manifests/pilot_download_manifest.json' for p in sorted((BASE/'network').glob('batch_*/bulk_complete.json'))]
   config['previous_runs']=[{'manifest':p.resolve().relative_to(data_root).as_posix(),'sha256':digest(p)} for p in previous_manifests]
 tests=read_json(OUT/'targeted_tests.json')
 if not tests.get('pass'):raise RuntimeError('Fast-track targeted tests must pass before network execution')
 atomic_json(OUT/'network_gate.json',{'protected_assets_pass':bool(progress()),'storage_pass':shutil.disk_usage(BASE).free>20*2**30,'source_permissions_attested':True,'crawler_tests_pass':True,'source_manifest_sha256':config['source_manifest_sha256']})
 build=BuildConfig.load(ROOT/config['builder_config'],ROOT)
 pilot=Pilot(config,build,read_json(ROOT/config['source_manifest'])['videos'],live=True)
 pilot.work=folder;pilot.plan.update(candidates=pending,previous_accepted_sources=0,scope=NETWORK_OPTIONS.get('scope','D1000_COMPAT_INTERNAL_BATCH'))
 if NETWORK_OPTIONS:
  pilot.parent_hashes.update(NETWORK_OPTIONS['parent_hashes'])
 metrics=pilot.run(max_wave=25,resume_connection_failures=resume_connection_batch is not None)
 for record in pilot.records():
  sid=record['source_video_id'];p=folder/'downloads'/(sid+'.mp4')
  if NETWORK_OPTIONS and record['status']=='REJECTED':
   reason=record.get('reason')
   status='REJECTED_DUPLICATE' if reason=='EXACT_CONTENT_DUPLICATE' else 'REJECTED_TECHNICAL'
   atomic_json(BASE/'states'/(sid+'.json'),{'status':status,'reason':reason,'sha256':record.get('sha256'),'acquisition_record':str(folder/'records'/(sid+'.json'))})
   continue
  if p.exists():
   for r in inventory:
    if r['source_video_id']==sid:r['existing_paths']=[str(p)]
  else:atomic_json(BASE/'states'/(sid+'.json'),{'status':'REJECTED_TECHNICAL','reason':record.get('reason'),'acquisition_record':str(folder/'records'/(sid+'.json'))})
 atomic_json(BASE/'inventory.json',sorted(inventory,key=lambda r:(not bool(r['existing_paths']),r['source_video_id'])))
 if len(pilot.records())==len(pending):atomic_json(folder/'bulk_complete.json',metrics)
 else:raise RuntimeError('Acquisition checkpoint: '+metrics['stop_reason'])

def prior_annotations(config):
 return load_jsonl(Path(config['prior'])) if config.get('prior') else []

def verify_pins(pins):
 for name,expected in pins.items():
  path=Path(name)
  info=path.stat()
  key=(info.st_size,info.st_mtime_ns,info.st_ctime_ns,expected)
  if _PROTECTED_CHECKS.get(name)!=key:
   assert digest(path)==expected,'Protected state changed: '+name
   _PROTECTED_CHECKS[name]=key

def progress():
 c=read_json(BASE/'config.json');prior=prior_annotations(c);new=prior+accepted()
 assert len(new)==len({r['source_video_id'] for r in new})==len({r['source_video_sha256'] for r in new})
 assert len(new)==len({r['qid'] for r in new}) and len(new)<=c['target_new']
 parent_rows=load_jsonl(Path(c['parent']))
 assert not {r['vid'].removeprefix('lumae_ads_') for r in parent_rows}&{r['source_video_id'] for r in new}
 verify_pins(c['protected'])
 if c.get('prior'):assert digest(Path(c['prior']))==c['prior_sha256']
 base=c.get('base_count',500);target=c['target_new']
 if c.get('base_count'):
  assert len(parent_rows)==base
  assert not {r['qid'] for r in parent_rows}&{r['qid'] for r in new}
  assert not set(c['parent_hashes'])&{r['source_video_sha256'] for r in new}
 result={'historical':base,'new_accepted':len(new),'total':base+len(new),'remaining':target-len(new),'complete':len(new)==target,'free_bytes':shutil.disk_usage(BASE).free,'policy':POLICY,'human_verified_new':0}
 jsonl(OUT/'accepted_annotations.jsonl',new)
 jsonl(OUT/'provisional_dataset.jsonl',parent_rows+[canonical_row(r,r['qid']) for r in new])
 atomic_json(OUT/'progress.json',result)
 states={p.stem:read_json(p) for p in (BASE/'states').glob('*.json')}
 inventory=read_json(BASE/'inventory.json')
 result.update(cached_remaining=sum(bool(r['existing_paths']) and not states.get(r['source_video_id'],{}).get('status','').startswith(('ACCEPTED','REJECTED')) for r in inventory),
  current_run_outcomes=dict(__import__('collections').Counter(r['status'] for r in states.values() if not r.get('inherited_from'))),
  network_candidates_remaining=sum(not r['existing_paths'] and not states.get(r['source_video_id'],{}).get('status','').startswith(('ACCEPTED','REJECTED')) for r in inventory))
 atomic_json(OUT/'progress.json',result)
 ledger=OUT/'execution_progress.jsonl'
 last=json.loads(ledger.read_text(encoding='utf-8').splitlines()[-1]) if ledger.exists() else None
 if last is None or last['new_accepted']!=result['new_accepted'] or last['cached_remaining']!=result['cached_remaining']:
  with ledger.open('a',encoding='utf-8') as stream:stream.write(json.dumps({**result,'observed_epoch':time.time(),'accepted_manifest_sha256':digest(OUT/'accepted_annotations.jsonl')},sort_keys=True)+'\n');stream.flush();os.fsync(stream.fileno())
 return result

def overview(sid,cache,index):
 old=PREVIOUS/'evidence'/sid/'overview.jpg'
 if old.exists():return old
 from PIL import Image,ImageDraw
 p=BASE/'evidence'/sid/'overview.jpg';p.parent.mkdir(parents=True,exist_ok=True)
 if not p.exists():
  frames=index['frames'];canvas=Image.new('RGB',(1280,166*((len(frames)+4)//5)),'#171717');draw=ImageDraw.Draw(canvas)
  for n,f in enumerate(frames):
   x=(n%5)*256;y=(n//5)*166
   with Image.open(cache/sid/f['path']) as im:
    im.thumbnail((250,140));canvas.paste(im,(x+3,y+3))
   draw.text((x+4,y+146),str(f['timestamp'])+'s',fill='white')
  canvas.save(p,quality=90)
 return p

def freeze():
 """Seal the exact acquisition corpus after verifying every original source binding."""
 from collections import Counter
 from data_process.validation.temporal import validate_windows

 config=init()
 summary=progress()
 if not summary['complete']:
  raise ValueError('D1000 freeze requires exactly 500 accepted additions')
 parent_path=Path(config['parent'])
 parent_manifest=read_json(parent_path.with_name('dataset_manifest.json'))
 historical=load_jsonl(parent_path)
 additions=load_jsonl(OUT/'accepted_annotations.jsonl')
 pins=parent_manifest['source_video_sha256']
 assert len(historical)==500 and set(pins)=={r['qid'] for r in historical}
 source_ids=set()
 source_hashes={}
 for row in historical:
  pin=pins[row['qid']]
  filename=pin['video_filename']
  assert filename==Path(filename).name==row['metadata']['video_filename']
  assert digest(ROOT/'local_data/raw/adsqa/videos'/filename)==pin['sha256'],filename
  source_ids.add(Path(filename).stem)
  source_hashes[row['vid']]=pin['sha256']

 for row in additions:
  sid=row['source_video_id']
  assert sid not in source_ids and 0<row['duration']<=150
  assert row['human_verified'] is False and row.get('research_gold',False) is False
  assert row['review_provenance']=='AI_PSEUDO_LABELED' and row['verifier']['pass'] is True
  assert row['split_eligibility'] in (
   'RESTRICTED_PENDING_CAMPAIGN_GROUP_REVIEW','TRAIN_CANDIDATE_ONLY_CAMPAIGN_UNRESOLVED')
  assert not query_errors(row['query']) and not interval_errors(*row['window'],row['duration'])
  for path_key,hash_key in [
   ('source_video_path','source_video_sha256'),
   ('semantic_artifact_path','semantic_artifact_sha256'),
   ('verification_image_path','verification_image_sha256'),
  ]:
   path=Path(row[path_key])
   if not path.is_absolute():
    path=BASE.parents[2]/path if (BASE.parents[2]/path).is_file() else ROOT/path
   assert digest(path)==row[hash_key],sid
  source_ids.add(sid)
  source_hashes['lumae_ads_'+sid]=row['source_video_sha256']

 new_rows=[canonical_row(r,r['qid']) for r in additions]
 all_rows=historical+new_rows
 assert len(all_rows)==len(source_ids)==len(source_hashes)==len(set(source_hashes.values()))==1000
 assert len({r['qid'] for r in all_rows})==len({r['vid'] for r in all_rows})==1000
 assert all(r['query'].strip() and validate_windows(r['relevant_windows'],r['duration'])[0] for r in all_rows)
 splits={name:parent_path.with_name('lumae_ads_d500_'+name+'.jsonl') for name in ('train','val','test')}
 train,val,test=[load_jsonl(splits[name]) for name in splits]
 assert (len(train),len(val),len(test))==(440,30,30)
 assert {r['vid'] for r in train+val+test}=={r['vid'] for r in historical}
 assert not {r['vid'] for r in train+new_rows}&{r['vid'] for r in val+test}

 states={p.stem:read_json(p) for p in (BASE/'states').glob('*.json')}
 network_ids=set()
 batches=[]
 for folder in sorted((BASE/'network').glob('batch_*')):
  selection=folder/'source_selection.json'
  report=folder/'bulk_complete.json'
  ids={r['source_video_id'] for r in read_json(selection)['candidates']}
  assert not network_ids & ids
  network_ids.update(ids)
  metrics=read_json(report)
  outcomes=Counter(states.get(sid,{}).get('status','PENDING_REVIEW') for sid in ids)
  batches.append({
   'batch':folder.name,'candidates_examined':len(ids),'downloads':metrics['successful_downloads'],
   'outcomes':dict(outcomes),'free_bytes_after_downloads':metrics['current_free_bytes'],
   'selection_sha256':digest(selection),'p1_metrics_sha256':digest(report),
  })
 cached_outcomes=Counter(
  r['status'] for sid,r in states.items() if sid not in network_ids and not r.get('inherited_from'))
 collection={
  'seed_additions':len(load_jsonl(Path(config['prior']))),'cached_outcomes':dict(cached_outcomes),
  'network_batches':batches,'network_candidates_remaining':summary['network_candidates_remaining'],
 }

 folder=OUT/'lumae_ads_d1000'
 folder.mkdir(parents=True,exist_ok=True)
 canonical=folder/'lumae_ads_d1000_all.jsonl'
 jsonl(canonical,all_rows)
 jsonl(folder/'provisional_train.jsonl',train+new_rows)
 shutil.copyfile(splits['val'],folder/'historical_val.jsonl')
 shutil.copyfile(OUT/'accepted_annotations.jsonl',folder/'accepted_annotations.jsonl')
 shutil.copyfile(OUT/'execution_progress.jsonl',folder/'execution_progress.jsonl')
 manifest={
  'dataset_version':'lumae_ads_d1000','run_id':BASE.name,
  'dataset_sha256':digest(canonical),'historical_canonical_sha256':digest(parent_path),
  'historical_split_sha256':{name:digest(path) for name,path in splits.items()},
  'counts':{'total':1000,'historical':500,'additions':500,'provisional_train':940,'historical_val':30,'historical_test':30},
  'historical_raw_videos_verified':500,'addition_raw_videos_verified':500,
  'source_video_sha256':source_hashes,'additions_human_verified':0,
  'acquisition_complete':True,'campaign_disjoint_certified':False,'benchmark_ready':False,
  'accepted_provenance_sha256':digest(folder/'accepted_annotations.jsonl'),
  'semantic_v3_algorithm_sha256':parent_manifest['semantic_v3_algorithm_sha256'],
  'collection':collection,
 }
 atomic_json(folder/'dataset_manifest.json',manifest)
 (folder/'checksums.sha256').write_text(''.join(
  f'{digest(p)}  {p.name}\n' for p in sorted(folder.iterdir()) if p.is_file() and p.name!='checksums.sha256'
 ),encoding='utf-8')
 return {'acquisition_complete':True,'sources':len(source_ids),'manifest':str(folder/'dataset_manifest.json')}

def features():
 """Original pinned Moment-DETR extractor on CPU; never re-encode D500."""
 import numpy as np
 import torch
 from data_process.lumae_scale.cache_validation import atomic_npz
 torch.set_num_threads(2)
 config=init()
 rows=accepted()
 src=BASE/'feature_extractor_source';package=src/'run_on_video';(package/'clip').mkdir(parents=True,exist_ok=True)
 for name,target in [('data_utils','data_utils.py'),('clip_clip','clip/clip.py'),('clip_model','clip/model.py'),('clip_simple_tokenizer','clip/simple_tokenizer.py')]:
  data=read_json(ROOT/'audit_outputs/2026-11-02'/('upstream_run_on_video_'+name+'.py.json'))
  data=data.get('structuredContent',data);content=data['content'].encode()
  assert hashlib.sha1(b'blob '+str(len(content)).encode()+b'\0'+content).hexdigest()==data['sha']
  assert 'b7e553ac3b0c898ee6b85e03ee507c064eab89ca' in data['display_url']
  p=package/target
  if p.exists():assert p.read_bytes()==content
  else:p.write_bytes(content)
 (package/'__init__.py').touch(exist_ok=True)
 (package/'clip/__init__.py').write_text('from .clip import *\n')
 import clip
 bpe=Path(clip.__file__).parent/'bpe_simple_vocab_16e6.txt.gz';dest=package/'clip'/bpe.name
 if not dest.exists():shutil.copyfile(bpe,dest)
 assert digest(dest)==digest(bpe)
 dependencies=BASE/'dependencies' if (BASE/'dependencies').is_dir() else PREVIOUS/'dependencies'
 sys.path.insert(0,str(dependencies));sys.path.insert(0,str(src))
 from run_on_video.data_utils import ClipFeatureExtractor
 weights=ROOT/'local_data/cache/clip_weights/ViT-B-32.pt'
 assert digest(weights)=='40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af'
 extractor=ClipFeatureExtractor(framerate=.5,size=224,centercrop=True,model_name_or_path=str(weights),device='cpu')
 root=BASE/'training_features';index={'video':{},'text':{},'checksums':{},'feature_contract':read_json(ROOT/'research_outputs/p0_20261103_01/assets/feature_cache/feature_manifest.json')['feature_contract'],'encoder_weights_sha256':digest(weights),'extractor_commit':'b7e553ac3b0c898ee6b85e03ee507c064eab89ca','historical_features_reencoded':0,'gold_ready':False}
 manifest_path=OUT/'new_feature_manifest.json'
 published_pairs=0
 if config.get('base_count'):
  index['historical_binding']=config['feature_binding']
  if manifest_path.exists():
   published_pairs=len(read_json(manifest_path)['video'])
 # The seed additions already have immutable training NPZ in PREVIOUS.
 # This run owns only accepted(), so never encode the seed additions again.
 for r in rows:
  sid=r['source_video_id'];qid=r['qid'];vid='lumae_ads_'+sid
  p=Path(r['source_video_path'])
  if not p.is_absolute():p=Path('E:/LUMAE')/p if (Path('E:/LUMAE')/p).exists() else ROOT/p
  assert digest(p)==r['source_video_sha256']
  vp=root/'clip_features'/(vid+'.npz');tp=root/'clip_text_features'/('qid'+qid+'.npz');binding=root/'bindings'/(sid+'.json')
  identity={'source_sha256':r['source_video_sha256'],'query':r['query'],'weights_sha256':index['encoder_weights_sha256'],'extractor_commit':index['extractor_commit'],'framerate':.5,'centercrop':True,'size':224}
  old=read_json(binding) if binding.exists() else None
  reusable=old and old['identity']==identity and vp.exists() and tp.exists() and digest(vp)==old['video_sha256'] and digest(tp)==old['text_sha256']
  if not reusable:
   if shutil.disk_usage(BASE).free<17*2**30:raise RuntimeError('Feature reserve limit')
   v=extractor.encode_video(str(p),bsz=8).float().cpu().numpy();t=extractor.encode_text([r['query']],bsz=1)[0].float().cpu().numpy()[:32]
   atomic_npz(vp,features=v);atomic_npz(tp,last_hidden_state=t)
   atomic_json(binding,{'identity':identity,'video_sha256':digest(vp),'text_sha256':digest(tp)})
  for kind,key,path,ident in [('video','features',vp,vid),('text','last_hidden_state',tp,qid)]:
   with np.load(path,allow_pickle=False) as z:
    a=z[key];assert set(z.files)=={key} and a.ndim==2 and a.shape[1]==512 and len(a)>0 and a.dtype==np.float32 and np.isfinite(a).all()
   index[kind][ident]=path.relative_to(root).as_posix();index['checksums'][path.relative_to(root).as_posix()]=digest(path)
  if len(index['video'])>published_pairs:
   atomic_json(manifest_path,index)
  print('FEATURE_READY '+sid,flush=True)
 if config.get('base_count'):
  atomic_json(manifest_path,index)
  return {'new_video_features':len(index['video']),'new_text_features':len(index['text']),'root':str(root),'historical_reencoded':0}
 historical=ROOT/'research_outputs/p0_20261103_01/assets/feature_cache'
 frozen=read_json(historical/'feature_manifest.json')
 index['historical_binding']={'root':str(historical),'manifest_sha256':digest(historical/'feature_manifest.json'),'video':len(frozen['video']),'text':len(frozen['text'])}
 prior_manifest=ROOT/'research_outputs/d1000_fasttrack_20261104_01/new_feature_manifest.json'
 prior_features=read_json(prior_manifest)
 index['prior_additions_binding']={'root':str(PREVIOUS/'training_features'),'manifest':str(prior_manifest),'manifest_sha256':digest(prior_manifest),'video':len(prior_features['video']),'text':len(prior_features['text'])}
 atomic_json(OUT/'new_feature_manifest.json',index)
 return {'new_video_features':len(index['video']),'new_text_features':len(index['text']),'root':str(root),'historical_reencoded':0,'d1000_ready':False}

def handoff():
 """Verify three existing runner bindings without copying historical/seed NPZ."""
 import numpy as np
 from experiments.M1_lumae_ads.cached_runner import dataset_type,feature_path
 summary=progress();new=read_json(OUT/'new_feature_manifest.json')
 seed_manifest=ROOT/'research_outputs/d1000_fasttrack_20261104_01/new_feature_manifest.json';seed=read_json(seed_manifest)
 assert digest(seed_manifest)==new['prior_additions_binding']['manifest_sha256']
 original=ROOT/'research_outputs/p0_20261103_01/assets/feature_cache'
 old=read_json(original/'feature_manifest.json')
 historical=BASE/'historical_training_features'
 staged=historical.is_dir()
 if staged:
  assert read_json(historical/'feature_manifest.json')==old
  assert read_json(OUT/'drive_d500_feature_manifest.json')==old
  refs=read_json(OUT/'drive_historical_feature_refs.json')
 else:historical=original
 config=read_json(ROOT/'experiments/M1_lumae_ads/config_d500_cached.json')
 cls=dataset_type(ROOT/'research_outputs/p0_20261103_01/upstream')
 additions=load_jsonl(OUT/'accepted_annotations.jsonl');jsonl(OUT/'accepted_additions_canonical.jsonl',[canonical_row(r,r['qid']) for r in additions])
 prior=load_jsonl(Path(read_json(BASE/'config.json')['prior']));current=accepted()
 assert {r['source_video_id'] for r in additions}=={r['source_video_id'] for r in prior+current}
 assert not {r['source_video_id'] for r in prior}&{r['source_video_id'] for r in current}
 for name,rows in [('prior_additions',prior),('current_additions',current)]:jsonl(OUT/(name+'_canonical.jsonl'),[canonical_row(r,r['qid']) for r in rows])
 assert set(new['video'])=={'lumae_ads_'+r['source_video_id'] for r in current} and set(new['text'])=={r['qid'] for r in current}
 assert set(seed['video'])=={'lumae_ads_'+r['source_video_id'] for r in prior} and set(seed['text'])=={r['qid'] for r in prior}
 parent_rows=load_jsonl(ROOT/'local_data/datasets/lumae_ads_d500/lumae_ads_d500_all.jsonl')
 missing=[r['vid'] for r in parent_rows if not (historical/old['video'][r['vid']]).exists() or not (historical/old['text'][r['qid']]).exists()]
 jsonl(OUT/'historical_available_for_loader.jsonl',[r for r in parent_rows if r['vid'] not in missing])
 checks={};logical_checks={};loaded={};drive_count=local_count=0
 prefixes={'D500_available_subset':'','prior_additions':'additions_fasttrack_20261104/','additions':'additions_agent_20261104/'}
 for name,annotations,root,manifest in [('D500_available_subset',OUT/'historical_available_for_loader.jsonl',historical,old),('prior_additions',OUT/'prior_additions_canonical.jsonl',PREVIOUS/'training_features',seed),('additions',OUT/'current_additions_canonical.jsonl',BASE/'training_features',new)]:
  d=cls(annotations,root,manifest,config)
  for i in range(len(d)):
   x=d[i]['model_inputs'];assert x['video_feat'].shape[1]==514 and x['query_feat'].shape[1]==512
  loaded[name]=len(d)
  for kind in ('video','text'):
   for key,relative in manifest[kind].items():
    p=feature_path(root,relative)
    if not p.exists():continue
    h=digest(p)
    checks[str(p)]=h
    logical_checks[prefixes[name]+relative]=h
    with np.load(p,allow_pickle=False) as z:
     expected='features' if kind=='video' else 'last_hidden_state';a=z[expected]
     assert set(z.files)=={expected} and a.ndim==2 and a.shape[1]==512 and len(a)>0
     assert np.issubdtype(a.dtype,np.floating) and np.isfinite(a).all()
    if staged and name=='D500_available_subset':
     provenance=historical/'provenance'/kind/(key+'.json')
     if provenance.is_file():
      record=read_json(provenance);ref=refs[kind][key]
      assert record['sha256']==h and record['id']==ref['id']
      assert record['relative_path']==ref['relative_path']==relative
      assert record['bytes']==p.stat().st_size==int(ref['size'])
      assert record['historical_reencoded'] is False
      drive_count+=1
     else:
      assert digest(feature_path(original,relative))==h
      local_count+=1
  if name!='D500_available_subset':
   for relative,h in manifest['checksums'].items():assert digest(root/relative)==h
 dataset='lumae_ads_d1000' if summary['complete'] else 'lumae_ads_d1000_partial'
 unified={**old,'dataset':dataset,'canonical_sha256':digest(OUT/'provisional_dataset.jsonl'),'counts':{'video':summary['total'],'text':summary['total']},'local_bindings':{'historical_root':str(historical),'prior_additions_root':str(PREVIOUS/'training_features'),'prior_additions_logical_prefix':'additions_fasttrack_20261104/','additions_root':str(BASE/'training_features'),'additions_logical_prefix':'additions_agent_20261104/'},'d1000_complete':summary['complete']}
 for kind in ('video','text'):unified[kind]={**old[kind],**{k:'additions_fasttrack_20261104/'+v for k,v in seed[kind].items()},**{k:'additions_agent_20261104/'+v for k,v in new[kind].items()}}
 atomic_json(OUT/'unified_feature_manifest.json',unified)
 atomic_json(OUT/'feature_checksums.json',logical_checks)
 if staged:
  atomic_json(OUT/'historical_feature_materialization_progress.json',{
   'expected_pairs':len(parent_rows),'content_validation_complete':not missing,
   'materialized':{kind:sum((historical/p).is_file() for p in old[kind].values()) for kind in ('video','text')},
   'root':str(historical),'validated_drive_downloads':drive_count,'unchanged_local_files_copied':local_count,
   'historical_reencoded':0,'prior_historical_modified':False,
  })
 jsonl(OUT/'provisional_train.jsonl',load_jsonl(ROOT/'local_data/datasets/lumae_ads_d500/lumae_ads_d500_train.jsonl')+[canonical_row(r,r['qid']) for r in additions])
 config.update(dataset=dataset,canonical_sha256=unified['canonical_sha256'],counts={'train':440+len(additions),'val':30},split_sha256={'train':digest(OUT/'provisional_train.jsonl'),'val':digest(ROOT/'local_data/datasets/lumae_ads_d500/lumae_ads_d500_val.jsonl')})
 atomic_json(OUT/'cached_runner_config.json',config)
 result={'pass':not missing,'available_features_loader_pass':True,'existing_cached_dataset_loaded':loaded,'visual_input_dim_with_tef':514,'text_dim':512,'missing_new_accepted_features':0,'missing_historical_sources':missing,'missing_current_accepted_features':len(missing),'d1000_complete':summary['complete'],'historical_reencoded':0,'seed_features_reencoded':0,'file_sha256':checks,'unified_manifest_sha256':digest(OUT/'unified_feature_manifest.json')}
 atomic_json(OUT/'feature_validation.json',result)
 (OUT/'COLAB_HANDOFF.md').write_text(
  '# Cached feature handoff\n\n'
  +f"Accepted sources: {summary['total']}. Acquisition complete: {summary['complete']}. "
  +f"All local feature bindings available: {result['pass']}. Missing historical pairs: {len(missing)}.\n\n"
  +'Reuse the existing D500 feature root. Assemble the prior additions under additions_fasttrack_20261104/ '
  +'and current additions under additions_agent_20261104/ beneath the same logical feature root. '
  +'Their physical roots and prefixes are recorded in unified_feature_manifest.json. '
  +'Reuse existing NPZ; do not encode or copy historical/seed features unnecessarily. Raw videos are not required.\n\n'
  +'Local validation loads the three physical bindings separately with the unchanged CachedDataset. '
  +'Run its complete single-root preflight after materializing the logical layout in Colab. '
  +'Use cached_runner_config.json, provisional_train.jsonl, and the untouched D500 Val30; keep Test30 sealed. '
  +'Preserve accepted_annotations.jsonl with the AI-pseudo provenance and unresolved campaign restrictions. '
  +'No human-verified or campaign-disjoint benchmark is claimed. M0 and upstream identities remain in the cached runner config.\n\n'
  +'This handoff does not train or synchronize Drive. Resolve the Drive target and missing historical NPZ before training.\n\n'
  +'Resume collection: python scripts/d1000_agent_run.py run\n'
  +'Extract only missing current features: python scripts/d1000_agent_run.py features\n'
  +'Validate the three bindings: python scripts/d1000_agent_run.py handoff\n\n'
  +'Collection handoffs require the active agent to inspect actual overview and interval images; '
  +'the command does not continue unattended after exit.\n',encoding='utf-8')
 if summary['complete'] and not missing:
  result.update(package_handoff(config,unified,logical_checks))
  atomic_json(OUT/'feature_validation.json',result)
 return {k:v for k,v in result.items() if k not in ('file_sha256','missing_historical_sources')}

def package_handoff(config,manifest,checksums):
 """Run the unchanged cached runner against one logical root and package the additions."""
 from tempfile import TemporaryDirectory
 from zipfile import ZIP_DEFLATED,ZipFile
 from experiments.M1_lumae_ads import cached_runner

 frozen=OUT/'lumae_ads_d1000'
 assert read_json(frozen/'dataset_manifest.json')['dataset_sha256']==manifest['canonical_sha256']
 train=frozen/'provisional_train.jsonl'
 val=frozen/'historical_val.jsonl'
 assert len(checksums)==2000
 p0=ROOT/'research_outputs/p0_20261103_01'
 upstream=p0/'upstream'
 checkpoint=p0/'assets/M0_pretrained.ckpt'
 reference=read_json(OUT/'m0_reference.json')
 assert reference['verified_drive_bytes'] and reference['sha256']==config['m0_sha256']
 bindings=manifest['local_bindings']
 roots={
  bindings['prior_additions_logical_prefix']:Path(bindings['prior_additions_root']),
  bindings['additions_logical_prefix']:Path(bindings['additions_root']),
 }
 historical=Path(bindings['historical_root'])
 added={}
 with TemporaryDirectory(prefix='d1000_preflight_',dir=BASE) as directory:
  cache=Path(directory).resolve()
  assert cache.parent==BASE.resolve()
  for relative,checksum in sorted(checksums.items()):
   prefix,root=next(((p,r) for p,r in roots.items() if relative.startswith(p)),('',historical))
   source=cached_runner.feature_path(root,relative.removeprefix(prefix))
   target=cached_runner.feature_path(cache,relative)
   target.parent.mkdir(parents=True,exist_ok=True)
   shutil.copyfile(source,target)
   assert digest(target)==checksum
   if prefix:added['features/'+relative]=source
  _,identity,exceptions=cached_runner.preflight(
   train,val,OUT/'unified_feature_manifest.json',cache,upstream,checkpoint,config)
 assert all(checksums[p]==h for p,h in identity['feature_sha256'].items())
 assert len(identity['feature_sha256'])==1940
 atomic_json(OUT/'colab_preflight.json',{'status':'PASS','identity':identity,'dtype_exceptions':exceptions})

 instructions=OUT/'COLAB_HANDOFF.md'
 instructions.write_text(
  '# LUMAE D1000 cached feature handoff\n\n'
  'Extract the ZIP into a fresh directory. Mount the Drive location holding the existing '
  'historical features. From the extracted directory, point DRIVE_ARTIFACTS at the mounted '
  'folder containing m1_d200_features and m1_d500_features. Stage the two feature folders below. '
  'The additions are already in the ZIP. feature_checksums.json pins all 2,000 tensors.\n\n'
  '[Existing Drive feature folders](https://drive.google.com/drive/folders/1cTUQP7DVdexC89FJWeWUNoncbBRVhrQo)\n\n'
  '```sh\n'
  'mkdir -p features/artifacts\n'
  'cp -r "$DRIVE_ARTIFACTS/m1_d200_features" "$DRIVE_ARTIFACTS/m1_d500_features" features/artifacts/\n'
  '```\n\n'
  f"Place the [verified M0 checkpoint]({reference['url']}) at M0_pretrained.ckpt. "
  'Keep the runner and its configuration together. The pinned upstream snapshot and local preflight '
  'environment are included. With PyTorch, NumPy, SciPy, pandas and tqdm available, run from the extracted directory.\n\n'
  '```sh\n'
  'PYTHONPATH="$PWD" python experiments/M1_lumae_ads/cached_runner.py preflight '
  '--train-jsonl dataset/provisional_train.jsonl --val-jsonl dataset/historical_val.jsonl '
  '--feature-manifest unified_feature_manifest.json --feature-root features --upstream-root upstream '
  '--checkpoint M0_pretrained.ckpt --output preflight.json\n'
  '```\n\n'
  'Compare the feature checksums returned by preflight against feature_checksums.json '
  'before starting a training run. Drive references for individual files are included.\n\n'
  'The verified split is provisional Train940 with historical Val30. Campaign grouping is unresolved '
  'for the additions. Historical Test30 stays outside the package and outside training selection.\n',
  encoding='utf-8')
 runner=ROOT/'experiments/M1_lumae_ads'
 files={
  **added,
  **{'dataset/'+p.name:p for p in (train,val,frozen/'dataset_manifest.json',frozen/'accepted_annotations.jsonl')},
  **{name:OUT/name for name in ('unified_feature_manifest.json','feature_checksums.json','drive_historical_feature_refs.json',
                               'm0_reference.json','colab_preflight.json','COLAB_HANDOFF.md')},
  'experiments/M1_lumae_ads/cached_runner.py':runner/'cached_runner.py',
  'experiments/M1_lumae_ads/config_d500_cached.json':OUT/'cached_runner_config.json',
  'upstream/SOURCE_MANIFEST.json':upstream/'SOURCE_MANIFEST.json',
  'data_process/__init__.py':ROOT/'data_process/__init__.py',
 }
 for item in read_json(upstream/'SOURCE_MANIFEST.json')['files']:
  files['upstream/'+item['path']]=upstream/item['path']
 for directory in ('data_process/validation','data_process/schemas'):
  for path in (ROOT/directory).glob('*.py'):
   files[path.relative_to(ROOT).as_posix()]=path
 archive=OUT/'lumae_d1000_colab.zip'
 temporary=archive.with_suffix('.zip.tmp')
 with ZipFile(temporary,'w',compression=ZIP_DEFLATED) as z:
  for name,path in sorted(files.items()):
   z.write(path,name)
 with ZipFile(temporary) as z:
  assert z.testzip() is None and len(z.namelist())==len(files)
 os.replace(temporary,archive)
 return {'single_root_preflight_pass':True,'package':str(archive),'package_sha256':digest(archive)}

def run(prepare=False,propose=False):
 c=init();summary=progress();runtime=None
 if summary['complete']:return summary
 parent_hashes=c.get('parent_hashes')
 if parent_hashes is None:
  parent=read_json(ROOT/'local_data/datasets/lumae_ads_d500/dataset_manifest.json')
  parent_hashes=[r['sha256'] for r in parent['source_video_sha256'].values()]
 hashes=set(parent_hashes)|{r['source_video_sha256'] for r in prior_annotations(c)+accepted()}
 queue=[]
 for row in read_json(BASE/'inventory.json'):
  sid=row['source_video_id'];state=BASE/'states'/(sid+'.json')
  if propose and not (BASE/'observations'/(sid+'.json')).exists():continue
  if state.exists() and read_json(state).get('status','').startswith(('ACCEPTED','REJECTED')):continue
  if (BASE/'accepted'/(sid+'.json')).exists():continue
  if shutil.disk_usage(BASE).free<15*2**30:return {'status':'BLOCKED_STORAGE','progress':progress()}
  if not row['existing_paths']:
   if prepare or propose:break
   network_batch()
   return run()
  path=Path(row['existing_paths'][0]);h=digest(path)
  if h in hashes:atomic_json(state,{'status':'REJECTED_DUPLICATE','sha256':h});continue
  probe=probe_video_file(path);duration=probe.get('duration') or 0
  if not probe.get('is_model_compatible') or not 0<duration<=150:
   atomic_json(state,{'status':'REJECTED_TECHNICAL','probe':probe});continue
  cache=None
  candidates=[ROOT/'local_data/cache/lumae_visual_index',Path('E:/LUMAE/cache/p1_20261103_01/visual_index'),Path('E:/LUMAE/cache/p1_p2_20261103_01/visual_index'),BASE/'cache']
  for root in candidates:
   m=root/sid/'metadata.json'
   if m.exists() and visual_index_valid(read_json(m),root/sid,h) and clip_npz_valid(root/sid/'clip_features.npz',read_json(m)):
    cache=root;break
  if cache is None:
   cache=BASE/'cache'
   previous=PREVIOUS/'cache'/sid/'metadata.json'
   if not (cache/sid/'metadata.json').exists() and previous.exists() and visual_index_valid(read_json(previous),previous.parent,h):
    # Bind existing immutable JPEG/contact sheets; write only new NPZ and metadata.
    saved=read_json(previous)
    saved['frames']=[{**f,'path':str(previous.parent/f['path'])} for f in saved['frames']]
    saved['contact_sheets']=[str(previous.parent/x) for x in saved['contact_sheets']]
    from data_process.lumae_scale.stage_cache import frame_fingerprint
    saved['visual_fingerprint']=frame_fingerprint(saved)
    saved['clip_features_status']='pending'
    saved.pop('clip_features_sha256',None);saved.pop('clip_fingerprint',None)
    atomic_json(cache/sid/'metadata.json',saved)
   try:build_visual_index(path,duration,h,work=BASE,strict=True,visual_cache=cache)
   except ValueError as exc:
    if not c.get('base_count'):raise
    atomic_json(state,{'status':'REJECTED_TECHNICAL','reason':str(exc),'probe':probe,'sha256':h})
    continue
  index=read_json(cache/sid/'metadata.json');op=overview(sid,cache,index)
  eventpath=BASE/'observations'/(sid+'.json')
  if prepare:
   queue.append({'source_id':sid,'overview':str(op),'duration':duration,'cache':str(cache),'overview_sha256':digest(op)})
   if len(queue)==25:break
   continue
  if not eventpath.exists():
   if propose:continue
   atomic_json(state,{'status':'PENDING_REVIEW','stage':'EVENTS','source':str(path),'cache':str(cache),'overview':str(op),'overview_sha256':digest(op),'duration':duration,'source_sha256':h})
   return {'status':'AWAITING_VISUAL_INSPECTION','source_id':sid,'overview':str(op),'duration':duration,'progress':summary}
  note=read_json(eventpath)
  assert note['overview_sha256']==digest(op) and note.get('inspection_method')=='agent_viewed_overview'
  if not note['events']:
   atomic_json(state,{'status':'REJECTED_ANNOTATION','reason':note['reason']});continue
  events=validate_events(sid,note['events'],duration,visual_cache=cache)
  if runtime is None:
   import torch
   torch.set_num_threads(2)
   from data_process.lumae_scale.clip_runtime import ClipRuntime
   runtime=ClipRuntime(str(ROOT/'local_data/cache/clip_weights/ViT-B-32.pt'),device='cpu',download_root=BASE/'models')
  if cache==BASE/'cache':extract_clip_features(sid,strict=True,visual_cache=cache,clip_runtime=runtime,batch_size=8)
  for i,event in enumerate(events[:3]):
   query=event['query']
   if query_errors(query):continue
   tag=hashlib.sha256(query.encode()).hexdigest();pp=BASE/'predictions'/sid/(tag+'.json')
   if not pp.exists():
    try:atomic_json(pp,localize(sid,query,events,duration,i,clip_runtime=runtime,visual_cache=cache))
    except ValueError as exc:
     atomic_json(BASE/'localization_failures'/sid/(tag+'.json'),{'query':query,'reason':str(exc),'source_level_failure':True})
     continue
   pred=read_json(pp)
   from data_process.lumae_scale import pipeline
   pipeline.VISUAL_CACHE=cache
   image,times=pipeline._verification_image(sid,pred,event,work=BASE)
   automatic=verify(query=query,prediction=pred,duration=duration,verification_image=image,evidence_timestamps=times)
   route=route_candidate(query=query,prediction=pred,duration=duration,visual_evidence={'timestamps':event['evidence_timestamps'],'contact_sheets':event['evidence_sheets']},automatic_verifier=automatic)
   if route['route']=='LOW':continue
   vp=BASE/'verdicts'/sid/(tag+'.json')
   if not vp.exists():
    atomic_json(state,{'status':'PENDING_REVIEW','stage':'VERIFY','source':str(path),'cache':str(cache),'prediction':str(pp),'image':str(image),'image_sha256':digest(image),'evidence_timestamps':times,'query':query})
    packet={'status':'AWAITING_INTERVAL_VERIFICATION','source_id':sid,'query':query,'prediction':pred,'image':str(image),'verdict_path':str(vp),'image_sha256':digest(image),'evidence_timestamps':times}
    if propose:queue.append(packet);break
    return packet
   v=read_json(vp)
   assert v['image_sha256']==digest(image) and v['prediction_sha256']==digest(pp) and v['evidence_timestamps']==times
   if not v.get('pass') or not v.get('notes'):continue
   record={'qid':c.get('qid_prefix','d1000_compat_')+sid,'source_video_id':sid,'source_video_path':str(path),'source_video_sha256':h,'source_url':row['source_url'],'duration':duration,'query':query,'window':[pred['start'],pred['end']],
    'query_sanitation':{'passed':True,'errors':[]},'approximate_event_phase_seconds':[event['coarse_start'],event['coarse_end']],'visual_evidence':{'timestamps':event['evidence_timestamps'],'contact_sheets':event['evidence_sheets']},'visual_index_sha256':digest(cache/sid/'metadata.json'),
    'verification_image_path':str(image),'verification_image_sha256':digest(image),'semantic_artifact_path':str(pp),'semantic_artifact_sha256':digest(pp),'semantic_v3_prediction_sha256':digest(pp),'semantic_v3_algorithm_sha256':pred['semantic_v3_algorithm_sha256'],
    'query_source':'AI_GENERATED','query_review_status':'AI_AUTO_ACCEPTED_FOR_TRAINING','temporal_label_source':'SEMANTIC_V3_PSEUDO_LABEL','review_provenance':'AI_PSEUDO_LABELED','reviewer_id':'AI_PIPELINE','query_provider_config':{'provider':'Codex actual image inspection','model_revision':'NOT_EXPOSED'},'verifier':v,'automatic_verifier':automatic,'router':route,'human_verified':False,
    'annotation_policy':POLICY,'status':'ACCEPTED_D500_COMPAT_AI_PSEUDO','research_gold':False,'split_eligibility':'TRAIN_CANDIDATE_ONLY_CAMPAIGN_UNRESOLVED','campaign_group_id':'UNKNOWN_'+sid,'cache_root':str(cache),'raw_prediction_preserved':True,'observation_sha256':digest(eventpath),'verdict_sha256':digest(vp)}
   if c.get('base_count'):
    record['split_eligibility']='CAMPAIGN_UNKNOWN'
    record['campaign_group_id']=row.get('campaign_group_id') or 'UNKNOWN_'+sid
    record['source_metadata']={k:v for k,v in row.items() if k!='existing_paths'}
    provenance=path.parent.parent/'records'/(sid+'.json')
    record['acquisition']={'method':'P1' if provenance.is_file() else 'LOCAL_REUSE','source_url':row['source_url']}
    if provenance.is_file():record['acquisition'].update(path=str(provenance),sha256=digest(provenance))
   validate_annotation(record,visual_cache=cache)
   atomic_json(BASE/'accepted'/(sid+'.json'),record);hashes.add(h);atomic_json(state,{'status':record['status'],'record_sha256':digest(BASE/'accepted'/(sid+'.json'))})
   summary=progress();print(json.dumps(summary),flush=True)
   if summary['complete']:return summary
   break
  else:atomic_json(state,{'status':'REJECTED_ANNOTATION','reason':'All bounded D500-compatible query candidates failed'})
 if prepare or propose:
  if propose:
   atomic_json(BASE/'verification_queue.json',queue)
   return {'status':'PROPOSED_PENDING_ACTUAL_VISUAL_VERDICTS','count':len(queue)}
  atomic_json(BASE/'review_queue.json',queue)
  return {'status':'PREPARED_CACHED_REVIEW_QUEUE','count':len(queue),'queue':queue}
 return {'status':'SOURCE_POOL_EXHAUSTED','progress':progress()}

if __name__=='__main__':
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('command',choices=['init','run','status','prepare','propose','freeze','features','handoff']);a=ap.parse_args()
 from data_process.lumae_scale.acquisition import run_lock
 lock='feature_extraction.lock' if a.command=='features' else 'controller.lock'
 with run_lock(BASE/lock):
  actions={'init':init,'status':progress,'freeze':freeze,'features':features,'handoff':handoff}
  action=actions.get(a.command)
  result=action() if action else run(a.command=='prepare',a.command=='propose')
  print(json.dumps(result,ensure_ascii=True,indent=2))
