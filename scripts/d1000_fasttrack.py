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

BASE=Path('E:/LUMAE/work/d1000/d1000_fasttrack_20261104_01')
OUT=ROOT/'research_outputs/d1000_fasttrack_20261104_01'
P21=ROOT/'research_outputs/p21_20261103_01'
POLICY='D500_COMPAT_20261104_v1'

def jsonl(path,rows):
 path.parent.mkdir(parents=True,exist_ok=True)
 data=''.join(json.dumps(r,sort_keys=True,ensure_ascii=False,allow_nan=False)+'\n' for r in rows)
 tmp=path.with_suffix('.tmp');tmp.write_text(data,encoding='utf-8');os.replace(tmp,path)

def init():
 BASE.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
 parent=ROOT/'local_data/datasets/lumae_ads_d500';manifest=read_json(parent/'dataset_manifest.json')
 canonical=parent/'lumae_ads_d500_all.jsonl'
 assert digest(canonical)==manifest['dataset_sha256']
 prior=load_jsonl(P21/'manifests/accepted_additions_pool.jsonl')
 assert len(prior)==len({r['source_video_id'] for r in prior})==23
 seal=read_json(P21/'manifests/p21_execution_manifest.json')
 for name,h in seal['artifact_sha256'].items():assert digest(P21/name)==h,name
 protected={str(p):digest(p) for p in parent.iterdir() if p.is_file()}
 protected.update({str(ROOT/name):digest(ROOT/name) for name in ['data_process/lumae_scale/temporal_labeler.py','data_process/lumae_scale/freeze.py','data_process/lumae_scale/quality_gate.py','data_process/lumae_scale/router.py','data_process/lumae_scale/visual_index.py']})
 identity={'policy':POLICY,'parent':str(canonical),'parent_sha256':digest(canonical),'prior':str(P21/'manifests/accepted_additions_pool.jsonl'),'prior_sha256':digest(P21/'manifests/accepted_additions_pool.jsonl'),'protected':protected,'target_new':500,'min_free_bytes':15*2**30,'batch_size':25,'query_attempts':3,'sampling_step':2,'wrapper':WRAPPER_VERSION,'research_gold':False}
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
 return identity

def accepted():
 records=[]
 for p in sorted((BASE/'accepted').glob('*.json')):
  r=read_json(p)
  assert r['annotation_policy']==POLICY and r['human_verified'] is False
  validate_annotation(r,visual_cache=Path(r['cache_root']))
  state=BASE/'states'/(r['source_video_id']+'.json')
  if state.exists() and read_json(state).get('record_sha256'):
   assert read_json(state)['record_sha256']==digest(p),'Accepted record changed'
  records.append(r)
 return records

def network_batch():
 """New bounded batches, same existing P1 crawler; no sealed pilot is resumed."""
 from data_process.lumae_scale.acquisition import Pilot,run_lock
 from data_process.lumae_scale.builder import BuildConfig
 inventory=read_json(BASE/'inventory.json')
 roots=read_json(P21/'candidate_ranking/next_pool_cache_audit.json')['registered_roots']
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
 config=read_json(P21/'benchmarks/fresh_wave_config.json')
 config.update(run_id=f'fasttrack_20261104_b{number:04d}',mode='pilot',target_accepted=25,max_candidates=25,previous_runs=[],engineering_gate=None,operational_gate=str(OUT/'network_gate.json'))
 config['source_permission']={**config['source_permission'],'scope':'Continuous D1000 training collection, bounded25 internal batches until500 additional accepted sources; current user request f9454685-fc3e-402c-b608-f755f0e11597','authorization_attachment':'f9454685-fc3e-402c-b608-f755f0e11597/Pasted text.txt sections7,9'}
 tests=read_json(OUT/'targeted_tests.json')
 if not tests.get('pass'):raise RuntimeError('Fast-track targeted tests must pass before network execution')
 atomic_json(OUT/'network_gate.json',{'protected_assets_pass':bool(progress()),'storage_pass':shutil.disk_usage(BASE).free>20*2**30,'source_permissions_attested':True,'crawler_tests_pass':True,'source_manifest_sha256':config['source_manifest_sha256']})
 build=BuildConfig.load(ROOT/config['builder_config'],ROOT)
 pilot=Pilot(config,build,read_json(ROOT/config['source_manifest'])['videos'],live=True)
 pilot.work=folder;pilot.plan.update(candidates=pending,previous_accepted_sources=0,scope='D1000_COMPAT_INTERNAL_BATCH')
 metrics=pilot.run(max_wave=25)
 for record in pilot.records():
  sid=record['source_video_id'];p=folder/'downloads'/(sid+'.mp4')
  if p.exists():
   for r in inventory:
    if r['source_video_id']==sid:r['existing_paths']=[str(p)]
  else:atomic_json(BASE/'states'/(sid+'.json'),{'status':'REJECTED_TECHNICAL','reason':record.get('reason'),'acquisition_record':str(folder/'records'/(sid+'.json'))})
 atomic_json(BASE/'inventory.json',sorted(inventory,key=lambda r:(not bool(r['existing_paths']),r['source_video_id'])))
 if len(pilot.records())==len(pending):atomic_json(folder/'bulk_complete.json',metrics)
 else:raise RuntimeError('Acquisition checkpoint: '+metrics['stop_reason'])

def progress():
 c=read_json(BASE/'config.json');prior=load_jsonl(Path(c['prior']));new=prior+accepted()
 assert len(new)==len({r['source_video_id'] for r in new})==len({r['source_video_sha256'] for r in new})
 assert len(new)==len({r['qid'] for r in new}) and len(new)<=c['target_new']
 parent_rows=load_jsonl(Path(c['parent']))
 assert not {r['vid'].removeprefix('lumae_ads_') for r in parent_rows}&{r['source_video_id'] for r in new}
 for p,h in c['protected'].items():assert digest(Path(p))==h,'Protected state changed: '+p
 assert digest(Path(c['prior']))==c['prior_sha256']
 result={'historical':500,'new_accepted':len(new),'total':500+len(new),'remaining':max(0,500-len(new)),'complete':len(new)==500,'free_bytes':shutil.disk_usage(BASE).free,'policy':POLICY,'human_verified_new':0}
 jsonl(OUT/'accepted_annotations.jsonl',new)
 jsonl(OUT/'provisional_dataset.jsonl',parent_rows+[canonical_row(r,r['qid']) for r in new])
 atomic_json(OUT/'progress.json',result)
 return result

def overview(sid,cache,index):
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

def features():
 """Original pinned Moment-DETR extractor on CPU; never re-encode D500."""
 import numpy as np
 import torch
 from data_process.lumae_scale.cache_validation import atomic_npz
 torch.set_num_threads(2)
 init();progress()
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
 sys.path.insert(0,str(BASE/'dependencies'));sys.path.insert(0,str(src))
 from run_on_video.data_utils import ClipFeatureExtractor
 weights=ROOT/'local_data/cache/clip_weights/ViT-B-32.pt'
 assert digest(weights)=='40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af'
 extractor=ClipFeatureExtractor(framerate=.5,size=224,centercrop=True,model_name_or_path=str(weights),device='cpu')
 root=BASE/'training_features';index={'video':{},'text':{},'checksums':{},'feature_contract':read_json(ROOT/'research_outputs/p0_20261103_01/assets/feature_cache/feature_manifest.json')['feature_contract'],'encoder_weights_sha256':digest(weights),'extractor_commit':'b7e553ac3b0c898ee6b85e03ee507c064eab89ca','historical_features_reencoded':0,'gold_ready':False}
 for r in load_jsonl(OUT/'accepted_annotations.jsonl'):
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
  atomic_json(OUT/'new_feature_manifest.json',index)
  print('FEATURE_READY '+sid,flush=True)
 historical=ROOT/'research_outputs/p0_20261103_01/assets/feature_cache'
 frozen=read_json(historical/'feature_manifest.json')
 index['historical_binding']={'root':str(historical),'manifest_sha256':digest(historical/'feature_manifest.json'),'video':len(frozen['video']),'text':len(frozen['text'])}
 atomic_json(OUT/'new_feature_manifest.json',index)
 return {'new_video_features':len(index['video']),'new_text_features':len(index['text']),'root':str(root),'historical_reencoded':0,'d1000_ready':False}

def handoff():
 """Verify both existing runner input bindings without copying historical NPZ."""
 import numpy as np
 from experiments.M1_lumae_ads.cached_runner import dataset_type
 summary=progress();new=read_json(OUT/'new_feature_manifest.json')
 historical=ROOT/'research_outputs/p0_20261103_01/assets/feature_cache';old=read_json(historical/'feature_manifest.json')
 config=read_json(ROOT/'experiments/M1_lumae_ads/config_d500_cached.json')
 cls=dataset_type(ROOT/'research_outputs/p0_20261103_01/upstream')
 additions=load_jsonl(OUT/'accepted_annotations.jsonl');jsonl(OUT/'accepted_additions_canonical.jsonl',[canonical_row(r,r['qid']) for r in additions])
 parent_rows=load_jsonl(ROOT/'local_data/datasets/lumae_ads_d500/lumae_ads_d500_all.jsonl')
 missing=[r['vid'] for r in parent_rows if not (historical/old['video'][r['vid']]).exists() or not (historical/old['text'][r['qid']]).exists()]
 jsonl(OUT/'historical_available_for_loader.jsonl',[r for r in parent_rows if r['vid'] not in missing])
 checks={};loaded={}
 for name,annotations,root,manifest in [('D500_available_subset',OUT/'historical_available_for_loader.jsonl',historical,old),('additions',OUT/'accepted_additions_canonical.jsonl',BASE/'training_features',new)]:
  d=cls(annotations,root,manifest,config)
  for i in range(len(d)):
   x=d[i]['model_inputs'];assert x['video_feat'].shape[1]==514 and x['query_feat'].shape[1]==512
  loaded[name]=len(d)
  for kind in ('video','text'):
   for key,relative in manifest[kind].items():
    p=root/relative
    if not p.exists():continue
    checks[str(p)]=digest(p)
    with np.load(p,allow_pickle=False) as z:
     expected='features' if kind=='video' else 'last_hidden_state';a=z[expected]
     assert set(z.files)=={expected} and a.ndim==2 and a.shape[1]==512 and np.isfinite(a).all()
  if name=='additions':
   for relative,h in new['checksums'].items():assert digest(root/relative)==h
 unified={**old,'dataset':'lumae_ads_d1000_partial','canonical_sha256':digest(OUT/'provisional_dataset.jsonl'),'video':{**old['video'],**{k:'additions_20261104/'+v for k,v in new['video'].items()}},'text':{**old['text'],**{k:'additions_20261104/'+v for k,v in new['text'].items()}},'counts':{'video':summary['total'],'text':summary['total']},'local_bindings':{'historical_root':str(historical),'additions_root':str(BASE/'training_features'),'additions_logical_prefix':'additions_20261104/'},'d1000_complete':False}
 atomic_json(OUT/'unified_feature_manifest.json',unified)
 jsonl(OUT/'provisional_train.jsonl',load_jsonl(ROOT/'local_data/datasets/lumae_ads_d500/lumae_ads_d500_train.jsonl')+[canonical_row(r,r['qid']) for r in additions])
 config.update(dataset='lumae_ads_d1000_partial',canonical_sha256=unified['canonical_sha256'],counts={'train':440+len(additions),'val':30},split_sha256={'train':digest(OUT/'provisional_train.jsonl'),'val':digest(ROOT/'local_data/datasets/lumae_ads_d500/lumae_ads_d500_val.jsonl')})
 atomic_json(OUT/'cached_runner_config.json',config)
 result={'pass':not missing,'available_features_loader_pass':True,'existing_cached_dataset_loaded':loaded,'visual_input_dim_with_tef':514,'text_dim':512,'missing_new_accepted_features':0,'missing_historical_sources':missing,'missing_current_accepted_features':len(missing),'d1000_complete':False,'historical_reencoded':0,'file_sha256':checks,'unified_manifest_sha256':digest(OUT/'unified_feature_manifest.json')}
 atomic_json(OUT/'feature_validation.json',result)
 (OUT/'COLAB_HANDOFF.md').write_text('''# Partial collection handoff — D1000 is incomplete\n\nNo training or Drive write was performed. Reuse the existing D500 feature root bound in unified_feature_manifest.json; upload only E:/LUMAE/work/d1000/d1000_fasttrack_20261104_01/training_features into additions_20261104/ under that same feature root. Do not duplicate the historical500 NPZ pairs. The unified relative-path manifest resolves after these two directories are assembled; local validation loads the two existing bindings separately through the unchanged CachedDataset class, not through a claimed single-root preflight.\n\nUse cached_runner_config.json with provisional_train.jsonl and the untouched D500 Val30 file. Keep Test30 sealed; no gold/campaign-disjoint benchmark is claimed. M0/upstream identities and recipe are inherited from the existing cached runner. Review Drive target mapping/authenticated connection before synchronization. This is a partial verified local handoff, not a completed D1000 package.\n\nResume collection: python scripts/d1000_fasttrack.py run\nResume only missing compatible feature extraction: python scripts/d1000_fasttrack.py features\nValidate both local bindings: python scripts/d1000_fasttrack.py handoff\n\nThe collection command returns an actual overview or verification strip when the D500 visual agent must inspect it. It never invents a verdict, silently converts pending to accepted, or continues in the background after exit.\n''',encoding='utf-8')
 return {k:v for k,v in result.items() if k not in ('file_sha256','missing_historical_sources')}

def run(prepare=False,propose=False):
 c=init();summary=progress();runtime=None
 if summary['complete']:return summary
 prior=load_jsonl(Path(c['prior']));parent=read_json(ROOT/'local_data/datasets/lumae_ads_d500/dataset_manifest.json')
 hashes={r['sha256'] for r in parent['source_video_sha256'].values()}|{r['source_video_sha256'] for r in prior+accepted()}
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
  probe=probe_video_file(path);duration=probe.get('duration',0)
  if not probe.get('is_model_compatible') or not 0<duration<=150:
   atomic_json(state,{'status':'REJECTED_TECHNICAL','probe':probe});continue
  cache=None
  candidates=[ROOT/'local_data/cache/lumae_visual_index',Path('E:/LUMAE/cache/p1_20261103_01/visual_index'),Path('E:/LUMAE/cache/p1_p2_20261103_01/visual_index'),BASE/'cache']
  for root in candidates:
   m=root/sid/'metadata.json'
   if m.exists() and visual_index_valid(read_json(m),root/sid,h) and clip_npz_valid(root/sid/'clip_features.npz',read_json(m)):
    cache=root;break
  if cache is None:
   cache=BASE/'cache';build_visual_index(path,duration,h,work=BASE,strict=True,visual_cache=cache)
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
   if not pp.exists():atomic_json(pp,localize(sid,query,events,duration,i,clip_runtime=runtime,visual_cache=cache))
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
   record={'qid':'d1000_compat_'+sid,'source_video_id':sid,'source_video_path':str(path),'source_video_sha256':h,'source_url':row['source_url'],'duration':duration,'query':query,'window':[pred['start'],pred['end']],
    'query_sanitation':{'passed':True,'errors':[]},'approximate_event_phase_seconds':[event['coarse_start'],event['coarse_end']],'visual_evidence':{'timestamps':event['evidence_timestamps'],'contact_sheets':event['evidence_sheets']},'visual_index_sha256':digest(cache/sid/'metadata.json'),
    'verification_image_path':str(image),'verification_image_sha256':digest(image),'semantic_artifact_path':str(pp),'semantic_artifact_sha256':digest(pp),'semantic_v3_prediction_sha256':digest(pp),'semantic_v3_algorithm_sha256':pred['semantic_v3_algorithm_sha256'],
    'query_source':'AI_GENERATED','query_review_status':'AI_AUTO_ACCEPTED_FOR_TRAINING','temporal_label_source':'SEMANTIC_V3_PSEUDO_LABEL','review_provenance':'AI_PSEUDO_LABELED','reviewer_id':'AI_PIPELINE','query_provider_config':{'provider':'Codex actual image inspection','model_revision':'NOT_EXPOSED'},'verifier':v,'automatic_verifier':automatic,'router':route,'human_verified':False,
    'annotation_policy':POLICY,'status':'ACCEPTED_D500_COMPAT_AI_PSEUDO','research_gold':False,'split_eligibility':'TRAIN_CANDIDATE_ONLY_CAMPAIGN_UNRESOLVED','campaign_group_id':'UNKNOWN_'+sid,'cache_root':str(cache),'raw_prediction_preserved':True,'observation_sha256':digest(eventpath),'verdict_sha256':digest(vp)}
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
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('command',choices=['init','run','status','prepare','propose','features','handoff']);a=ap.parse_args()
 from data_process.lumae_scale.acquisition import run_lock
 with run_lock(BASE/'controller.lock'):
  print(json.dumps(init() if a.command=='init' else progress() if a.command=='status' else features() if a.command=='features' else handoff() if a.command=='handoff' else run(a.command=='prepare',a.command=='propose'),ensure_ascii=True,indent=2))
