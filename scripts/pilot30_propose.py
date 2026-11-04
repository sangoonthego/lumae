"""Offline P1.4 inference and dense QC, isolated from all old P1 artifacts."""
import csv,math,shutil,sys,time,subprocess
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from PIL import Image,ImageDraw
import numpy as np
from data_process.lumae_scale.models import ROOT,read_json,atomic_json,digest
from data_process.lumae_scale.visual_index import extract_clip_features
from data_process.lumae_scale.temporal_labeler import localize
from data_process.lumae_scale.clip_runtime import ClipRuntime

O=ROOT/'research_outputs/p14_p15_20261103_01';E=Path('E:/LUMAE/work/d1000/p14_p15_20261103_01');CACHE=E/'visual_index'
if (ROOT/'research_outputs/p14_p15_20261103_02/pilot30_freeze_manifest.json').exists():raise PermissionError('Evidence is a dependency of frozen pilot02; generation requires a new version')
rows=list(csv.DictReader((O/'pilot30_inventory.csv').open(encoding='utf-8-sig')))
observations=read_json(O/'visual_observations.json')
import torch
torch.set_num_threads(4)
weights=ROOT/'local_data/cache/clip_weights/ViT-B-32.pt';runtime=ClipRuntime(str(weights),device='cpu',download_root=E/'model_cache')
for n,brand,query,start,end,seen,subject,action,obj in observations:
 r=rows[n-1];sid=r['source_id'];out=E/'annotations'/sid;out.mkdir(parents=True,exist_ok=True)
 if (out/'proposal.json').exists():continue
 began=time.monotonic();old=Path('E:/LUMAE/cache/p1_20261103_01/visual_index')/sid
 shutil.copytree(old,CACHE/sid,dirs_exist_ok=True)
 event={'query':query,'coarse_start':start,'coarse_end':end,'evidence_timestamps':[seen],'visible_subject':subject,'visible_action':action,'visible_object':obj,'evidence_sheets':['overview'],'evidence_note':'Actual overview viewed in this chat; AI observation, not human label'}
 atomic_json(out/'generation_input.json',{'source_id':sid,'query':query,'event':event,'brand_observation':brand,'overview_sha256':digest(E/f'pilot30_{n:02d}_overview.jpg'),'procedure':'Codex genuine multimodal visual overview inspection','model_revision':'NOT_EXPOSED','human_verified':False})
 extract_clip_features(sid,clip_runtime=runtime,strict=True,visual_cache=CACHE)
 # Separate blind-first numerical reviewer receives query/features, not proposed bounds or coarse event.
 with np.load(CACHE/sid/'clip_features.npz') as z:
  vector=runtime.encode_text(query)[0];scores=z['features']@vector;peak=int(np.argmax(scores));timestamp=float(z['timestamps'][peak]);independent=[max(0,timestamp-2),min(float(r['duration']),timestamp+2)]
 atomic_json(out/'blind_first_estimate.json',{'source_id':sid,'query':query,'interval':independent,'procedure':'Full-video CLIP retrieval without generator coarse phase/boundaries; independent procedure, same CLIP weights','peak_similarity':float(scores[peak]),'weights_sha256':digest(weights),'human_review':False,'semantic_verification':'REQUIRES_SEPARATE_VISUAL_QC'})
 try:
  pred=localize(sid,query,[event],float(r['duration']),0,clip_runtime=runtime,visual_cache=CACHE)
  result={'source_id':sid,'query':query,'prediction':pred,'status':'PROPOSED_NOT_ACCEPTED','generation_seconds':time.monotonic()-began}
 except Exception as ex:result={'source_id':sid,'query':query,'status':'LOCALIZATION_FAILED','reason':str(ex)}
 atomic_json(out/'proposal.json',result)
 dense=out/'dense';dense.mkdir(exist_ok=True);step=.5 if end-start<=2 else 1
 times=np.arange(max(0,start-2),min(float(r['duration'])-.05,end+2)+.01,step).tolist();records=[]
 for j,t in enumerate(times):
  p=dense/f'{j:03d}.jpg'
  subprocess.run(['ffmpeg','-v','error','-ss',str(t),'-i',r['path'],'-frames:v','1','-vf','scale=320:-2','-q:v','3',str(p)],check=True,capture_output=True,timeout=30)
  records.append({'timestamp':t,'path':str(p),'sha256':digest(p)})
 sheet=Image.new('RGB',(1320,150*math.ceil(len(records)/6)));draw=ImageDraw.Draw(sheet)
 for j,item in enumerate(records):
  with Image.open(item['path']) as im:
   im=im.convert('RGB');im.thumbnail((218,122));x=j%6*220;y=j//6*150;sheet.paste(im,(x,y));draw.text((x+2,y+125),str(item['timestamp'])+'s',fill='white')
 sheet.save(E/f'pilot30_{n:02d}_dense.jpg',quality=90);atomic_json(out/'dense_evidence.json',records)
 print(n,sid,result['status'],flush=True)

# Explicit query revision after visual QC; old proposal is never overwritten.
sid=rows[6]['source_id'];out=E/'annotations'/sid
query='A diver lifts diving goggles away from the face at the water surface.'
event=read_json(out/'generation_input.json')['event'];event['query']=query
with np.load(CACHE/sid/'clip_features.npz') as z:
 vector=runtime.encode_text(query)[0];scores=z['features']@vector;peak=int(np.argmax(scores));timestamp=float(z['timestamps'][peak])
atomic_json(out/'blind_first_revision2.json',{'query':query,'interval':[max(0,timestamp-2),min(float(rows[6]['duration']),timestamp+2)],'procedure':'Query-only full-video CLIP retrieval before revised localization; no coarse/proposed bounds input','model_weights_sha256':digest(weights),'human_review':False})
prediction=localize(sid,query,[event],float(rows[6]['duration']),0,clip_runtime=runtime,visual_cache=CACHE)
atomic_json(out/'proposal_revision2.json',{'source_id':sid,'query':query,'prediction':prediction,'revision_reason':'Unsupported gendered pronoun replaced; original proposal retained','original_proposal_path':str(out/'proposal.json'),'human_verified':False})
