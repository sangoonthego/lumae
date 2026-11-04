"""Bounded dense-evidence recovery. Frozen ranker/refinement are unchanged."""
import csv,math,shutil,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image,ImageDraw
from data_process.lumae_scale.models import ROOT,read_json,atomic_json,digest
from data_process.lumae_scale.clip_runtime import ClipRuntime
from data_process.lumae_scale.visual_index import extract_clip_features
from data_process.lumae_scale.temporal_labeler import localize
import torch
torch.set_num_threads(4)
O=ROOT/'research_outputs/p14_p15_20261103_02'
if (O/'pilot30_freeze_manifest.json').exists():raise PermissionError('Frozen pilot: use read-only validation; generation requires a new version')
O.mkdir(exist_ok=True)
OLD=ROOT/'research_outputs/p14_p15_20261103_01';A=Path('E:/LUMAE/work/d1000/p14_p15_20261103_01');E=A.with_name('p14_p15_20261103_02');E.mkdir(exist_ok=True)
rows=list(csv.DictReader((OLD/'pilot30_inventory.csv').open(encoding='utf-8-sig')));notes=read_json(OLD/'visual_qc_notes.json');obs={v[0]:v for v in read_json(OLD/'visual_observations.json')}
weights=ROOT/'local_data/cache/clip_weights/ViT-B-32.pt';runtime=ClipRuntime(str(weights),device='cpu',download_root=E/'model_cache')
# Corrected visible phase brackets from actual dense visual inspection, not Test GT.
phases={1:(5.5,8),2:(20.5,22),3:(11,12),4:(10.5,11.5),6:(19,20),8:(26,30.01),9:(42,47),10:(20.5,21.5),11:(28.5,29.5),12:(17,23),13:(2,5.5),14:(21.5,24),15:(12.5,18),18:(28.5,31.5),19:(22.5,27.5),20:(19.5,25.5),21:(50,51.5),23:(19,20.5),24:(16.5,18),25:(20.5,21.5),26:(67,74),27:(16.5,18.5),28:(11.5,13.5),29:(38,39),30:(14.5,17.5)}
rewrites={1:'A blonde performer in a red and black patterned outfit sings into a microphone.',6:'A red-haired woman takes off a face mask.',21:'An illustrated booklet flips from a brown cover to black-and-white pages above a car hood.',23:'A person poses with a yellow skirt spread out in a sunlit room.',29:'A white video-call background changes to a furnished room behind a smiling person holding up a hand.'}
for n,r in enumerate(rows,1):
 if n==22:continue
 sid=r['source_id'];out=E/'annotations'/sid;out.mkdir(parents=True,exist_ok=True)
 if n in (5,7,16,17):
  if not (E/'visual_index'/sid).exists():shutil.copytree(A/'visual_index'/sid,E/'visual_index'/sid)
  continue
 if (out/'qc.json').exists():continue
 started=time.monotonic();query=rewrites.get(n,obs[n][2]);s,e=phases[n];e=min(e,float(r['duration']))
 frames={}
 for name in ('dense_evidence.json','boundary_2fps.json'):
  for f in read_json(A/'annotations'/sid/name):frames[f['timestamp']]=f
 times=sorted(frames);positive=[t for t in times if s<=t<e]
 if not positive:raise ValueError('No actual positive evidence frames')
 cache=E/'visual_index'/sid;cache.mkdir(parents=True,exist_ok=True);items=[]
 for j,t in enumerate(times):
  target=cache/f'frame_{j:04d}.jpg';shutil.copyfile(frames[t]['path'],target);items.append({'timestamp':t,'path':target.name,'sha256':digest(target)})
 metadata={'video_id':sid,'video_path':r['path'],'video_sha256':r['sha256'],'duration':float(r['duration']),'frames':items,'contact_sheets':[],'clip_features_status':'pending','sampling_step_seconds':.5,'sampling_note':'Exact seek2FPS boundary frames plus1FPS wider context; actual timestamps explicitly listed'}
 atomic_json(cache/'metadata.json',metadata)
 extract_clip_features(sid,clip_runtime=runtime,visual_cache=E/'visual_index')
 # Query-only estimate is calculated first with no phase or generator interval input.
 with np.load(cache/'clip_features.npz') as z:
  scores=z['features']@runtime.encode_text(query)[0];t=float(z['timestamps'][int(np.argmax(scores))])
 atomic_json(out/'blind_first.json',{'query':query,'interval':[max(0,t-2),min(float(r['duration']),t+2)],'procedure':'Query-only retrieval over exact-seek evidence; no generator proposal/phase as numerical input','weights_sha256':digest(weights),'human_verified':False})
 event={'query':query,'coarse_start':s,'coarse_end':e,'evidence_timestamps':positive,'visible_subject':obs[n][6],'visible_action':obs[n][7],'visible_object':obs[n][8]}
 atomic_json(out/'generation_input.json',{'query':query,'visible_phase':[s,e],'positive_evidence_timestamps':positive,'prior_proposal_sha256':digest(A/'annotations'/sid/'proposal.json'),'procedure':'Dense actual visual review, exact-seek CLIP and unchanged Semantic V3; no human label','model_revision':'NOT_EXPOSED'})
 try:pred=localize(sid,query,[event],float(r['duration']),0,clip_runtime=runtime,visual_cache=E/'visual_index');result={'source_id':sid,'query':query,'prediction':pred,'generation_seconds':time.monotonic()-started}
 except Exception as ex:result={'source_id':sid,'query':query,'error':str(ex)}
 atomic_json(out/'proposal.json',result)
 if 'prediction' not in result:print(n,result['error'],flush=True);continue
 pred=result['prediction'];start=max(0,pred['start']-1);end=min(float(r['duration'])-.05,pred['end']+1);ts=np.arange(start,end+.01,.5).tolist();qc=[];d=out/'qc';d.mkdir(exist_ok=True)
 for j,t in enumerate(ts):
  p=d/f'{j:03d}.jpg';subprocess.run(['ffmpeg','-v','error','-ss',str(t),'-i',r['path'],'-frames:v','1','-vf','scale=320:-2','-q:v','3',str(p)],check=True,capture_output=True,timeout=30);qc.append({'timestamp':t,'path':str(p),'sha256':digest(p)})
 image=Image.new('RGB',(1320,150*math.ceil(len(qc)/6)));draw=ImageDraw.Draw(image)
 for j,f in enumerate(qc):
  with Image.open(f['path']) as im:
   im=im.convert('RGB');im.thumbnail((218,122));a=j%6*220;b=j//6*150;image.paste(im,(a,b));draw.text((a+2,b+125),str(round(f['timestamp'],3)),fill='white')
 sheet=E/f'pilot30_{n:02d}_qc.jpg';image.save(sheet,quality=90)
 # Original whole-video contact sheets remain explicit visual-index evidence.
 for p in (A/'visual_index'/sid/'contact_sheets').glob('*.jpg'):
  target=cache/'contact_sheets'/p.name;target.parent.mkdir(exist_ok=True);shutil.copyfile(p,target)
 meta=read_json(cache/'metadata.json');meta['contact_sheets']=[str(p.relative_to(cache)).replace('\\','/') for p in sorted((cache/'contact_sheets').glob('*.jpg'))];atomic_json(cache/'metadata.json',meta)
 atomic_json(out/'qc.json',qc);print(n,pred['start'],pred['end'],flush=True)

# Correct candidate descriptions as explicit revisions; old raw outputs retained.
for n,subject,action,obj in [(21,'illustrated booklet','flips from a brown cover to','black-and-white pages above a car hood'),(23,'person','poses with','yellow skirt spread out in a sunlit room'),(29,'white video-call background','changes to','furnished room behind a smiling person holding up a hand')]:
 r=rows[n-1];sid=r['source_id'];out=E/'annotations'/sid;v=read_json(out/'generation_input.json')
 event={'query':v['query'],'coarse_start':v['visible_phase'][0],'coarse_end':v['visible_phase'][1],'evidence_timestamps':v['positive_evidence_timestamps'],'visible_subject':subject,'visible_action':action,'visible_object':obj}
 pred=localize(sid,v['query'],[event],float(r['duration']),0,clip_runtime=runtime,visual_cache=E/'visual_index')
 atomic_json(out/'candidate_revision.json',{'query':v['query'],'prediction':pred,'event':event,'reason':'Align candidate semantic fields with actual reviewed event; prior proposal retained','prior_proposal_sha256':digest(out/'proposal.json')})
 print('candidate revision',n,pred['start'],pred['end'],flush=True)

for n,query,phase,subject,obj in [(9,'A wide view shows rows of empty cinema seats beneath a 4DX sign.',(43.5,46.5),'wide cinema view','rows of empty seats beneath a 4DX sign'),(12,'A video-call grid displays six copies of the same man.',(19,23),'video-call grid','six copies of the same man'),(26,'An elastic strap is attached to a curved plastic face shield in an animation.',(67.5,70.5),'elastic strap','curved plastic face shield'),(30,'A yellow shopping bag and a red shopping bag are displayed side by side.',(14.5,17.5),'yellow shopping bag and red shopping bag','side-by-side display')]:
 r=rows[n-1];sid=r['source_id'];out=E/'annotations'/sid;meta=read_json(E/'visual_index'/sid/'metadata.json')
 event={'query':query,'coarse_start':phase[0],'coarse_end':phase[1],'evidence_timestamps':[f['timestamp'] for f in meta['frames'] if phase[0]<=f['timestamp']<phase[1]],'visible_subject':subject,'visible_action':'shows','visible_object':obj}
 with np.load(E/'visual_index'/sid/'clip_features.npz') as z:
  t=float(z['timestamps'][int(np.argmax(z['features']@runtime.encode_text(query)[0]))])
 atomic_json(out/'blind_query_revision.json',{'query':query,'interval':[max(0,t-2),min(float(r['duration']),t+2)],'procedure':'Query-only retrieval before revised generator; no phase/interval supplied','human_verified':False})
 pred=localize(sid,query,[event],float(r['duration']),0,clip_runtime=runtime,visual_cache=E/'visual_index')
 atomic_json(out/('schema_revision.json' if n==30 else 'query_revision.json'),{'query':query,'prediction':pred,'event':event,'reason':'Concrete visible subject/phase clarified; strict original sanitation retained; old proposals retained'})
 print('query revision',n,pred['start'],pred['end'],flush=True)
