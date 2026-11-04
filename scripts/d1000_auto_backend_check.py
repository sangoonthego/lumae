"""Actual local vision diagnostics. Does not manufacture observations or accept labels."""
import argparse,hashlib,importlib.util,json,os,sys,time,urllib.request
from datetime import datetime,timezone
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from data_process.lumae_scale.models import read_json,atomic_json,digest
OLD=Path('E:/LUMAE/work/d1000/d1000_fasttrack_20261104_01')
OUT=ROOT/'research_outputs/d1000_autofix_20261104_01'
WORK=Path('E:/LUMAE/work/d1000/d1000_autofix_20261104_01')

def structured(answer):
 try:
  r=json.loads(answer.strip().removeprefix('```json').removesuffix('```').strip())
  if not isinstance(r,dict) or not isinstance(r.get('events'),list):return False
  for e in r['events']:
   if not all(isinstance(e.get(k),str) and e[k].strip() for k in ['query','visible_subject','visible_action','visible_object']):return False
   if not isinstance(e.get('evidence_timestamps'),list) or not e['evidence_timestamps']:return False
   a,b=e.get('coarse_start'),e.get('coarse_end')
   if not isinstance(a,(int,float)) or not isinstance(b,(int,float)) or not 0<=a<b:return False
  return True
 except (ValueError,TypeError,AttributeError):return False

def main():
 OUT.mkdir(parents=True,exist_ok=True);WORK.mkdir(parents=True,exist_ok=True)
 baseline=read_json(ROOT/'research_outputs/d1000_fasttrack_20261104_01/progress.json')
 oldseal=ROOT/'research_outputs/d1000_fasttrack_20261104_01/checkpoint_manifest.json'
 frozen=read_json(oldseal)
 # New run only: verify the prior evidence before using it, never call old init/run.
 for group in ('artifacts','code'):
  for p,h in frozen[group].items():
   if digest(Path(p))!=h:raise RuntimeError('Protected prior artifact changed: '+p)
 complete=OUT/'auto_annotation_smoke_results.json'
 if complete.exists() and len(read_json(complete).get('sources',[]))==10:
  for r in read_json(complete)['sources']:
   if digest(Path(r['image']))!=r['image_sha256'] or digest(Path(r['source_path']))!=r['source_sha256']:raise RuntimeError('Diagnostic input changed')
  print(json.dumps({'status':'RESUMED_IDENTICAL_DIAGNOSTICS','model_calls':0,'sources':10,'accepted':0,'global_blocker':'MISSING_CALLABLE_VISION_BACKEND'}));return
 environment={'modules':{m:bool(importlib.util.find_spec(m)) for m in ['torch','transformers','openai','ollama','llama_cpp','accelerate']},
  'configured_credentials_presence':{k:bool(os.getenv(k)) for k in ['OPENAI_API_KEY','ANTHROPIC_API_KEY','GOOGLE_API_KEY','GEMINI_API_KEY','LUMAE_VLM_COMMAND','OLLAMA_HOST']},'prior_seal_sha256':digest(oldseal)}
 try:
  with urllib.request.urlopen('http://127.0.0.1:11434/api/tags',timeout=2) as r:environment['ollama_tags']=json.load(r)
 except Exception as e:environment['ollama_service']=type(e).__name__
 atomic_json(OUT/'backend_inventory.json',environment)
 import torch
 from PIL import Image
 from transformers import AutoModelForImageTextToText,AutoProcessor
 torch.set_num_threads(2);modelpath=ROOT/'local_data/cache/smolvlm_256m'
 start=time.perf_counter();processor=AutoProcessor.from_pretrained(modelpath,local_files_only=True)
 processor.chat_template=processor.tokenizer.chat_template;processor.image_processor.size={'longest_edge':1024}
 model=AutoModelForImageTextToText.from_pretrained(modelpath,local_files_only=True,torch_dtype=torch.float32).eval()
 identity={'model_id':'HuggingFaceTB/SmolVLM-256M-Instruct','model_path':str(modelpath),'weights_sha256':digest(modelpath/'model.safetensors'),'device':'cpu','torch':torch.__version__,'load_seconds':time.perf_counter()-start,'max_new_tokens':160,'do_sample':False}
 def infer(imagepath,question):
  t=time.perf_counter();started=datetime.now(timezone.utc).isoformat();prompt=f'<|im_start|>User:<image>{question}<end_of_utterance>\nAssistant:'
  with Image.open(imagepath) as im:inputs=processor(text=prompt,images=[im.convert('RGB')],return_tensors='pt')
  with torch.inference_mode():tokens=model.generate(**inputs,do_sample=False,max_new_tokens=160)
  answer=processor.batch_decode(tokens[:,inputs['input_ids'].shape[1]:],skip_special_tokens=True)[0]
  return {'image':str(imagepath),'image_sha256':digest(imagepath),'prompt':question,'answer':answer,'started_utc':started,'completed_utc':datetime.now(timezone.utc).isoformat(),'elapsed_seconds':time.perf_counter()-t,'input_tokens':int(inputs['input_ids'].shape[1]),'output_tokens':int(tokens.shape[1]-inputs['input_ids'].shape[1]),'model':identity,'actual_model_generate_executed':True}
 control=read_json(ROOT/'local_data/reports/lumae_ads_d200/d200_vlm_smoke.json');imagepath=ROOT/control['image'];controls=[]
 for name,key,expected in [('positive','positive_query','YES'),('negative','negative_query','NO')]:
  p=OUT/('control_'+name+'.json')
  if p.exists():result=read_json(p)
  else:
   result=infer(imagepath,'This image has timestamped before, inside, and after panels from a video. Does this exact event occur in the inside panels: '+control[key]+' Describe what is actually visible and answer YES, NO, or UNCERTAIN.')
   result.update(expected=expected,control=name);atomic_json(p,result)
  controls.append(result);print('CONTROL '+name+' '+result['answer'][:120],flush=True)
 # Any contradiction or uncertainty fails closed. Never convert all abstentions to content rejects.
 control_pass=all(r['answer'].strip().upper().startswith(r['expected']) for r in controls)
 states={p.stem:read_json(p) for p in (OLD/'states').glob('*.json')}
 rows=[r for r in read_json(OLD/'inventory.json') if r['existing_paths'] and not states.get(r['source_video_id'],{}).get('status','').startswith(('ACCEPTED','REJECTED'))][:10]
 if len(rows)!=10:raise RuntimeError('Cannot select ten pending cached sources')
 results=[]
 for row in rows:
  sid=row['source_video_id'];p=WORK/'diagnostics'/(sid+'.json');op=OLD/'evidence'/sid/'overview.jpg'
  if p.exists():result=read_json(p)
  else:
   result=infer(op,'Inspect only the timestamped video frames. Return only JSON: {"events": [{"query": "one concrete visible event", "visible_subject": "", "visible_action": "", "visible_object": "", "coarse_start": 0.0, "coarse_end": 0.0, "evidence_timestamps": [0.0]}]}. Use actual visible times. If no defensible event exists return {"events": []}. Do not invent motion between frames.')
   result.update(source_video_id=sid,source_path=row['existing_paths'][0],source_sha256=digest(Path(row['existing_paths'][0])),structured_events_schema_valid=structured(result['answer']),manual_observations_used=False,manual_verdicts_used=False)
   atomic_json(p,result)
  result={**result,'state':'ABSTAIN_BACKEND_NOT_VALIDATED','accepted':False,'reason':'Model is callable but no validated unattended semantic-verification path is established; original D500 validator is not bypassed','localization_executed':False,'original_acceptance_validator_executed':False}
  results.append(result)
  atomic_json(OUT/'auto_annotation_smoke_results.json',{'smoke_pass':False,'full_pipeline_smoke_pass':False,'diagnostic_inference_attempts':len(results),'accepted':0,'annotation_rejected':0,'backend_abstained':len(results),'manual_input_required':False,'new_downloads':0,'controls_pass':control_pass,'controls':controls,'sources':results,'D1000_total':baseline['total'],'global_blocker':'MISSING_CALLABLE_VISION_BACKEND','blocker_detail':'No VALIDATED callable annotation backend. SmolVLM itself is callable; do not claim no model exists. Controls and structured output must be assessed before enabling acceptance.'})
  print('SOURCE_DIAGNOSTIC '+sid,flush=True)
 print(json.dumps({'smoke_pass':False,'diagnostic_sources':len(results),'controls_pass':control_pass,'total':baseline['total'],'accepted':0,'global_blocker':'MISSING_CALLABLE_VISION_BACKEND'}))

if __name__=='__main__':
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('command',choices=['probe']);ap.parse_args();main()
