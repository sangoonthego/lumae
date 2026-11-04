"""Offline, versioned P1.4 inventory and visual evidence; never imports downloader."""
import csv,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from data_process.lumae_scale.models import ROOT,read_json,atomic_json,digest
from data_process.lumae_scale.pilot_quality import validate_video
from PIL import Image,ImageDraw

OUT=ROOT/'research_outputs/p14_p15_20261103_01'
if (ROOT/'research_outputs/p14_p15_20261103_02/pilot30_freeze_manifest.json').exists():raise PermissionError('Evidence is a dependency of frozen pilot02; extraction requires a new version')
EVIDENCE=Path('E:/LUMAE/work/d1000/p14_p15_20261103_01')
OLD=Path('E:/LUMAE/work/d1000/pilot_20261103_01')
CACHE=Path('E:/LUMAE/cache/p1_20261103_01/visual_index')
OUT.mkdir(exist_ok=True);EVIDENCE.mkdir(exist_ok=True)
records=sorted(read_json(OLD/'manifests/pilot_download_manifest.json')['records'],key=lambda r:r['source_video_id'])
assert len(records)==30
inventory=[]
for i,r in enumerate(records):
 p=Path('E:/LUMAE')/r['path'];assert digest(p)==r['sha256'] and p.stat().st_size==r['bytes']
 info=validate_video(p);probe=info['probe']
 row={'candidate_id':f'pilot30_{i+1:02d}','source_id':r['source_video_id'],'source_url':r['source_url'],'path':str(p),'sha256':r['sha256'],'bytes':r['bytes'],'duration':probe['duration'],'width':probe['width'],'height':probe['height'],'fps':probe['fps'],'codec':probe['codec'],'technical_pass':info['pass'],'original_disposition':r['status'],'status':'AWAITING_ACTUAL_VISUAL_REVIEW' if info['pass'] else 'REJECTED_TECHNICAL'}
 inventory.append(row)
 if not info['pass']:continue
 index=read_json(CACHE/r['source_video_id']/'metadata.json');frames=index['frames']
 canvas=Image.new('RGB',(1320,150*((len(frames)+5)//6)),(20,20,20));draw=ImageDraw.Draw(canvas)
 for j,f in enumerate(frames):
  image_path=CACHE/r['source_video_id']/f['path'];assert digest(image_path)==f['sha256']
  with Image.open(image_path) as image:
   image=image.convert('RGB');image.thumbnail((218,122));x=(j%6)*220;y=(j//6)*150;canvas.paste(image,(x,y));draw.text((x+3,y+125),str(f['timestamp'])+'s',fill='white')
 canvas.save(EVIDENCE/(row['candidate_id']+'_overview.jpg'),quality=90)
 print(row['candidate_id'],r['source_video_id'],flush=True)
with (OUT/'pilot30_inventory.csv').open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(inventory[0]));w.writeheader();w.writerows(inventory)
atomic_json(OUT/'inventory_validation.json',{'pass':True,'count':30,'technical_valid':sum(r['technical_pass'] for r in inventory),'raw_bytes':sum(r['bytes'] for r in inventory),'network_requests':0,'old_p1_manifest_sha256':digest(OLD/'manifests/pilot_download_manifest.json')})
atomic_json(OUT/'visual_review_queue.json',[{'candidate_id':r['candidate_id'],'source_id':r['source_id'],'overview':str(EVIDENCE/(r['candidate_id']+'_overview.jpg'))} for r in inventory if r['technical_pass']])
