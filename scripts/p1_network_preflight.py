"""Small permitted-host metadata probe, explicitly enabled; never fetch video bodies."""
import argparse
from datetime import datetime,timezone
from pathlib import Path
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from data_process.lumae_scale.models import read_json,atomic_json
from data_process.lumae_scale.acquisition import allowed_url,AGENT
import requests

ap=argparse.ArgumentParser();ap.add_argument('--plan',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--enable-live-network',action='store_true');a=ap.parse_args()
if not a.enable_live_network:raise PermissionError('Network probe requires --enable-live-network')
records=[]
for method,url in [('GET','https://video.adsoftheworld.com/robots.txt'),('HEAD',read_json(a.plan)['candidates'][0]['source_url'])]:
    allowed_url(url);record={'utc':datetime.now(timezone.utc).isoformat(),'method':method,'url':url}
    try:
        with requests.request(method,url,timeout=(5,15),allow_redirects=False,stream=True,headers={'User-Agent':AGENT}) as r:
            record.update(status=r.status_code,headers={k:r.headers[k] for k in ('Content-Type','Content-Length','Retry-After') if k in r.headers})
            if method=='GET':record['body_excerpt']=next(r.iter_content(4096),b'').decode('utf-8',errors='replace')[:4096]
    except requests.RequestException as e:record['error_type']=type(e).__name__
    records.append(record)
atomic_json(a.output,{'records':records,'no_video_body_requested':True,'robots_interpretation':'4xx unavailable; not proof of acquisition permission. Operator explicitly attested existing permission. Video access denials remain terminal.'})
print(json.dumps(records))
