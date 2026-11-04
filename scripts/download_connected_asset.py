"""Materialize one already-authorized connector file; scrub temporary access URL."""
import hashlib
import json
from pathlib import Path
import sys
import urllib.request

p=Path(sys.argv[1]);r=json.loads(p.read_text());target=Path(r['output']);target.parent.mkdir(parents=True,exist_ok=True)
try:
    uri=r['file_uri'];url=uri.get('download_url') if isinstance(uri,dict) else uri
    h=hashlib.sha256();n=0
    with urllib.request.urlopen(url,timeout=60) as response,target.open('xb') as out:
        while data:=response.read(1024*1024):
            out.write(data);h.update(data);n+=len(data)
    result={'status':'VERIFIED' if h.hexdigest()==r['sha256'] and n==r['expected_bytes'] else 'FAIL',
            'bytes':n,'sha256':h.hexdigest(),'path':str(target.resolve())}
except Exception as e:
    result={'status':'BLOCKED','error_type':type(e).__name__}
finally:
    p.write_text(json.dumps(result,indent=2));print(json.dumps(result))
if result['status']!='VERIFIED':sys.exit(1)
