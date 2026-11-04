"""Deterministic subset freeze, distinct from a final D1000 dataset freeze."""
from pathlib import Path
import hashlib,json
from .incremental import immutable_write,json_bytes
from .models import digest
from .freeze import validate_annotation
from .builder import BuildConfig

class CompletionConfig(BuildConfig):
    """Bind existing builder recovery to the isolated P1.4 visual-cache layout."""
    @property
    def visual_cache(self):
        return self.data_root/'work/d1000'/self.version/'visual_index'

def validate_completion(inventory,accepted,*,root,visual_cache):
    if len(inventory)!=30 or len({r['source_id'] for r in inventory})!=30:raise ValueError('Exactly original30 required')
    by={r['source_id']:r for r in inventory};qids=set();sources=set()
    for r in accepted:
        sid=r['source_video_id']
        if sid not in by or by[sid]['technical_pass'] not in (True,'True'):raise ValueError('Accepted source not technically valid')
        if r['qid'] in qids or sid in sources:raise ValueError('Duplicate accepted identity')
        if r.get('review_provenance')!='AI_PSEUDO_LABELED' or r.get('human_verified') is not False:raise ValueError('False human provenance')
        if r['source_video_sha256']!=by[sid]['sha256']:raise ValueError('Source identity mismatch')
        validate_annotation(r,root=Path(root),visual_cache=Path(visual_cache));qids.add(r['qid']);sources.add(sid)
    return {'pass':True,'original_candidates':30,'accepted_sources_with_annotations':len(sources),'accepted_annotations':len(qids),'human_verified':0,'human_IAA':None}

def freeze_completion(output,manifest):
    output=Path(output)
    if manifest.get('final_d1000_freeze') is not False:raise ValueError('Pilot freeze cannot claim D1000 completion')
    data=json_bytes(manifest);immutable_write(output/'pilot30_freeze_manifest.json',data)
    immutable_write(output/'pilot30_freeze_manifest.sha256',(hashlib.sha256(data).hexdigest()+'\n').encode())
    return hashlib.sha256(data).hexdigest()
