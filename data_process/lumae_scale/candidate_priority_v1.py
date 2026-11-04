"""Predownload policy: auditable equal priors when legitimate metadata is absent."""
import hashlib
from urllib.parse import urlparse

METADATA=('title','description','category','brand','campaign','duration')
FORBIDDEN=('source_split','query','window','label','clip_score','download_status','accepted')

def score_candidate(row,seed=20261103):
    sid=row['source_video_id'];url=row.get('source_url','')
    eligible=urlparse(url).scheme=='https' and urlparse(url).hostname=='video.adsoftheworld.com'
    present=[k for k in METADATA if row.get(k) not in (None,'')]
    # No calibrated mapping from optional metadata to annotation yield exists.
    # Availability is disclosed, never represented as semantic quality.
    return {'source_video_id':sid,'source_url':url,'eligible':eligible,'priority_score':0.0,
        'available_metadata':'|'.join(present),'missing_metadata':'|'.join(k for k in METADATA if k not in present),
        'tie_key':hashlib.sha256(f'{seed}:{sid}'.encode()).hexdigest(),
        'method':'EQUAL_PRIOR_SEEDED_FALLBACK','semantic_quality_prediction_available':False}

def rank_candidates(rows,seed=20261103):
    result=[score_candidate(r,seed) for r in rows]
    return sorted(result,key=lambda r:(not r['eligible'],-r['priority_score'],r['tie_key']))
