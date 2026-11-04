"""Apply hash-bound final QC exclusions without rewriting trial ledgers."""
from pathlib import Path
from .models import read_json,digest

def load_exclusions(path):
 registry=read_json(path)
 if registry.get('schema')!='P21_FINAL_QC_EXCLUSIONS_V1':raise ValueError('Unknown exclusion schema')
 result={}
 for item in registry['exclusions']:
  sid=item['source_video_id']
  if sid in result or item.get('effective_status')!='WITHDRAWN_FINAL_BOUNDARY_AUDIT':raise ValueError('Invalid exclusion identity/status')
  if item.get('human_verified') is not False or item.get('actual_original_resolution_frames_inspected') is not True:raise ValueError('Actual disclosed AI inspection required')
  if digest(Path(item['accepted_record_path']))!=item['accepted_record_sha256']:raise ValueError('Withdrawn trial record changed')
  if not item.get('evidence_sha256'):raise ValueError('Withdrawal needs actual evidence')
  if not 0<=item['earlier_confirmed_positive_seconds']<item['window'][0]:raise ValueError('Earlier confirmed action must precede rejected proposal onset')
  for p,h in item['evidence_sha256'].items():
   if digest(Path(p))!=h:raise ValueError('Withdrawal evidence changed')
  result[sid]=item
 return result

def effective_records(rows,exclusions):
 byid={r['source_video_id']:r for r in rows}
 if len(byid)!=len(rows):raise ValueError('Duplicate source in trial acceptances')
 if set(exclusions)-set(byid):raise ValueError('Exclusion outside new P21 trial cohort')
 for sid,item in exclusions.items():
  r=byid[sid]
  for k in ('qid','source_video_sha256','semantic_artifact_sha256','window'):
   if item[k]!=r[k]:raise ValueError('Excluded annotation identity changed: '+k)
 return [r for r in rows if r['source_video_id'] not in exclusions]
