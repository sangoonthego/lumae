"""Offline-only eligibility correction after the disclosed V1 cache miss.

This guard was not used in the evaluated fresh wave and is not a ranking-quality
improvement. Existing assets require reuse/revalidation, never auto acceptance.
"""
import os
from pathlib import Path

VERSION='offline_all_registered_raw_roots_v2_not_live_evaluated'

def inventory_mp4(roots):
 inventory={}
 for value in roots:
  root=Path(value).resolve()
  if not root.is_dir():raise FileNotFoundError('Required asset root unavailable: '+str(root))
  # scandir errors propagate: inaccessible is never interpreted as empty.
  with os.scandir(root) as entries:
   for entry in entries:
    if not entry.name.lower().endswith('.mp4'):continue
    if entry.is_symlink():raise ValueError('Unresolved asset alias requires explicit audit')
    if entry.is_file():inventory.setdefault(Path(entry.name).stem,[]).append(str(Path(entry.path).resolve()))
 return {sid:sorted(paths) for sid,paths in inventory.items()}

def partition_candidates(rows,inventory,attempted_ids):
 result={'network_eligible':[],'cached_requires_revalidation':[],'already_attempted':[]}
 seen=set()
 for row in rows:
  sid=row['source_video_id']
  if sid in seen:raise ValueError('Duplicate candidate ID')
  seen.add(sid)
  if sid in attempted_ids:result['already_attempted'].append(row)
  elif sid in inventory:result['cached_requires_revalidation'].append({**row,'existing_raw_paths':inventory[sid],'automatic_acceptance':False})
  else:result['network_eligible'].append(row)
 return result
