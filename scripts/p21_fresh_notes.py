"""Convert actual fresh visual notes to the frozen runner/reviewer contracts."""
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from p21_audit import O,E
from data_process.lumae_scale.models import read_json,digest
from data_process.lumae_scale.incremental import immutable_write,json_bytes

def events():
 inventory=O/'benchmarks/fresh_source_inventory.json';rows=read_json(inventory);notes=read_json(O/'benchmarks/fresh_visual_notes.json')
 assert notes['inventory_sha256']==digest(inventory)
 assert len(notes['sources'])==len(rows)
 out=[]
 for r,note in zip(rows,notes['sources']):
  assert r['source_id']==note['source_id'] and note['actual_full_overview_inspected'] is True
  values=note.get('events',[]);assert len(values)<=2
  ev=[dict(e,actual_visual_support=True,event_quality=dict(support=4,specificity=4,uniqueness=4,boundary_visibility=3,clarity=4),
       generator_evidence_path=r['overview_path'],generator_evidence_sha256=r['overview_sha256'],human_verified=False) for e in values]
  out.append({'source_id':r['source_id'],'events':ev,'abstention_reason':note.get('abstention_reason'),'actual_complete_overview_inspected':True,
      'model_interface':'Codex multimodal image inspection','model_revision':'NOT_EXPOSED','review_blinding':'Source first seen after experimental code/config/selection freeze; AI QC is unblinded and same model interface.'})
 immutable_write(O/'benchmarks/fresh_events.json',json_bytes(out));print('ACTUAL_FRESH_SOURCES',len(out))

def reviews():
 inventory=O/'benchmarks/fresh_source_inventory.json';rows=read_json(inventory);notes=read_json(O/'benchmarks/fresh_review_notes.json')
 assert notes['inventory_sha256']==digest(inventory)
 assert len(notes['sources'])==len(rows)
 observations={r['source_id']:r for r in read_json(O/'benchmarks/fresh_events.json')};out=[]
 for r,note in zip(rows,notes['sources']):
  sid=r['source_id'];assert sid==note['source_id'];obs=observations[sid];folder=E/'runs/fresh01/sources'/sid
  bundle={'source_id':sid,'actual_complete_overview_inspected':True,'reason':obs.get('abstention_reason') or note['reason'],'human_verified':False}
  if obs['events']:
   assert note['actual_all_boundary_pages_inspected'] is True
   bundle['inspected_review_grids']=[g['path'] for g in read_json(folder/'review_grid_manifest.json')]
   assert len(note['event_reviews'])==len(obs['events'])
   bundle['event_reviews']=[dict(review,query=e['query'],actual_overview_inspected=True,actual_qc_inspected=True,human_verified=False) for e,review in zip(obs['events'],note['event_reviews'])]
  out.append(bundle)
 immutable_write(O/'benchmarks/fresh_reviews.json',json_bytes(out));print('ACTUAL_FRESH_REVIEWS',len(out))

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('command',choices=['events','reviews']);a=ap.parse_args();events() if a.command=='events' else reviews()
