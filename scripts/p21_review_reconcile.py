"""Fail-closed adapter for pre-localization sanitation abstentions.

The fresh frozen helper assumed every nonempty event list had proposals. One
real query failed sanitation first. This adapter persists that failure without
inventing QC or revising inputs. It reproduces the actual manual reconciliation
used in the timed run; it does not change the frozen annotation method.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
O=Path(__file__).resolve().parents[1]/'research_outputs/p21_20261103_01'
E=Path('E:/LUMAE/work/d1000/p21_20261103_01')
from data_process.lumae_scale.models import read_json,digest
from data_process.lumae_scale.incremental import immutable_write,json_bytes
from data_process.lumae_scale.annotation_v4 import rank_events
from data_process.lumae_scale.quality_gate import query_errors

def prepare():
 inventory=O/'benchmarks/fresh_source_inventory.json';notes=read_json(O/'benchmarks/fresh_review_notes.json')
 assert notes['inventory_sha256']==digest(inventory)
 rows=read_json(inventory);assert len(rows)==len(notes['sources'])
 observations={x['source_id']:x for x in read_json(O/'benchmarks/fresh_events.json')}
 bundles=[];machine=[]
 for row,note in zip(rows,notes['sources']):
  sid=row['source_id'];assert sid==note['source_id'];obs=observations[sid];folder=E/'runs/fresh01/sources'/sid
  assert obs['actual_complete_overview_inspected'] is True
  bundle={'source_id':sid,'actual_complete_overview_inspected':True,'reason':obs.get('abstention_reason') or note['reason'],'human_verified':False}
  if obs['events'] and not rank_events(obs['events']):
   assert note.get('machine_abstention') and read_json(folder/'complete.json')['status']=='ABSTAIN_NO_DEFENSIBLE_EVENT'
   errors=[query_errors(e['query']) for e in obs['events']]
   assert any(errors) and not (folder/'proposals.json').exists()
   bundle.update(actual_qc_inspected=False,query_sanitation_errors=errors)
   immutable_write(folder/'visual_review.json',json_bytes(bundle))
   decision={'source_id':sid,'status':'ABSTAIN_QUERY_SANITATION','reason':note['reason'],'query_sanitation_errors':errors,'variants':[],'human_verified':False,'new_dataset_frozen':False}
   immutable_write(folder/'acceptance_decision.json',json_bytes(decision));machine.append(decision);continue
  if obs['events']:
   assert note['actual_all_boundary_pages_inspected'] and len(note['event_reviews'])==len(obs['events'])
   assert (folder/'proposals.json').exists()
   bundle['inspected_review_grids']=[g['path'] for g in read_json(folder/'review_grid_manifest.json')]
   bundle['event_reviews']=[dict(v,query=e['query'],actual_overview_inspected=True,actual_qc_inspected=True,human_verified=False) for e,v in zip(obs['events'],note['event_reviews'])]
  bundles.append(bundle)
 immutable_write(O/'benchmarks/fresh_reviews.json',json_bytes(bundles))
 immutable_write(O/'benchmarks/fresh_machine_abstentions.json',json_bytes({'reviewed_total':len(bundles)+len(machine),'import_bundles':len(bundles),'machine_abstentions':machine,'note':'Frozen review helper requires nonempty event lists to have proposals; sanitation abstention is explicitly reconciled without localization or fabricated QC. Original query and producer code remain unchanged.'}))
 return {'visual_bundles':len(bundles),'machine_abstentions':len(machine)}

if __name__=='__main__':print(prepare())
