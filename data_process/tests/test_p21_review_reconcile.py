import importlib.util,json,sys
from pathlib import Path
import pytest

def module():
 p=Path(__file__).resolve().parents[2]/'scripts/p21_review_reconcile.py'
 spec=importlib.util.spec_from_file_location('p21_reconcile_test',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def fixture(m,tmp_path,monkeypatch):
 o=tmp_path/'out';e=tmp_path/'e';(o/'benchmarks').mkdir(parents=True);folder=e/'runs/fresh01/sources/sid';folder.mkdir(parents=True)
 monkeypatch.setattr(m,'O',o);monkeypatch.setattr(m,'E',e)
 inv=o/'benchmarks/fresh_source_inventory.json';inv.write_text(json.dumps([{'source_id':'sid'}]))
 obs={'source_id':'sid','actual_complete_overview_inspected':True,'events':[{'query':'Green shoots grow upward out of dark soil.','coarse_start':1,'coarse_end':2,'actual_visual_support':True}]}
 (o/'benchmarks/fresh_events.json').write_text(json.dumps([obs]))
 (o/'benchmarks/fresh_review_notes.json').write_text(json.dumps({'inventory_sha256':m.digest(inv),'sources':[{'source_id':'sid','reason':'Actual sanitation failure','machine_abstention':True}]}))
 (folder/'complete.json').write_text(json.dumps({'status':'ABSTAIN_NO_DEFENSIBLE_EVENT'}));return folder,o

def test_machine_abstention_cannot_become_acceptance_and_is_idempotent(tmp_path,monkeypatch):
 m=module();folder,o=fixture(m,tmp_path,monkeypatch)
 assert m.prepare()=={'visual_bundles':0,'machine_abstentions':1}
 before=(folder/'acceptance_decision.json').read_bytes();m.prepare();assert (folder/'acceptance_decision.json').read_bytes()==before
 d=json.loads(before);v=json.loads((folder/'visual_review.json').read_bytes())
 assert d['status']=='ABSTAIN_QUERY_SANITATION' and d['human_verified'] is False and v['actual_qc_inspected'] is False
 assert json.loads((o/'benchmarks/fresh_reviews.json').read_text())==[]

def test_abstention_requires_real_terminal_status_and_no_proposal(tmp_path,monkeypatch):
 m=module();folder,_=fixture(m,tmp_path,monkeypatch);(folder/'proposals.json').write_text('[]')
 with pytest.raises(AssertionError):m.prepare()
