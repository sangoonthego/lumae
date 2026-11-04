import importlib.util,json
from pathlib import Path
import pytest

@pytest.fixture
def controller(tmp_path,monkeypatch):
 spec=importlib.util.spec_from_file_location('fasttrack_test',Path(__file__).resolve().parents[2]/'scripts/d1000_fasttrack.py')
 m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 monkeypatch.setattr(m,'BASE',tmp_path/'work');monkeypatch.setattr(m,'OUT',tmp_path/'out');m.BASE.mkdir();m.OUT.mkdir()
 return m

def test_resume_counts_unique_and_does_not_change_prior(controller,monkeypatch):
 m=controller;prior=m.BASE/'prior.jsonl';parent=m.BASE/'parent.jsonl'
 m.jsonl(prior,[{'qid':'q1','source_video_id':'new1','source_video_sha256':'h1'}]);m.jsonl(parent,[{'vid':'lumae_ads_old'}])
 config={'prior':str(prior),'prior_sha256':m.digest(prior),'parent':str(parent),'protected':{},'target_new':500}
 m.atomic_json(m.BASE/'config.json',config)
 monkeypatch.setattr(m,'accepted',lambda:[]);monkeypatch.setattr(m,'canonical_row',lambda r,q:{'vid':'lumae_ads_'+r['source_video_id'],'qid':q})
 before=prior.read_bytes();a=m.progress();b=m.progress()
 assert a['remaining']==499 and a['new_accepted']==1 and a['total']==501
 assert a['new_accepted']==b['new_accepted'] and prior.read_bytes()==before
 monkeypatch.setattr(m,'accepted',lambda:[{'qid':'q2','source_video_id':'new1','source_video_sha256':'h2'}])
 with pytest.raises(AssertionError):m.progress()

def test_accepted_record_checksum_prevents_tampered_resume(controller,monkeypatch):
 m=controller;p=m.BASE/'accepted'/'s.json';m.atomic_json(p,{'source_video_id':'s','annotation_policy':m.POLICY,'human_verified':False,'cache_root':str(m.BASE)})
 monkeypatch.setattr(m,'validate_annotation',lambda *args,**kwargs:None)
 m.atomic_json(m.BASE/'states/s.json',{'record_sha256':m.digest(p)})
 assert len(m.accepted())==1
 r=m.read_json(p);r['window']=[1,9];m.atomic_json(p,r)
 with pytest.raises(AssertionError,match='Accepted record changed'):m.accepted()

def test_gold_or_human_cannot_be_silently_imported(controller):
 m=controller;m.atomic_json(m.BASE/'accepted/s.json',{'annotation_policy':m.POLICY,'human_verified':True})
 with pytest.raises(AssertionError):m.accepted()

def test_cached_inventory_prevents_network(controller,monkeypatch):
 m=controller;m.atomic_json(m.BASE/'inventory.json',[{'source_video_id':'s','existing_paths':[]}])
 monkeypatch.setattr(m,'read_json',lambda p: {'registered_roots':['registered']} if 'next_pool_cache_audit' in str(p) else [{'source_video_id':'s','existing_paths':[]}])
 monkeypatch.setattr(m,'inventory_mp4',lambda roots:{'s':['existing.mp4']})
 saved=[];monkeypatch.setattr(m,'atomic_json',lambda p,r:saved.append(r))
 m.network_batch()
 assert saved[0][0]['existing_paths']==['existing.mp4']
