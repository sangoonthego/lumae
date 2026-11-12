import importlib.util,json
from pathlib import Path
import pytest

@pytest.fixture
def controller(tmp_path,monkeypatch):
 spec=importlib.util.spec_from_file_location('fasttrack_test',Path(__file__).resolve().parents[2]/'scripts/d1000_agent_run.py')
 m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 monkeypatch.setattr(m,'BASE',tmp_path/'work');monkeypatch.setattr(m,'OUT',tmp_path/'out');m.BASE.mkdir();m.OUT.mkdir()
 m.atomic_json(m.BASE/'inventory.json',[])
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


def test_real_agent_acceptances_match_original_predictions():
    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
    import d1000_agent_run as m
    rows=m.accepted()
    assert rows
    for r in rows:
        p=m.read_json(Path(r['semantic_artifact_path']))
        assert r['window']==[p['start'],p['end']]
        assert r['human_verified'] is False and r['verifier']['pass'] is True
        assert r['annotation_policy']==m.POLICY


def test_reused_frame_bindings_reference_immutable_prior_cache():
    import json
    from data_process.lumae_scale.cache_validation import visual_index_valid
    root=Path('E:/LUMAE/work/d1000/d1000_agent_20261104_01/cache')
    paths=sorted(root.glob('*/metadata.json'))
    assert paths
    bound=[p for p in paths if 'd1000_fasttrack_20261104_01' in p.read_text()]
    assert bound
    for p in bound:
        r=json.loads(p.read_text())
        assert visual_index_valid(r,p.parent,r['video_sha256'])

def test_acceptance_cache_revalidates_changed_source(controller,monkeypatch):
 m=controller;raw=m.BASE/'source.mp4';raw.write_bytes(b'original')
 r={'source_video_id':'s','annotation_policy':m.POLICY,'human_verified':False,'cache_root':str(m.BASE),'source_video_path':str(raw),'source_video_sha256':m.digest(raw)}
 m.atomic_json(m.BASE/'accepted/s.json',r);calls=[]
 def validator(record,**kwargs):
  calls.append(record['source_video_id'])
  if m.digest(raw)!=record['source_video_sha256']:raise ValueError('changed source')
 monkeypatch.setattr(m,'validate_annotation',validator)
 m.accepted();m.accepted();assert len(calls)==1
 raw.write_bytes(b'changed source bytes')
 with pytest.raises(ValueError,match='changed source'):m.accepted()


@pytest.mark.parametrize("stage_historical",[False,True])
def test_handoff_loads_three_immutable_feature_roots(controller,monkeypatch,tmp_path,stage_historical):
 import numpy as np
 from experiments.M1_lumae_ads import cached_runner
 m=controller;repo=Path(__file__).resolve().parents[2]
 cls=cached_runner.dataset_type(repo/'research_outputs/p0_20261103_01/upstream')
 recipe=m.read_json(repo/'experiments/M1_lumae_ads/config_d500_cached.json')
 root=tmp_path/'repo';seed_root=tmp_path/'seed'
 monkeypatch.setattr(m,'ROOT',root);monkeypatch.setattr(m,'PREVIOUS',seed_root)
 monkeypatch.setattr(cached_runner,'dataset_type',lambda _:cls)
 historical=root/'research_outputs/p0_20261103_01/assets/feature_cache'
 parent=root/'local_data/datasets/lumae_ads_d500'
 rows=[];manifests=[];protected={}
 for name,folder in [('historical',historical),('seed',seed_root/'training_features'),('current',m.BASE/'training_features')]:
  folder.mkdir(parents=True,exist_ok=True)
  record={'qid':name,'source_video_id':name,'query':'A person opens a wooden box.','duration':8.,'window':[2.,4.],'source_video_path':name+'.mp4'}
  rows.append(record)
  np.savez(folder/'video.npz',features=np.ones((4,512),dtype=np.float32))
  np.savez(folder/'text.npz',last_hidden_state=np.ones((3,512),dtype=np.float32))
  checks={p.name:m.digest(p) for p in folder.glob('*.npz')}
  protected.update({str(p):m.digest(p) for p in folder.glob('*.npz')})
  manifests.append({'video':{'lumae_ads_'+name:'video.npz'},'text':{name:'text.npz'},'checksums':checks,'feature_contract':{'tef_in_cache':False}})
 old,seed,new=manifests
 m.atomic_json(historical/'feature_manifest.json',old)
 if stage_historical:
  import shutil
  staging=m.BASE/'historical_training_features'
  shutil.copytree(historical,staging)
  m.atomic_json(m.OUT/'drive_d500_feature_manifest.json',old)
  ref={'id':'verified-drive-id','relative_path':'video.npz','size':str((staging/'video.npz').stat().st_size)}
  m.atomic_json(m.OUT/'drive_historical_feature_refs.json',{'video':{'lumae_ads_historical':ref}})
  m.atomic_json(staging/'provenance/video/lumae_ads_historical.json',{
   **ref,'bytes':int(ref['size']),'sha256':old['checksums']['video.npz'],'historical_reencoded':False,
  })
 seed_manifest=root/'research_outputs/d1000_fasttrack_20261104_01/new_feature_manifest.json';m.atomic_json(seed_manifest,seed)
 new['prior_additions_binding']={'manifest_sha256':m.digest(seed_manifest)}
 m.atomic_json(m.OUT/'new_feature_manifest.json',new)
 m.atomic_json(root/'experiments/M1_lumae_ads/config_d500_cached.json',recipe)
 prior=m.BASE/'prior.jsonl';m.jsonl(prior,[rows[1]]);m.atomic_json(m.BASE/'config.json',{'prior':str(prior)})
 canon=[m.canonical_row(r,r['qid']) for r in rows]
 m.jsonl(parent/'lumae_ads_d500_all.jsonl',canon[:1]);m.jsonl(parent/'lumae_ads_d500_train.jsonl',canon[:1]);m.jsonl(parent/'lumae_ads_d500_val.jsonl',[])
 m.jsonl(m.OUT/'accepted_annotations.jsonl',rows[1:]);m.jsonl(m.OUT/'provisional_dataset.jsonl',canon)
 monkeypatch.setattr(m,'progress',lambda:{'total':3,'complete':False});monkeypatch.setattr(m,'accepted',lambda:rows[2:])
 result=m.handoff()
 assert result['existing_cached_dataset_loaded']=={'D500_available_subset':1,'prior_additions':1,'additions':1}
 assert result['pass'] and result['visual_input_dim_with_tef']==514 and result['text_dim']==512
 unified=m.read_json(m.OUT/'unified_feature_manifest.json')
 assert set(unified['video'])=={'lumae_ads_'+r['source_video_id'] for r in rows}
 assert all(not Path(p).is_absolute() and '..' not in Path(p).parts for kind in ('video','text') for p in unified[kind].values())
 assert all(m.digest(Path(p))==h for p,h in protected.items())
 if stage_historical:
  progress=m.read_json(m.OUT/'historical_feature_materialization_progress.json')
  assert progress['content_validation_complete'] and progress['validated_drive_downloads']==1
  assert progress['unchanged_local_files_copied']==1
  assert Path(unified['local_bindings']['historical_root'])==staging
  (staging/'video.npz').write_bytes((staging/'video.npz').read_bytes()+b'tampered')
  with pytest.raises(AssertionError):m.handoff()
