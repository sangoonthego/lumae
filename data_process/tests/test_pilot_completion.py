import pytest
from data_process.lumae_scale.pilot_completion import freeze_completion,validate_completion

def test_pilot_freeze_repetition_is_identical_and_tamper_refused(tmp_path):
    manifest={'final_d1000_freeze':False,'accepted_annotations':[],'original_candidates':30}
    a=freeze_completion(tmp_path,manifest);before=(tmp_path/'pilot30_freeze_manifest.json').read_bytes()
    assert freeze_completion(tmp_path,manifest)==a
    assert (tmp_path/'pilot30_freeze_manifest.json').read_bytes()==before
    with pytest.raises(ValueError):freeze_completion(tmp_path,{**manifest,'original_candidates':31})

def test_pilot_cannot_be_claimed_final_d1000(tmp_path):
    with pytest.raises(ValueError,match='D1000'):freeze_completion(tmp_path,{'final_d1000_freeze':True})

def test_missing_original_candidate_fails_closed(tmp_path):
    with pytest.raises(ValueError,match='original30'):validate_completion([],[],root=tmp_path,visual_cache=tmp_path)

def test_human_promotion_blocked_before_artifact_access(tmp_path):
    inventory=[{'source_id':str(i),'technical_pass':True,'sha256':'abc'} for i in range(30)]
    accepted=[{'source_video_id':'0','qid':'q','review_provenance':'HUMAN_VERIFIED','human_verified':True}]
    with pytest.raises(ValueError,match='human'):validate_completion(inventory,accepted,root=tmp_path,visual_cache=tmp_path)

def test_interrupted_acceptance_uses_completion_cache_binding(tmp_path,monkeypatch):
    # Isolated fault injection at atomic accepted-file / ledger-event boundary.
    from data_process.lumae_scale import builder
    from data_process.lumae_scale.pilot_completion import CompletionConfig
    from data_process.tests.test_general_builder import parent_fixture,pool,record_for,probe
    from data_process.lumae_scale.incremental import immutable_write,json_bytes
    parent=tmp_path/'parent';parent_fixture(parent,1);cfg=CompletionConfig('recovery','parent',parent,tmp_path/'store',target_videos=2)
    sources=pool(1);session=builder.BuilderSession(cfg,sources);record=record_for(cfg,sources[0]);sid=record['source_video_id']
    session.prepare_existing(sid,cfg.data_root/record['source_video_path'],probe=probe)
    session._event(sid,'visual_index_ready');session._event(sid,'annotation_candidate_ready')
    immutable_write(cfg.work/'accepted'/(sid+'.json'),json_bytes(record))
    calls=[]
    def validator(value,*,visual_cache,root):
        assert visual_cache==cfg.data_root/'work/d1000/recovery/visual_index'
        assert root==cfg.data_root;calls.append(value['source_video_id'])
    monkeypatch.setattr(builder,'validate_annotation',validator)
    restored=builder.BuilderSession(cfg,sources);size=restored.path.stat().st_size
    assert calls==[sid] and restored.states[sid]=='accepted'
    assert builder.BuilderSession(cfg,sources).path.stat().st_size==size
