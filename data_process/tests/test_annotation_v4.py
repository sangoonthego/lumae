import numpy as np
import pytest
from data_process.lumae_scale.annotation_v4 import DEFAULTS,key_for,adaptive_times,rank_events,component_bounds,review_errors,CachedRuntime


def test_key_binds_model_content_preprocess_timestamps_and_text():
    args=dict(video_sha='a',encoder_sha='model1',config=DEFAULTS,timestamps=[0,1])
    base=key_for('images',**args)
    assert base!=key_for('images',**(args|{'encoder_sha':'model2'}))
    assert base!=key_for('images',**(args|{'video_sha':'b'}))
    assert base!=key_for('images',**(args|{'timestamps':[0,.5,1]}))
    assert base!=key_for('images',**(args|{'config':DEFAULTS|{'preprocess':'different'}}))
    assert key_for('text',encoder_sha='m',config=DEFAULTS,query='a')!=key_for('text',encoder_sha='m',config=DEFAULTS,query='b')


def test_dense_sampling_is_bounded_and_preserves_whole_uniform_coverage():
    times=adaptive_times(30,[{'coarse_start':10,'coarse_end':11}])
    assert set(range(0,30,2))<=set(times)
    assert 10.25 in times and 2.25 not in times
    with pytest.raises(ValueError,match='budget'):adaptive_times(150,[{'coarse_start':0,'coarse_end':149}],DEFAULTS|{'max_dense_frames_per_source':3})


def test_short_component_has_no_forced_four_second_envelope_and_stops_at_cut():
    times=np.arange(0,3,.25);features=np.zeros((len(times),512));features[:5,0]=1;features[5:,1]=1
    scores=np.ones(len(times))*.2;scores[3]=.21
    result=component_bounds(times,features,scores,(0,2))
    assert result['end']<=1.125 and result['end']-result['start']<2
    assert result['cut_pairs']


def test_invalid_or_absent_temporal_evidence_abstains():
    with pytest.raises(ValueError):component_bounds([0,0],np.zeros((2,512)),[1,1],(0,1))
    with pytest.raises(ValueError,match='No frame'):component_bounds([0,2],np.zeros((2,512)),[1,1],(.5,1))


def test_visual_review_rejects_truncation_false_semantics_and_fake_human():
    r={'human_verified':False,'actual_overview_inspected':True,'actual_qc_inspected':True,'semantic_match':True,'unique_event':True,'complete_occurrence':True,
        'reason':'Actual synthetic fixture only','onset_bracket_seconds':[1,1.25],'offset_bracket_seconds':[2,2.25]}
    assert not review_errors(r,{'start':1.125,'end':2.125},10)
    assert review_errors(r,{'start':1.5,'end':2.125},10)
    assert review_errors(r|{'semantic_match':False},{'start':1.125,'end':2.125},10)
    assert review_errors(r|{'human_verified':True},{'start':1.125,'end':2.125},10)


def test_event_rank_is_visual_not_clip_similarity_or_repeated_paraphrases():
    common={'actual_visual_support':True,'coarse_start':1,'coarse_end':2,'visible_action':'opens','visible_object':'box',
        'event_quality':dict(support=5,specificity=5,uniqueness=5,boundary_visibility=5,clarity=5)}
    events=[common|{'query':'A person opens a red cardboard box.'},common|{'query':'The person is opening a red box.'}]
    assert len(rank_events(events))==1
    assert rank_events([common|{'query':'A product appears.'}])==[]


def test_text_cache_reuses_exact_input_and_invalidates_changed_model(tmp_path):
    class Runtime:
        calls=0
        def encode_text(self,q):self.calls+=1;return np.ones((1,512),dtype=np.float32)
    raw=Runtime();cache=CachedRuntime(raw,tmp_path,'m1')
    cache.encode_text('A query');cache.encode_text('A query');assert raw.calls==1 and cache.text_hits==1
    CachedRuntime(raw,tmp_path,'m2').encode_text('A query');assert raw.calls==2
