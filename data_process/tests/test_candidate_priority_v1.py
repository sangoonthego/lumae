from data_process.lumae_scale.candidate_priority_v1 import rank_candidates,score_candidate

def test_postdownload_and_split_cannot_change_ranking():
    a={'source_video_id':'x','source_url':'https://video.adsoftheworld.com/a.mp4'}
    assert score_candidate(a)==score_candidate({**a,'source_split':'test','accepted':True,'window':[1,2],'clip_score':999})

def test_equal_prior_deterministic_and_domain_gate():
    a=[{'source_video_id':s,'source_url':'https://video.adsoftheworld.com/'+s} for s in ('b','a')]
    assert rank_candidates(a)==rank_candidates(list(reversed(a)))
    assert all(r['priority_score']==0 and not r['semantic_quality_prediction_available'] for r in rank_candidates(a))
    assert not score_candidate({'source_video_id':'bad','source_url':'https://example.com/x'})['eligible']
