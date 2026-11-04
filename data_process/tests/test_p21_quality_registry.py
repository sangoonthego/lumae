import pytest
from data_process.lumae_scale.p21_quality_registry import effective_records

def test_withdrawal_is_effective_without_mutating_trial_record():
 r={'source_video_id':'x','qid':'q','source_video_sha256':'raw','semantic_artifact_sha256':'pred','window':[1,2]}
 other={**r,'source_video_id':'y','qid':'q2'};before=dict(r)
 assert effective_records([r,other],{'x':dict(r)})==[other]
 assert r==before

def test_changed_or_out_of_cohort_withdrawal_fails_closed():
 r={'source_video_id':'x','qid':'q','source_video_sha256':'raw','semantic_artifact_sha256':'pred','window':[1,2]}
 with pytest.raises(ValueError):effective_records([r],{'x':{**r,'window':[1,3]}})
 with pytest.raises(ValueError):effective_records([r],{'historical':r})
