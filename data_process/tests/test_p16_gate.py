"""The new gate may waive yield20, but never reconciliation or identity checks."""
import pytest
from data_process.lumae_scale.acquisition import validate_scale_gate
from data_process.lumae_scale.models import atomic_json,digest
from data_process.tests.test_p1_acquisition import setup,make,Response


def gate_fixture(tmp_path):
    artifact=tmp_path/'pilot30_recovery_metrics.json'
    atomic_json(artifact,{'accepted_unique_new_pairs':7,'human_verified':0})
    freeze=tmp_path/'freeze.json'
    atomic_json(freeze,{'final_d1000_freeze':False,'artifact_sha256':{artifact.name:digest(artifact)}})
    gate=tmp_path/'engineering_gate.json'
    values={'schema':'LUMAE_P16_ENGINEERING_V1','engineering_pass':True,'critical_failure':False,
            'reconciled_candidates':30,'prior_annotations_reaudited':16,'failed_sources_recovery_attempted':13,
            'freeze_path':str(freeze),'freeze_sha256':digest(freeze),'quality_signal_20_pairs':False}
    atomic_json(gate,values)
    return {'engineering_gate':{'path':str(gate),'sha256':digest(gate)}},gate,values,artifact


def test_low_yield_requires_complete_intact_engineering_gate(tmp_path):
    config,gate,values,artifact=gate_fixture(tmp_path)
    validate_scale_gate(config)
    atomic_json(artifact,{'accepted_unique_new_pairs':20})
    with pytest.raises(PermissionError,match='artifact changed'):validate_scale_gate(config)


@pytest.mark.parametrize('field,value', [('reconciled_candidates',29),('prior_annotations_reaudited',15),
    ('failed_sources_recovery_attempted',12),('critical_failure',True),('engineering_pass',False)])
def test_incomplete_or_failed_gate_cannot_be_used(tmp_path,field,value):
    config,gate,values,artifact=gate_fixture(tmp_path)
    values[field]=value;atomic_json(gate,values);config['engineering_gate']['sha256']=digest(gate)
    with pytest.raises(PermissionError):validate_scale_gate(config)


def test_gate_hash_and_legacy_failure_are_enforced(tmp_path):
    config,gate,values,artifact=gate_fixture(tmp_path)
    values['engineering_pass']=False;atomic_json(gate,values)
    with pytest.raises(PermissionError,match='identity changed'):validate_scale_gate(config)
    with pytest.raises(PermissionError):validate_scale_gate({})


def test_uncertain_campaign_retention_preserves_test_exclusion(setup,tmp_path):
    config,gate,values,artifact=gate_fixture(tmp_path)
    setup[0].update(config)
    p,http=make(setup,[Response()]);p.run(max_wave=1)
    p.baseline['complete']=True  # Synthetic-only review fixture, no corpus assets.
    sid=p.records()[0]['source_video_id']
    evidence=p.work/'reviews/actual_fixture.json';atomic_json(evidence,{'observed':'synthetic test only'})
    review={'reviewer_id':'AI_PIPELINE','reviewed_at':'fixture','evidence_paths':['reviews/actual_fixture.json'],
        'campaign_id':'unknown_fixture','decision':'RETAIN_SPLIT_RESTRICTED','comparison_scope':'D500_AND_PILOT',
        'relationship':'UNKNOWN','fresh_test_eligible':True}
    with pytest.raises(ValueError,match='Test exclusion'):p.review_source(sid,review)
    review['fresh_test_eligible']=False;p.review_source(sid,review)
    record=p.records()[0]
    assert record['status']=='SOURCE_ACCEPTED'
    assert record['campaign_status']=='RETAINED_SPLIT_RESTRICTED'
    assert record['fresh_test_eligible'] is False
    assert record['split_eligibility']=='RESTRICTED_PENDING_CAMPAIGN_GROUP_REVIEW'
    assert record['annotation_status']!='AI_PSEUDO_LABELED'
