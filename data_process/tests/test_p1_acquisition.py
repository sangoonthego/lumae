"""Fault injection for P1; HTTP/video mocks never count as live-pilot evidence."""
import copy
import json
from pathlib import Path
import pytest
import requests
from data_process.lumae_scale import acquisition as a
from data_process.lumae_scale.builder import BuildConfig
from data_process.lumae_scale.models import atomic_json,read_json,digest
from data_process.tests.test_general_builder import parent_fixture,pool

class Response:
    def __init__(self,status=200,chunks=(b'video',),headers=None,error=None):
        self.status_code=status;self.chunks=chunks;self.headers=headers or {};self.error=error
    def __enter__(self):return self
    def __exit__(self,*_):pass
    def close(self):pass
    def iter_content(self,*_):
        yield from self.chunks
        if self.error:raise self.error

class HTTP:
    def __init__(self,responses):self.responses=iter(responses);self.calls=[]
    def get(self,url,**kwargs):self.calls.append((url,kwargs));return next(self.responses)

def valid(_):return {'pass':True,'reason':None,'probe':{'duration':20},'fingerprint':{'algorithm':'three_frame_dhash64_v1','hashes':['0'*16]*3,'duration':20}}

@pytest.fixture
def setup(tmp_path,monkeypatch):
    monkeypatch.setattr(a,'ROOT',tmp_path/'repository')
    monkeypatch.setattr(a.time,'sleep',lambda _:None)
    parent=tmp_path/'parent';parent_fixture(parent,1);root=tmp_path/'E';root.mkdir();gate=tmp_path/'gate.json'
    atomic_json(gate,{'protected_assets_pass':True,'storage_pass':True,'source_permissions_attested':True,'crawler_tests_pass':True,'source_manifest_sha256':'abc'})
    cfg={'run_id':'synthetic_test','mode':'pilot','data_root':str(root),'target_accepted':25,'max_candidates':30,'max_http_requests':45,'max_download_bytes':1000,'max_file_bytes':500,'max_runtime_seconds':1000,'max_staging_bytes':1000000,'min_free_bytes':1,'max_consecutive_failures':3,'retries_per_source':2,'timeout_seconds':10,'request_interval_seconds':1,'concurrency':1,'max_domain_error_rate':.6,'domain_error_min_requests':5,'network_execution_authorized':True,'source_permission':{'status':'OPERATOR_ATTESTED','evidence':'Synthetic fixture only','host':a.HOST},'operational_gate':str(gate),'source_manifest_sha256':'abc'}
    build=BuildConfig('next','parent',parent,root,target_videos=1000)
    return cfg,build,pool(35)

def make(setup,responses,live=True):
    c,b,p=setup;http=HTTP(responses);pilot=a.Pilot(c,b,p,live=live,session=http,validator=valid)
    if live:pilot.initialize()
    return pilot,http

def test_network_denied_dry_run_creates_nothing(setup):
    p,http=make(setup,[],False)
    with pytest.raises(PermissionError):p.fetch(setup[2][0])
    with pytest.raises(PermissionError):p.run()
    assert not p.work.exists() and not http.calls

def test_interrupted_download_range_resume_and_completed_idempotency(setup):
    row=setup[2][0]
    p,http=make(setup,[Response(chunks=[b'abc'],headers={'ETag':'"stable"','Content-Length':'6'},error=requests.ConnectionError()),Response(206,[b'def'],{'ETag':'"stable"','Content-Range':'bytes 3-5/6','Content-Length':'3'})])
    with pytest.raises(requests.ConnectionError):p.fetch(row)
    path=p.fetch(row)
    assert path.read_bytes()==b'abcdef'
    assert http.calls[1][1]['headers']['Range']=='bytes=3-'
    assert http.calls[1][1]['headers']['If-Range']=='"stable"'
    assert p.fetch(row)==path and len(http.calls)==2
    assert sum(r['bytes_received'] for r in p.attempts())==6

@pytest.mark.parametrize('response',[Response(206,[b'abc'],{'Content-Range':'bytes 2-4/5'}),Response(200,[b'html'],{'Content-Type':'text/html'}),Response(403),Response(302,[],{'Location':'https://evil.example/file'})])
def test_invalid_range_html_access_denial_and_redirect_are_terminal(setup,response):
    p,http=make(setup,[response])
    with pytest.raises(a.PermanentSource):p.fetch(setup[2][0])
    assert len(http.calls)==1 and not list((p.work/'downloads').glob('*.mp4'))

def test_ignored_range_replaces_partial_not_append(setup):
    row=setup[2][0];p,http=make(setup,[Response(200,[b'complete'],{'Content-Length':'8'})]);part=p.work/'downloads'/(row['target_name']+'.part');part.write_bytes(b'bad')
    atomic_json(part.with_suffix('.identity.json'),{'etag':'"old"'})
    assert p.fetch(row).read_bytes()==b'complete'

def test_stream_limit_stops_before_overrun(setup):
    setup[0]['max_download_bytes']=4;p,http=make(setup,[Response(200,[b'abc',b'def'])])
    with pytest.raises(a.BudgetStop):p.fetch(setup[2][0])
    assert sum(x['bytes_received'] for x in p.attempts())==3
    assert next((p.work/'downloads').glob('*.part')).read_bytes()==b'abc'

def test_content_length_truncation_does_not_promote(setup):
    p,http=make(setup,[Response(200,[b'abc'],{'Content-Length':'8'})])
    with pytest.raises(requests.ConnectionError):p.fetch(setup[2][0])
    assert not list((p.work/'downloads').glob('*.mp4'))

def test_quarantine_resume_rebuild_and_manifest_tamper(setup):
    p,http=make(setup,[Response(200,[b'real mock bytes'])]);first=p.run(max_wave=1)
    assert first['quarantined_candidates']==1 and first['accepted_unique_videos']==0
    assert first['annotation_accepted']==0
    # Same source is never requested again; next wave processes the next ID.
    p2,http2=make(setup,[Response(200,[b'other mock'])]);second=p2.run(max_wave=1)
    assert second['quarantined_candidates']==2 and len(http2.calls)==1
    record=p2.records()[0];path=p2.root/record['path'];path.write_bytes(b'changed')
    with pytest.raises(RuntimeError,match='Completed source changed'):p2.run(max_wave=1)

def test_terminal_access_failures_stop_at_consecutive_threshold(setup):
    p,http=make(setup,[Response(403),Response(403),Response(403),Response(200)])
    result=p.run()
    assert result['http_requests']==3 and result['rejected_videos']==3
    assert result['stop_reason']=='Consecutive failed request threshold'

def test_disk_reserve_and_config_identity_fail_closed(setup,monkeypatch):
    p,http=make(setup,[])
    monkeypatch.setattr(a.shutil,'disk_usage',lambda _:type('Usage',(),{'free':0})())
    with pytest.raises(a.BudgetStop):p.check_limits()
    monkeypatch.undo()
    setup[0]['target_accepted']=26
    with pytest.raises(ValueError):p.initialize()

def test_source_permission_and_gate_required(setup):
    c,b,rows=setup;c['source_permission']['evidence']=''
    p=a.Pilot(c,b,rows,live=True)
    with pytest.raises(PermissionError):p.initialize()
    assert not p.work.exists()

def test_exact_duplicate_rejected_and_campaign_never_assumed(setup):
    p,http=make(setup,[]);row=setup[2][0];f=p.work/'downloads'/row['target_name'];f.write_bytes(b'duplicate')
    p.parent_hashes.add(digest(f));r=p.process(row,f)
    assert r['status']=='REJECTED' and r['reason']=='EXACT_CONTENT_DUPLICATE'

def test_incomplete_exports_recover_from_authoritative_records(setup):
    p,http=make(setup,[Response()]);p.run(max_wave=1)
    (p.work/'manifests/pilot_download_manifest.json').write_text('{partial')
    metrics=p.export();assert metrics['quarantined_candidates']==1
    assert read_json(p.work/'manifests/pilot_download_manifest.json')['metrics']==metrics

def test_unreviewed_source_cannot_accept_annotation(setup):
    p,http=make(setup,[Response()]);p.run(max_wave=1);sid=p.records()[0]['source_video_id']
    with pytest.raises(ValueError,match='review must pass'):p.accept_annotation({'source_video_id':sid})

def test_fingerprints_flag_similar_frames_not_independence():
    from data_process.lumae_scale.pilot_quality import fingerprint_matches
    fp={'algorithm':'three_frame_dhash64_v1','hashes':['000000000000000f']}
    assert fingerprint_matches(fp,[{'source_video_id':'a','fingerprint':{'algorithm':fp['algorithm'],'hashes':['0000000000000000']}}])[0]['distance']==4

def test_previous_pilot_binding_excludes_sources_and_detects_tamper(setup):
    p,http=make(setup,[Response()]);p.run(max_wave=1)
    prior=p.work/'manifests/pilot_download_manifest.json';sid=p.records()[0]['source_video_id']
    c=copy.deepcopy(setup[0]);c['run_id']='next_stage';c['previous_runs']=[{'manifest':prior.relative_to(p.root).as_posix(),'sha256':digest(prior)}]
    next_run=a.Pilot(c,setup[1],setup[2])
    assert sid not in {r['source_video_id'] for r in next_run.plan['candidates']}
    assert next_run.plan['previous_accepted_sources']==0
    prior.write_text('{}')
    with pytest.raises(ValueError,match='Prior pilot manifest changed'):a.Pilot(c,setup[1],setup[2])

def test_complete_baseline_cannot_omit_parent_hashes(setup,tmp_path):
    c=copy.deepcopy(setup[0]);baseline=tmp_path/'baseline.json'
    atomic_json(baseline,{'complete':True,'videos':[],'dataset_sha256':'fake'})
    c['baseline_fingerprints']=str(baseline)
    with pytest.raises(ValueError,match='full frozen parent'):a.Pilot(c,setup[1],setup[2])
