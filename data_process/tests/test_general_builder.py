"""Offline synthetic acceptance and D1000 construction; no network/GPU."""
import copy
import hashlib
import json
from pathlib import Path
import pytest
from data_process.lumae_scale.builder import BuildConfig,BuilderSession,candidate_plan,parent_snapshot
from data_process.lumae_scale.incremental import json_bytes
from data_process.lumae_scale.models import digest


def parent_fixture(path,n):
    path.mkdir(parents=True);rows=[];sources={}
    for i in range(n):
        sid=f'{i:032x}';qid=f'parent_{i}';h=hashlib.sha256(f'parent-{i}'.encode()).hexdigest()
        rows.append({'qid':qid,'vid':'lumae_ads_'+sid,'query':f'Original event {i}','duration':20,'relevant_windows':[[2,6]],'saliency_scores':None,'relevant_clip_ids':None,
            'metadata':{'source_video_id':sid,'source_video_sha256':h,'source_url':f'https://video.adsoftheworld.com/{sid}.mp4','review_provenance':'AI_PSEUDO_LABELED','video_filename':sid+'.mp4'}})
        sources[qid]={'sha256':h,'video_filename':sid+'.mp4'}
    for split,part in [('all',rows),('train',rows),('val',[]),('test',[])]:
        (path/f'parent_{split}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in part))
    (path/'dataset_manifest.json').write_text(json.dumps({'total_samples':n,'dataset_sha256':digest(path/'parent_all.jsonl'),'source_video_sha256':sources}))
    (path/'checksums.sha256').write_text(''.join(f'{digest(p)}  {p.name}\n' for p in sorted(path.iterdir())))
    return rows

def pool(n,start=1000):
    return [{'source_video_id':f'{i:032x}','target_name':f'{i:032x}.mp4','source_url':f'https://video.adsoftheworld.com/{i:032x}.mp4'} for i in range(start,start+n)]

def probe(_):return {'is_model_compatible':True,'duration':20}

def validate(record,**kwargs):
    # Synthetic provider only; production default invokes full frozen annotation validator.
    assert record['review_provenance']=='AI_PSEUDO_LABELED'

def record_for(cfg,source):
    sid=source['source_video_id'];p=cfg.videos/(sid+'.mp4');p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(('synthetic video '+sid).encode())
    return {'source_video_id':sid,'source_url':source['source_url'],'source_video_path':p.relative_to(cfg.data_root).as_posix(),
        'source_video_sha256':digest(p),'query':'A person raises a bottle '+sid,'duration':20.,'window':[2,6],
        'review_provenance':'AI_PSEUDO_LABELED','query_source':'AI_GENERATED','temporal_label_source':'SEMANTIC_V3_PSEUDO_LABEL'}

def test_d1000_deterministic_manifest_and_frozen_parent(tmp_path):
    parent=tmp_path/'parent';old=parent_fixture(parent,500);before={p.name:p.read_bytes() for p in parent.iterdir()};sources=pool(500)
    manifests=[]
    for target in [tmp_path/'first',tmp_path/'second']:
        cfg=BuildConfig('synthetic_d1000','parent',parent,target,split_policy='historical_extension');plan=candidate_plan(cfg,sources)
        assert plan['new_accepted_required']==500 and plan['download_started'] is False
        assert plan==candidate_plan(cfg,list(reversed(sources)))
        session=BuilderSession(cfg,sources)
        # Import a synthetic, already-reviewed cohort; acceptance/resume separately tested below.
        for src in sources:
            r=record_for(cfg,src);sid=r['source_video_id'];dest=cfg.work/'accepted'/(sid+'.json');dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(json_bytes(r))
            session.accepted[sid]=r;session.accepted_digests[sid]=digest(dest)
        manifest=session.freeze(annotation_validator=validate);assert manifest['target_videos']==1000 and manifest['counts']=={'train':1000,'val':0,'test':0}
        manifests.append((cfg.dataset/'dataset_manifest.json').read_bytes())
        assert session.freeze(annotation_validator=validate)==manifest
        feature=json.loads((cfg.dataset/'feature_manifest.json').read_text());assert len(feature['features'])==1000
        assert all(not f['ready'] for f in feature['features'])
    assert manifests[0]==manifests[1]
    assert {p.name:p.read_bytes() for p in parent.iterdir()}==before

def test_resume_idempotency_retry_and_duplicate_guards(tmp_path):
    parent=tmp_path/'parent';parent_fixture(parent,1);cfg=BuildConfig('next','parent',parent,tmp_path/'store',target_videos=3,split_policy='historical_extension');sources=pool(3)
    session=BuilderSession(cfg,sources);r=record_for(cfg,sources[0]);sid=r['source_video_id']
    with pytest.raises(PermissionError):session.fetch(sid)
    def interrupted(*args,**kwargs):raise TimeoutError('interrupted download')
    with pytest.raises(TimeoutError):session.fetch(sid,allow_network=True,downloader=interrupted)
    assert session.states[sid]=='download_deferred'
    restarted=BuilderSession(cfg,sources);assert restarted.attempts[sid]==1
    restarted.prepare_existing(sid,cfg.data_root/r['source_video_path'],probe=probe)
    bad={**r,'window':[-1,4]}
    with pytest.raises(ValueError,match='timestamp'):restarted.accept(bad,annotation_validator=validate)
    restarted.accept(r,annotation_validator=validate);size=restarted.path.stat().st_size
    restarted.accept(copy.deepcopy(r),annotation_validator=validate);assert restarted.path.stat().st_size==size
    again=BuilderSession(cfg,sources);assert again.accepted[sid]==r
    duplicate=record_for(cfg,sources[1]);duplicate_path=cfg.data_root/duplicate['source_video_path'];duplicate_path.write_bytes((cfg.data_root/r['source_video_path']).read_bytes())
    with pytest.raises(ValueError,match='Duplicate exact'):again.prepare_existing(sources[1]['source_video_id'],duplicate_path,probe=probe)
    fresh=record_for(cfg,sources[2]);p=cfg.data_root/fresh['source_video_path']
    with pytest.raises(ValueError,match='Corrupt'):again.prepare_existing(sources[2]['source_video_id'],p,probe=lambda p:{'is_model_compatible':False})
    with pytest.raises(ValueError,match='Missing'):again.prepare_existing(sources[2]['source_video_id'],cfg.data_root/'missing.mp4',probe=probe)
    again.reject(sources[2]['source_video_id'],'no usable event');assert BuilderSession(cfg,sources).states[sources[2]['source_video_id']]=='rejected'
    changed={**r,'query':'rewritten'}
    with pytest.raises(ValueError,match='Immutable'):again.accept(changed,annotation_validator=validate)

def test_duplicate_url_and_id_selection_and_parent_tamper(tmp_path):
    parent=tmp_path/'parent';parent_fixture(parent,1);cfg=BuildConfig('next','parent',parent,tmp_path/'store',target_videos=3)
    a,b=pool(2);duplicate={**b,'source_url':a['source_url']+'#fragment'}
    plan=candidate_plan(cfg,[a,a,duplicate]);assert len(plan['candidates'])==1
    assert {x['reason'] for x in plan['excluded']}=={'duplicate_source_id','duplicate_source_url'}
    with pytest.raises(ValueError):BuildConfig('lumae_ads_d500','parent',parent,tmp_path/'store')
    session=BuilderSession(cfg,[a,b]);(parent/'parent_all.jsonl').write_text('tampered')
    with pytest.raises(RuntimeError,match='parent changed'):session.guard()

def test_default_annotation_validator_rejects_missing_evidence(tmp_path):
    parent=tmp_path/'parent';parent_fixture(parent,1);cfg=BuildConfig('next','parent',parent,tmp_path/'store',target_videos=2);src=pool(1)[0]
    session=BuilderSession(cfg,[src]);r=record_for(cfg,src);session.prepare_existing(src['source_video_id'],cfg.data_root/r['source_video_path'],probe=probe)
    with pytest.raises((ValueError,KeyError)):session.accept(r)
    assert not session.accepted

def test_inherited_rejection_and_http_range_resume(tmp_path,monkeypatch):
    from data_process.lumae_scale import source_pool
    parent=tmp_path/'parent';parent_fixture(parent,1);sources=pool(2)
    (parent/'selection_manifest.json').write_text(json.dumps({'candidates':[{'source_video_id':sources[1]['source_video_id'],'status':'rejected','rejection_reason':'no visual event'}]}))
    cfg=BuildConfig('next','parent',parent,tmp_path/'store',target_videos=2);session=BuilderSession(cfg,sources)
    assert [r['source_video_id'] for r in session.plan['candidates']]==[sources[0]['source_video_id']]
    assert session.plan['excluded'][0]['reason']=='inherited_terminal_rejection'
    calls=[]
    class Response:
        def __init__(self,interrupted):self.interrupted=interrupted;self.status_code=200 if interrupted else 206;self.headers={} if interrupted else {'Content-Range':'bytes 3-5/6'}
        def __enter__(self):return self
        def __exit__(self,*args):return False
        def raise_for_status(self):pass
        def iter_content(self,bs):
            yield b'abc' if self.interrupted else b'def'
            if self.interrupted:raise TimeoutError('synthetic interrupted transfer')
    def fake_get(url,**kwargs):calls.append(kwargs['headers']);return Response(len(calls)==1)
    monkeypatch.setattr(source_pool.requests,'get',fake_get)
    sid=sources[0]['source_video_id']
    with pytest.raises(TimeoutError):session.fetch(sid,allow_network=True,probe=probe)
    part=cfg.videos/(sid+'.mp4.part');assert part.read_bytes()==b'abc'
    resumed=BuilderSession(cfg,sources);resumed.fetch(sid,allow_network=True,probe=probe)
    assert calls==[{}, {'Range':'bytes=3-'}]
    assert (cfg.videos/(sid+'.mp4')).read_bytes()==b'abcdef'

def test_campaign_freeze_seals_test_and_feature_references(tmp_path):
    from data_process.splitting.campaign import group_campaigns,deterministic_campaign_split
    parent=tmp_path/'parent';base=parent_fixture(parent,2);cfg=BuildConfig('next','parent',parent,tmp_path/'store',target_videos=4);sources=pool(2)
    session=BuilderSession(cfg,sources);new=[]
    for src in sources:
        r=record_for(cfg,src);session.prepare_existing(r['source_video_id'],cfg.data_root/r['source_video_path'],probe=probe);session.accept(r,annotation_validator=validate);new.append(r)
    videos=[{'vid':r['vid'],'source_id':r['metadata']['source_video_id'],'source_sha256':r['metadata']['source_video_sha256'],'source_url':r['metadata']['source_url'],'historically_exposed':True,'campaign_status':'SINGLETON_REVIEWED'} for r in base]
    videos += [{'vid':'lumae_ads_'+r['source_video_id'],'source_id':r['source_video_id'],'source_sha256':r['source_video_sha256'],'source_url':r['source_url'],'historically_exposed':False,'campaign_status':'SINGLETON_REVIEWED'} for r in new]
    plan=deterministic_campaign_split(group_campaigns(videos),{'train':2,'val':1,'test':1})
    bad=copy.deepcopy(plan);test_vid=next(v for v,s in bad['assignments'].items() if s=='test');train_vid=next(v for v,s in bad['assignments'].items() if s=='train');bad['grouping']['video_to_group'][train_vid]=bad['grouping']['video_to_group'][test_vid]
    with pytest.raises(ValueError,match='Campaign crosses'):session.freeze(bad['assignments'],split_plan=bad,annotation_validator=validate)
    manifest=session.freeze(plan['assignments'],split_plan=plan,annotation_validator=validate);assert manifest['counts']=={'train':2,'val':1,'test':1}
    training=json.loads((cfg.dataset/'training_asset_manifest.json').read_text());test_rows=json.loads('['+','.join((cfg.dataset/'next_test.jsonl').read_text().splitlines())+']')
    assert not ({r['qid'] for r in test_rows}&set(training['queries']))
    assert {f['qid'] for f in training['features']}==set(training['queries'])
    assert (cfg.dataset/'campaign_grouping.json').is_file()

def test_full_annotation_validator_accepts_configured_root_and_detects_tamper(tmp_path):
    from data_process.annotation.assisted_deployment import ALGORITHM_SHA256
    parent=tmp_path/'parent';parent_fixture(parent,1);cfg=BuildConfig('next','parent',parent,tmp_path/'store',target_videos=2,split_policy='historical_extension');src=pool(1)[0]
    session=BuilderSession(cfg,[src]);r=record_for(cfg,src);sid=r['source_video_id'];r['query']='A person lifts a red bottle from the table.'
    session.prepare_existing(sid,cfg.data_root/r['source_video_path'],probe=probe)
    visual=cfg.visual_cache/sid;visual.mkdir(parents=True);(visual/'contact.jpg').write_bytes(b'synthetic image fixture')
    (visual/'metadata.json').write_text(json.dumps({'video_id':sid}))
    verification=cfg.work/'verification.jpg';verification.write_bytes(b'synthetic verifier image')
    semantic=cfg.work/'semantic.json';semantic.write_text(json.dumps({'query':r['query'],'start':2,'end':6,'semantic_v3_algorithm_sha256':ALGORITHM_SHA256}));original=semantic.read_bytes()
    r.update(query_sanitation={'passed':True,'errors':[]},approximate_event_phase_seconds=[0,12],visual_evidence={'contact_sheets':['contact.jpg']},verifier={'pass':True},
        visual_index_sha256=digest(visual/'metadata.json'),verification_image_path=verification.relative_to(cfg.data_root).as_posix(),verification_image_sha256=digest(verification),
        semantic_artifact_path=semantic.relative_to(cfg.data_root).as_posix(),semantic_artifact_sha256=digest(semantic),semantic_v3_prediction_sha256=digest(semantic),
        query_review_status='AI_AUTO_ACCEPTED_FOR_TRAINING',reviewer_id='AI_PIPELINE',semantic_v3_algorithm_sha256=ALGORITHM_SHA256,query_provider_config={'provider':'synthetic fixture; not real data'})
    session.accept(r)
    semantic.write_text('tampered')
    with pytest.raises(ValueError,match='semantic'):session.freeze()
    semantic.write_bytes(original);assert session.freeze()['target_videos']==2
