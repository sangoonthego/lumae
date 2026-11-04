"""Independent oracle comes from the original E0 notebook AST, not the runner."""
import ast
import json
import os
from pathlib import Path
import re

import numpy as np
import pytest
import torch

from experiments.M1_lumae_ads import cached_runner as runner


def test_manifest_path_and_selection():
    with pytest.raises(ValueError):runner.feature_path(Path.cwd(),'../outside.npz')
    with pytest.raises(ValueError):runner.feature_path(Path.cwd(),str(Path.cwd()/'absolute.npz'))
    assert runner.improves({'R1@0.5':10,'mIoU':5},None)
    assert runner.improves({'R1@0.5':10,'mIoU':6},(10,5))
    assert not runner.improves({'R1@0.5':10,'mIoU':5},(10,5))
    assert not runner.improves({'R1@0.5':9,'mIoU':100},(10,5))


@pytest.fixture
def real_assets():
    p=os.getenv('LUMAE_P0_ASSETS')
    if not p:pytest.skip('Set LUMAE_P0_ASSETS to verified versioned M0/cache/upstream assets')
    return Path(p).resolve()


def notebook_nodes(path,names):
    result={}
    for source in re.split(r'^CELL \d+\s*$',path.read_text(encoding='utf-8'),flags=re.M):
        try:tree=ast.parse(source)
        except SyntaxError:continue
        for n in tree.body:
            if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names:result[n.name]=n
    assert set(result)==set(names)
    return result


def oracle_namespace(assets,tmp_path,rows):
    from moment_detr.start_end_dataset import StartEndDataset,start_end_collate,prepare_batch_inputs
    from moment_detr.span_utils import span_cxw_to_xx
    from utils.basic_utils import l2_normalize_np_array
    from standalone_eval.eval import compute_mr_r1,compute_mr_ap
    root=Path(__file__).resolve().parents[2]
    nodes=notebook_nodes(root/'audit_outputs/2026-11-02/e0_notebook_sources.txt',
                         ['LumaeD500Dataset','temporal_iou','evaluate_val'])
    ns=dict(torch=torch,np=np,Path=Path,StartEndDataset=StartEndDataset,l2_normalize_np_array=l2_normalize_np_array,
        TEXT_KEY='last_hidden_state',VIDEO_KEY='features',DataLoader=torch.utils.data.DataLoader,
        start_end_collate=start_end_collate,prepare_batch_inputs=prepare_batch_inputs,
        span_cxw_to_xx=span_cxw_to_xx,compute_mr_r1=compute_mr_r1,compute_mr_ap=compute_mr_ap,
        DEVICE=torch.device('cpu'))
    exec(compile(ast.fix_missing_locations(ast.Module(body=list(nodes.values()),type_ignores=[])),str(root/'03_UNUSED_not_executed'),'exec'),ns)
    manifest=runner.read(assets/'assets/feature_cache/feature_manifest.json');cache=assets/'assets/feature_cache'
    for r in rows:
        for kind,identity,outname in [('video',r['vid'],r['vid']),('text',str(r['qid']),str(r['qid']))]:
            dest=tmp_path/kind/(outname+'.npz');dest.parent.mkdir(parents=True,exist_ok=True)
            dest.write_bytes((cache/manifest[kind][identity]).read_bytes())
    def dataset(path):
        return ns['LumaeD500Dataset'](dset_name='hl',data_path=str(path),v_feat_dirs=[str(tmp_path/'video')],q_feat_dir=str(tmp_path/'text'),
            q_feat_type='last_hidden_state',max_q_l=32,max_v_l=75,ctx_mode='video_tef',normalize_v=True,normalize_t=True,
            load_labels=True,clip_len=2,max_windows=5,span_loss_type='l1',txt_drop_ratio=0)
    return ns,dataset


def test_real_notebook_input_forward_loss_evaluator_parity(real_assets,tmp_path):
    # Tolerances fixed BEFORE comparison: inputs bit-exact; forward/loss atol1e-6 rtol1e-5; metrics atol1e-8.
    torch.set_num_threads(2)
    assets=real_assets;root=Path(__file__).resolve().parents[2];cfg=runner.read(runner.CONFIG)
    cls=runner.dataset_type(assets/'upstream');fm=runner.read(assets/'assets/feature_cache/feature_manifest.json')
    train_path=root/'local_data/datasets/lumae_ads_d500/lumae_ads_d500_train.jsonl'
    val_path=root/'local_data/datasets/lumae_ads_d500/lumae_ads_d500_val.jsonl'
    train=cls(train_path,assets/'assets/feature_cache',fm,cfg);val=cls(val_path,assets/'assets/feature_cache',fm,cfg)
    selected=[0,1,next(i for i,r in enumerate(train.data) if r['vid']=='lumae_ads_008ff6f5e15fae24e9a72bc79deae917')]
    ns,make_oracle=oracle_namespace(assets,tmp_path,train.data[:2]+[train.data[selected[2]]]+val.data)
    old_train=make_oracle(train_path);old_val=make_oracle(val_path)
    assert [(r['qid'],r['vid']) for r in train.data]==[(r['qid'],r['vid']) for r in old_train.data]
    assert [(r['qid'],r['vid']) for r in val.data]==[(r['qid'],r['vid']) for r in old_val.data]
    for new,old,indices in [(train,old_train,selected),(val,old_val,range(30))]:
        for i in indices:
            a,b=new[i],old[i];assert a['meta']==b['meta']
            for key in a['model_inputs']:torch.testing.assert_close(a['model_inputs'][key],b['model_inputs'][key],atol=0,rtol=0)
    from moment_detr.start_end_dataset import start_end_collate,prepare_batch_inputs
    ba=start_end_collate([train[i] for i in selected]);bb=start_end_collate([old_train[i] for i in selected])
    ia,ta=prepare_batch_inputs(ba[1],torch.device('cpu'));ib,tb=prepare_batch_inputs(bb[1],torch.device('cpu'))
    for key in ia:torch.testing.assert_close(ia[key],ib[key],atol=0,rtol=0)
    ma,ca,_=runner.load_model(assets/'upstream',assets/'assets/M0_pretrained.ckpt')
    mb,cb,_=runner.load_model(assets/'upstream',assets/'assets/M0_pretrained.ckpt')
    ma.eval();mb.eval()
    with torch.no_grad():
        oa=ma(**ia);ob=mb(**ib);la=ca(oa,ta);lb=cb(ob,tb)
    def compare(a,b):
        if isinstance(a,dict):
            assert a.keys()==b.keys()
            for k in a:compare(a[k],b[k])
        elif isinstance(a,list):
            assert len(a)==len(b)
            for x,y in zip(a,b):compare(x,y)
        elif torch.is_tensor(a):torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-5)
        else:assert a==b
    compare(oa,ob);compare(la,lb)
    ns.update(val_ds=old_val,val_qids={r['qid'] for r in old_val.data})
    expected,pred_old=ns['evaluate_val'](mb);actual,pred_new=runner.evaluate(ma,val)
    for key in expected:assert actual[key]==pytest.approx(expected[key],abs=1e-8)
    assert [p['qid'] for p in pred_old]==[p['qid'] for p in pred_new]
    np.testing.assert_allclose([p['pred_relevant_windows'] for p in pred_new],[p['pred_relevant_windows'] for p in pred_old],atol=1e-6,rtol=1e-5)
    # Identical seed and DataLoader generator yield identical order without feature loads.
    def order():return list(torch.utils.data.RandomSampler(range(440),generator=torch.Generator().manual_seed(cfg['seed'])))
    assert order()==order()
    result={'status':'PASS','train_order_ids':440,'val_order_ids':30,'tensor_samples':33,'includes_float16_exception':True,
        'M0_strict_load':True,'forward':'PASS','loss_components':list(la),'full_val30_evaluator':'PASS','metrics':actual,
        'tolerances':{'input_atol':0,'input_rtol':0,'forward_loss_atol':1e-6,'forward_loss_rtol':1e-5,'metric_atol':1e-8},
        'scope':'CPU input/forward/loss/evaluation parity, NOT full GPU training reproduction'}
    with (tmp_path/'cached_runner_parity_results.json').open('w',encoding='utf-8') as f:json.dump(result,f,indent=2)


def test_missing_and_corrupt_cache(real_assets,tmp_path):
    cls=runner.dataset_type(real_assets/'upstream');cfg=runner.read(runner.CONFIG)
    row={'qid':'q','vid':'v','query':'A person raises a bottle','duration':4,'relevant_windows':[[1,3]],'saliency_scores':None,'relevant_clip_ids':None}
    annotations=tmp_path/'train.jsonl';annotations.write_text(json.dumps(row)+'\n');fm={'video':{'v':'video.npz'},'text':{'q':'text.npz'}}
    ds=cls(annotations,tmp_path,fm,cfg)
    with pytest.raises(FileNotFoundError):ds[0]
    np.savez(tmp_path/'video.npz',features=np.ones((90,512),dtype=np.float16));np.savez(tmp_path/'text.npz',last_hidden_state=np.ones((40,512),dtype=np.float32))
    sample=ds[0];assert sample['model_inputs']['video_feat'].shape==(75,514);assert sample['model_inputs']['query_feat'].shape==(32,512)
    np.savez(tmp_path/'video.npz',features=np.full((4,512),np.nan,dtype=np.float32))
    with pytest.raises(ValueError,match='Invalid video'):ds[0]

def test_training_rng_matches_notebook_after_model_construction(real_assets):
    import random
    seed=20260930
    runner.initialize_training_model(real_assets/'upstream',real_assets/'assets/M0_pretrained.ckpt',seed,'cpu')
    actual=(random.random(),float(np.random.rand()),torch.rand(8))
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    expected=(random.random(),float(np.random.rand()),torch.rand(8))
    assert actual[:2]==expected[:2]
    torch.testing.assert_close(actual[2],expected[2],atol=0,rtol=0)
