"""Manifest-resolved D500 E0 cached training. No video/ZIP/CLIP extraction imports.

The pinned upstream model, criterion, span labels and collate remain authoritative.
Data/forward parity is distinct from full GPU checkpoint reproduction.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import os
import platform
from pathlib import Path
import random
import subprocess
import sys
import time

UPSTREAM = 'b7e553ac3b0c898ee6b85e03ee507c064eab89ca'
CONFIG = Path(__file__).with_name('config_d500_cached.json')

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def canonical(value):
    return json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()

def bind_upstream(root):
    root=Path(root).resolve()
    manifest=root/'SOURCE_MANIFEST.json'
    if manifest.is_file():
        pin=read(manifest)
        if pin['commit']!=UPSTREAM:raise ValueError('Wrong upstream revision')
        required={'moment_detr/model.py','moment_detr/start_end_dataset.py','standalone_eval/eval.py'}
        if not required <= {x['path'] for x in pin['files']}:raise ValueError('Incomplete source snapshot')
        for item in pin['files']:
            p=(root/item['path']).resolve()
            if not p.is_relative_to(root) or sha(p)!=item['sha256']:raise ValueError('Pinned source checksum mismatch')
    else:
        commit=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
        if commit!=UPSTREAM:raise ValueError('Use pinned Moment-DETR commit, never HEAD')
        if subprocess.check_output(['git','-C',str(root),'diff','HEAD','--','moment_detr','utils','standalone_eval'],text=True):
            raise ValueError('Upstream source has uncommitted changes')
    for name in ('moment_detr','utils','standalone_eval'):
        module=sys.modules.get(name)
        if module and getattr(module,'__file__',None) and not Path(module.__file__).resolve().is_relative_to(root):
            raise RuntimeError('Another upstream checkout is already imported; use a fresh Python process')
    if str(root) not in sys.path:sys.path.insert(0,str(root))
    return root

def feature_path(root, entry):
    root=Path(root).resolve();p=Path(entry)
    if p.is_absolute() or '..' in p.parts:raise ValueError('Feature manifest requires safe relative paths')
    p=(root/p).resolve()
    if not p.is_relative_to(root):raise ValueError('Feature escapes cache root')
    return p

def dataset_type(upstream_root):
    bind_upstream(upstream_root)
    import numpy as np
    import torch
    from moment_detr.start_end_dataset import StartEndDataset
    from utils.basic_utils import l2_normalize_np_array

    class CachedDataset(StartEndDataset):
        def __init__(self, annotations, feature_root, manifest, config):
            self.feature_root=Path(feature_root);self.manifest=manifest;self.recipe=config
            super().__init__(dset_name='hl',data_path=str(annotations),v_feat_dirs=[str(feature_root)],
                q_feat_dir=str(feature_root),q_feat_type=config['text_key'],max_q_l=config['max_query_length'],
                max_v_l=config['max_video_length'],ctx_mode='video_tef',normalize_v=config['normalize_video'],
                normalize_t=config['normalize_text'],load_labels=True,clip_len=config['clip_length'],
                max_windows=5,span_loss_type='l1',txt_drop_ratio=0)
            qids=[str(r['qid']) for r in self.data]
            if len(qids)!=len(set(qids)):raise ValueError('Duplicate query IDs')
            for row in self.data:
                from data_process.validation.temporal import validate_duration, validate_windows
                if not validate_duration(row['duration'])[0] or not validate_windows(row['relevant_windows'],row['duration'])[0]:
                    raise ValueError('Invalid temporal annotation')
                if not row['query'].strip():raise ValueError('Empty query')
                if row.get('saliency_scores') is not None or row.get('relevant_clip_ids') is not None:
                    raise ValueError('D500 E0 does not use fabricated saliency targets')
                for kind,identity in [('video',row['vid']),('text',str(row['qid']))]:
                    if identity not in manifest[kind]:raise FileNotFoundError(f'Missing {kind} mapping: {identity}')
                    feature_path(self.feature_root,manifest[kind][identity])

        def array(self, kind, identity):
            path=feature_path(self.feature_root,self.manifest[kind][str(identity)])
            key=self.recipe['video_key'] if kind=='video' else self.recipe['text_key']
            with np.load(path,allow_pickle=False) as z:
                if set(z.files)!={key}:raise ValueError(f'Wrong NPZ keys: {path}')
                a=z[key]
                if a.ndim!=2 or a.shape[1]!=512 or len(a)<1 or a.dtype.kind!='f' or not np.isfinite(a).all():
                    raise ValueError(f'Invalid {kind} tensor: {path}')
                limit=self.max_v_l if kind=='video' else self.max_q_l
                a=a[:limit].astype(np.float32)
            if self.normalize_v if kind=='video' else self.normalize_t:a=l2_normalize_np_array(a)
            return torch.from_numpy(a)

        def __getitem__(self,index):
            meta=self.data[index];video=self.array('video',meta['vid']);text=self.array('text',meta['qid']);n=len(video)
            start=torch.arange(n,dtype=torch.float32)/n
            video=torch.cat([video,torch.stack([start,start+1.0/n],dim=1)],dim=1)
            spans=self.get_span_labels([list(w) for w in meta['relevant_windows']],n)
            return {'meta':meta,'model_inputs':{'query_feat':text,'video_feat':video,'span_labels':spans}}
    return CachedDataset

def load_model(upstream_root, checkpoint, device='cpu'):
    bind_upstream(upstream_root)
    import torch
    from moment_detr.model import build_model
    config=read(CONFIG)
    if sha(checkpoint)!=config['m0_sha256']:raise ValueError('M0 SHA mismatch; never initialize E0 from M1-D200')
    with torch.serialization.safe_globals([argparse.Namespace]):
        parent=torch.load(checkpoint,map_location='cpu',weights_only=True)
    opt=copy.deepcopy(parent['opt']);opt.device=torch.device(device);opt.lw_saliency=0.0
    if (opt.v_feat_dim,opt.t_feat_dim,opt.num_queries,opt.hidden_dim,opt.span_loss_type)!=(514,512,10,256,'l1'):
        raise ValueError('M0 architecture mismatch')
    model,criterion=build_model(opt);model.load_state_dict(parent['model'],strict=True)
    model.to(opt.device);criterion.to(opt.device)
    if criterion.weight_dict['loss_saliency']!=0:raise ValueError('Saliency loss must be disabled')
    return model,criterion,opt

def initialize_training_model(upstream_root,checkpoint,seed,device='cuda'):
    # Cell11 resets RNG on an ALREADY constructed Cell10 model. Module
    # initialization consumes randomness even when all weights are then loaded.
    import numpy as np
    import torch
    model,criterion,opt=load_model(upstream_root,checkpoint,device)
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    return model,criterion,opt

def preflight(train_jsonl,val_jsonl,feature_manifest,feature_root,upstream_root,checkpoint,config=None):
    config=config or read(CONFIG)
    if config['upstream_commit']!=UPSTREAM or config['test_used_for_selection'] is not False:raise ValueError('Wrong E0 recipe')
    manifest=read(feature_manifest)
    if manifest['canonical_sha256']!=config['canonical_sha256'] or manifest['feature_contract']['tef_in_cache']:
        raise ValueError('Wrong dataset/TEF contract')
    cls=dataset_type(upstream_root);datasets={}
    for split,path in [('train',Path(train_jsonl)),('val',Path(val_jsonl))]:
        if sha(path)!=config['split_sha256'][split]:raise ValueError(f'{split} frozen checksum mismatch')
        datasets[split]=cls(path,feature_root,manifest,config)
        if len(datasets[split])!=config['counts'][split]:raise ValueError('Wrong split count')
    for key in ['vid','qid']:
        if {str(r[key]) for r in datasets['train'].data}&{str(r[key]) for r in datasets['val'].data}:raise ValueError('Train/Val leakage')
    feature_ids={};exceptions=[]
    for split,dset in datasets.items():
        for row in dset.data:
            for kind,identity in [('video',row['vid']),('text',str(row['qid']))]:
                path=feature_path(feature_root,manifest[kind][identity]);dset.array(kind,identity)
                feature_ids[manifest[kind][identity]]=sha(path)
                import numpy as np
                with np.load(path,allow_pickle=False) as z:
                    if z[z.files[0]].dtype!=np.float32:exceptions.append({'split':split,'kind':kind,'id':identity,'dtype':str(z[z.files[0]].dtype)})
    if sha(checkpoint)!=config['m0_sha256']:raise ValueError('Missing/changed M0')
    bind_upstream(upstream_root)
    import torch
    import numpy as np
    identity={'recipe':config,'feature_sha256':feature_ids,'feature_manifest_sha256':sha(feature_manifest),'m0_sha256':sha(checkpoint),
        'runner_sha256':sha(__file__),'upstream_commit':UPSTREAM,'evaluator_sha256':sha(Path(upstream_root)/'standalone_eval/eval.py'),
        'runtime':{'python':platform.python_version(),'torch':str(torch.__version__),'numpy':np.__version__,'cuda_build':torch.version.cuda}}
    return datasets,identity,exceptions

def temporal_iou(a,b):
    inter=max(0.,min(a[1],b[1])-max(a[0],b[0]));union=max(a[1],b[1])-min(a[0],b[0])
    return inter/union if union>0 else 0.

def evaluate(model,dataset,device='cpu',batch_size=4):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from moment_detr.start_end_dataset import start_end_collate,prepare_batch_inputs
    from moment_detr.span_utils import span_cxw_to_xx
    from standalone_eval.eval import compute_mr_r1,compute_mr_ap
    predictions=[];model.eval()
    with torch.no_grad():
        for meta,batched in DataLoader(dataset,batch_size=batch_size,shuffle=False,num_workers=0,collate_fn=start_end_collate):
            inputs,_=prepare_batch_inputs(batched,torch.device(device));out=model(**inputs)
            spans=span_cxw_to_xx(out['pred_spans']).cpu().numpy();scores=out['pred_logits'].softmax(-1)[...,0].cpu().numpy()
            for i,row in enumerate(meta):
                windows=[]
                for j in np.argsort(-scores[i]):
                    start,end=spans[i,j]*float(row['duration']);start=float(np.clip(start,0,row['duration']));end=float(np.clip(end,0,row['duration']))
                    windows.append([min(start,end),max(start,end),float(scores[i,j])])
                predictions.append({k:row[k] for k in ('qid','query','vid')}|{'pred_relevant_windows':windows})
    refs=dataset.data;r1=compute_mr_r1(predictions,refs,iou_thds=[.5,.7]);ap=compute_mr_ap(predictions,refs,num_workers=1,iou_thds=np.linspace(.5,.95,10))
    gt={r['qid']:r for r in refs};ious=[max(temporal_iou(p['pred_relevant_windows'][0][:2],w) for w in gt[p['qid']]['relevant_windows']) for p in predictions]
    return {'R1@0.5':r1['0.5'],'R1@0.7':r1['0.7'],'MR_mAP':ap['average'],'mAP@0.5':ap['0.5'],'mAP@0.75':ap['0.75'],'mIoU':round(float(np.mean(ious))*100,2)},predictions

def improves(metrics,best):
    return best is None or (float(metrics['R1@0.5']),float(metrics['mIoU']))>tuple(best)

def train(datasets,identity,upstream_root,checkpoint,output,resume=False):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from moment_detr.start_end_dataset import start_end_collate,prepare_batch_inputs
    cfg=identity['recipe'];output=Path(output).resolve()
    if not torch.cuda.is_available():raise RuntimeError('E0 training requires CUDA AMP; use CPU parity tests for validation')
    fingerprint=hashlib.sha256(canonical(identity)).hexdigest()
    if resume:
        if read(output/'run_config.json')['fingerprint']!=fingerprint:raise ValueError('Resume inputs/recipe changed')
        if not (output/'best.ckpt').is_file():raise FileNotFoundError('Resume requires preserved best.ckpt')
    else:
        output.mkdir(parents=True,exist_ok=False)
        (output/'run_config.json').write_text(json.dumps({'fingerprint':fingerprint,'identity':identity,'source_revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'torch':torch.__version__,'numpy':np.__version__},indent=2))
    seed=cfg['seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    model,criterion,opt=initialize_training_model(upstream_root,checkpoint,seed,'cuda')
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=cfg['lr_step_size'],gamma=cfg['lr_gamma'])
    scaler=torch.amp.GradScaler('cuda',enabled=cfg['amp']);generator=torch.Generator().manual_seed(seed)
    history=[];best=None;best_epoch=None;bad=0;start_epoch=0
    if resume:
        with torch.serialization.safe_globals([argparse.Namespace]):saved=torch.load(output/'latest.ckpt',map_location='cpu',weights_only=True)
        if saved['fingerprint']!=fingerprint:raise ValueError('Resume checkpoint identity mismatch')
        model.load_state_dict(saved['model'],strict=True);optimizer.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler']);scaler.load_state_dict(saved['scaler'])
        history=saved['history'];best=saved['best'];best_epoch=saved['best_epoch'];bad=saved['bad_epochs'];start_epoch=saved['epoch']+1
        generator.set_state(saved['generator']);torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state_all(saved['cuda_rng']);random.setstate(saved['python_rng'])
        n=saved['numpy_rng'];np.random.set_state((n[0],np.array(n[1],dtype=np.uint32),n[2],n[3],n[4]))
        if bad>=cfg['patience'] or start_epoch>=cfg['epochs']:return history
    loader=DataLoader(datasets['train'],batch_size=cfg['batch_size'],shuffle=True,generator=generator,num_workers=0,pin_memory=True,collate_fn=start_end_collate)
    def persist(name,value):
        tmp=output/(name+'.tmp');torch.save(value,tmp);os.replace(tmp,output/name)
    for epoch in range(start_epoch,cfg['epochs']):
        model.train();criterion.train();losses=[];started=time.perf_counter()
        for _,batched in loader:
            inputs,targets=prepare_batch_inputs(batched,torch.device('cuda'),non_blocking=True);optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda',dtype=torch.float16,enabled=cfg['amp']):
                loss_map=criterion(model(**inputs),targets);loss=sum(v*criterion.weight_dict[k] for k,v in loss_map.items() if k in criterion.weight_dict)
            if not torch.isfinite(loss).item():raise RuntimeError('Nonfinite loss; previous checkpoint preserved')
            scaler.scale(loss).backward();scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip']);scaler.step(optimizer);scaler.update();losses.append(float(loss.detach().cpu()))
        scheduler.step();metrics,predictions=evaluate(model,datasets['val'],'cuda',cfg['batch_size']);improved=improves(metrics,best)
        if improved:best=(float(metrics['R1@0.5']),float(metrics['mIoU']));best_epoch=epoch+1;bad=0
        else:bad+=1
        history.append({'epoch':epoch+1,'train_loss':round(sum(losses)/len(losses),6),**metrics,'learning_rate':optimizer.param_groups[0]['lr'],'elapsed_seconds':round(time.perf_counter()-started,2),'is_best':improved})
        ns=np.random.get_state();saved={'model':model.state_dict(),'opt':opt,'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'scaler':scaler.state_dict(),'epoch':epoch,'best':best,'best_epoch':best_epoch,'bad_epochs':bad,'history':history,'fingerprint':fingerprint,'generator':generator.get_state(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'python_rng':random.getstate(),'numpy_rng':(ns[0],ns[1].tolist(),ns[2],ns[3],ns[4])}
        if improved:
            persist('best.ckpt',saved);(output/'best_val_predictions.json').write_text(json.dumps(predictions));(output/'best_val_metrics.json').write_text(json.dumps(metrics))
        persist('latest.ckpt',saved);(output/'training_history.json').write_text(json.dumps(history,indent=2))
        if bad>=cfg['patience']:break
    return history

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['preflight','train'])
    for arg in ['train-jsonl','val-jsonl','feature-manifest','feature-root','upstream-root','checkpoint','output']:p.add_argument('--'+arg,type=Path,required=True)
    p.add_argument('--resume',action='store_true');a=p.parse_args()
    if any(a.output.resolve().is_relative_to(x.resolve()) for x in [a.train_jsonl.parent,a.val_jsonl.parent,a.feature_root]):p.error('Output must be outside frozen inputs/cache')
    datasets,identity,exceptions=preflight(a.train_jsonl,a.val_jsonl,a.feature_manifest,a.feature_root,a.upstream_root,a.checkpoint)
    if a.action=='preflight':
        with a.output.open('x',encoding='utf-8') as f:json.dump({'status':'PASS','identity':identity,'dtype_exceptions':exceptions},f,indent=2)
    else:train(datasets,identity,a.upstream_root,a.checkpoint,a.output,a.resume)

if __name__=='__main__':main()
