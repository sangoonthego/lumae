"""Uniform full-source query-only crosscheck; explicitly POST HOC, not blind-first."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from p2_controlled import pilot
from p16_recovery import O,R
from data_process.lumae_scale.models import ROOT,read_json,digest
from data_process.lumae_scale.incremental import immutable_write,json_bytes
from data_process.lumae_scale.clip_runtime import ClipRuntime
import numpy as np
import torch

torch.set_num_threads(4)
p=pilot();weights=ROOT/'local_data/cache/clip_weights/ViT-B-32.pt'
runtime=ClipRuntime(str(weights),device='cpu',download_root=p.work/'model_cache')
rows=[]
for proposal in sorted((p.work/'evidence').rglob('proposal.json')):
    sid=proposal.relative_to(p.work/'evidence').parts[0]
    cache=R/'cache'/('p1_'+p.c['run_id'])/'visual_index'/sid
    pred=read_json(proposal);duration=read_json(cache/'metadata.json')['duration']
    with np.load(cache/'clip_features.npz') as z:
        times=z['timestamps'];mask=np.isclose(np.mod(times,2),0,atol=1e-6)
        uniform=times[mask]
        assert np.allclose(uniform,np.arange(0,duration-.05,2))
        scores=z['features'][mask]@runtime.encode_text(pred['query'])[0]
        peak=float(uniform[int(np.argmax(scores))])
    rows.append({'source_id':sid,'query':pred['query'],'proposal_path':str(proposal),'proposal_sha256':digest(proposal),
        'uniform_query_only_peak':peak,'uniform_query_only_interval':[max(0,peak-2),min(duration,peak+2)],
        'generator_interval':[pred['start'],pred['end']],'uniform_step_seconds':2,'full_source_coverage':True,
        'generator_phase_as_retrieval_input':False,'chronology':'POST_HOC_CROSSCHECK_AFTER_GENERATOR_AND_VISUAL_ADJUDICATION',
        'same_clip_model':True,'human_verified':False})
result={'procedure':'Query-only retrieval on fixed0,2,4,... seconds, removing candidate-phase sampling density from retrieval',
    'limitations':'Earlier numerical blind-first retrieval did not receive bounds as scalar inputs, but used denser generator-selected phase sampling. Therefore it was only partially blinded. This uniform crosscheck is post hoc, not retroactive blind-first evidence, and same CLIP weights are not model independence or human IAA.',
    'weights_sha256':digest(weights),'human_verified':0,'human_IAA':None,'model_revision':'Local CLIP ViT-B/32 pinned weights; Codex visual interface revision NOT_EXPOSED','rows':rows}
immutable_write(p.work/'uniform_grid_posthoc_review.json',json_bytes(result))
immutable_write(O/'review_procedure_limitations.json',json_bytes({'pilot_recovery_freeze_sha256':digest(O/'pilot30_recovery_freeze_manifest.json'),
    'p16_prior16_audit':'Unblinded re-audit explicitly recorded; no new independent-model or human claim',
    'p16_failed13_and_p2_numerical_blind_first':'Scalar phase/bounds not supplied to retrieval; sampling density still conditioned on generator phase, so partial blinding only',
    'p2_uniform_crosscheck_path':str(p.work/'uniform_grid_posthoc_review.json'),'sha256':digest(p.work/'uniform_grid_posthoc_review.json'),
    'posthoc_not_blind_first':True,'distinct_independent_model':False,'human_verified':0,'human_IAA':None}))
print(len(rows))
