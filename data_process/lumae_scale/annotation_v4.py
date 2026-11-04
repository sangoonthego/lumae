"""Versioned CLIP boundary experiment. The historical V3 wrapper is untouched.

V4 removes the unconditional four-second floor and one-second context padding.
It selects a query-similar contiguous component inside the inspected coarse
phase, stopping at strong visual discontinuities. Actual visual review remains
mandatory: CLIP similarity and this proposal are not correctness probabilities.
"""
from __future__ import annotations
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import math
import time
from pathlib import Path
import numpy as np

from .quality_gate import query_errors,interval_errors
from .incremental import immutable_write,json_bytes
from .models import read_json,digest
from .cache_validation import atomic_npz
from ..annotation.assisted_deployment import ALGORITHM_SHA256,verify_frozen_ranker

VERSION='annotation_v4_1_clip_component_no_fixed_context'
SCHEMA='clip512_float32_l2_v1'
PREPROCESS='openai_clip_vit_b32_official_rgb224_center_crop_v1'
DEFAULTS={'pipeline_version':VERSION,'max_events':2,'sparse_step':2.0,
    'dense_step':.5,'short_step':.25,'short_phase_max_seconds':4.0,
    'context_seconds':2.0,'max_dense_frames_per_source':192,
    'similarity_drop':.03,'cut_cosine':.65,'max_component_gap_steps':1.51,
    'max_image_batch':8,'min_available_memory_bytes':1073741824,
    'min_free_bytes':16106127360,'max_processing_bytes':2147483648,
    'human_verified':False,'automatic_acceptance_enabled':False,
    'preprocess':PREPROCESS,'feature_schema':SCHEMA}


def key_for(kind,*,video_sha=None,encoder_sha,config,timestamps=None,query=None):
    payload={'kind':kind,'video_sha':video_sha,'encoder_sha':encoder_sha,
        'preprocess':config.get('preprocess',PREPROCESS),'schema':config.get('feature_schema',SCHEMA),
        'timestamps':None if timestamps is None else [round(float(t),6) for t in timestamps],
        'query':query}
    return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def adaptive_times(duration,events,config=DEFAULTS):
    if not math.isfinite(duration) or not 0<duration<=150:raise ValueError('Invalid source duration')
    sparse=np.arange(0,max(0,duration-.05),config['sparse_step']).tolist();dense=[]
    for e in events[:config['max_events']]:
        s,end=float(e['coarse_start']),float(e['coarse_end'])
        if interval_errors(s,end,duration):raise ValueError('Invalid observed coarse phase')
        step=config['short_step'] if end-s<=config['short_phase_max_seconds'] else config['dense_step']
        dense.extend(np.arange(max(0,s-config['context_seconds']),min(duration-.05,end+config['context_seconds']),step).tolist())
    dense=sorted(set(round(t,6) for t in dense))
    if len(dense)>config['max_dense_frames_per_source']:raise ValueError('Dense extraction budget exceeded')
    return sorted(set(round(t,6) for t in sparse+dense))


def rank_events(events,config=DEFAULTS):
    valid=[];seen=set()
    for e in events[:config['max_events']]:
        if query_errors(e['query']):continue
        if not e.get('actual_visual_support'):continue
        if not 0<=e['coarse_start']<e['coarse_end']:continue
        identity=(e.get('visible_action','').strip().lower(),e.get('visible_object','').strip().lower())
        if identity in seen:continue
        seen.add(identity)
        axes=e.get('event_quality',{})
        if set(axes)!={'support','specificity','uniqueness','boundary_visibility','clarity'}:raise ValueError('Explicit visual-quality axes required')
        if any(type(v) not in (int,float) or not math.isfinite(v) or not 0<=v<=5 for v in axes.values()):raise ValueError('Invalid event quality score')
        score=3*axes['support']+3*axes['specificity']+2*axes['uniqueness']+3*axes['boundary_visibility']+2*axes['clarity']
        valid.append({**e,'selection_score':score,'selection_score_is_calibrated_probability':False})
    return sorted(valid,key=lambda e:(-e['selection_score'],e['query']))


def component_bounds(times,features,scores,phase,config=DEFAULTS):
    times=np.asarray(times,dtype=float);features=np.asarray(features);scores=np.asarray(scores)
    if times.ndim!=1 or features.shape!=(len(times),512) or scores.shape!=(len(times),):raise ValueError('Invalid feature schema')
    if not np.isfinite(features).all() or not np.isfinite(times).all() or not np.isfinite(scores).all() or np.any(np.diff(times)<=0):raise ValueError('Invalid feature values/order')
    allowed=(times>=phase[0])&(times<phase[1])
    if not allowed.any():raise ValueError('No frame in phase; abstain')
    best=int(np.argmax(np.where(allowed,scores,-np.inf)));left=right=best
    step=config['short_step'] if phase[1]-phase[0]<=config['short_phase_max_seconds'] else config['dense_step']
    threshold=float(scores[best]-config['similarity_drop'])
    cuts=[]
    def crosses(a,b):
        discontinuity=float(features[a]@features[b])<config['cut_cosine']
        gap=times[b]-times[a]>step*config['max_component_gap_steps']
        if discontinuity:cuts.append([float(times[a]),float(times[b])])
        return discontinuity or gap
    while left>0 and allowed[left-1] and scores[left-1]>=threshold and not crosses(left-1,left):left-=1
    while right+1<len(times) and allowed[right+1] and scores[right+1]>=threshold and not crosses(right,right+1):right+=1
    # Half of actual sampling step brackets sampled states, rather than forcing
    # a desired event duration or importing the adjudicator's reviewed interval.
    start=max(float(phase[0]),float(times[left]-step/2))
    end=min(float(phase[1]),float(times[right]+step/2))
    if start>=end:raise ValueError('Empty component; abstain')
    return {'start':round(start,6),'end':round(end,6),'peak_timestamp':float(times[best]),
        'component_sample_times':times[left:right+1].tolist(),'cut_pairs':cuts,'sampling_step':step,
        'boundary_uncertainty_seconds':step/2,'similarity_threshold':threshold,
        'coarse_phase_used':list(map(float,phase))}


def localize_v4(video_id,query,events,duration,*,visual_cache,runtime,config=DEFAULTS):
    from .temporal_labeler import localize
    verify_frozen_ranker()
    # Run unchanged SemanticV3/old wrapper and retain its raw output separately.
    baseline=localize(video_id,query,events,duration,0,clip_runtime=runtime,visual_cache=visual_cache)
    with np.load(Path(visual_cache)/video_id/'clip_features.npz') as z:
        vector=runtime.encode_text(query)[0];scores=z['features']@vector
        details=component_bounds(z['timestamps'],z['features'],scores,
            (events[0]['coarse_start'],events[0]['coarse_end']),config)
    if interval_errors(details['start'],details['end'],duration):raise ValueError('Invalid V4 interval')
    prediction={**baseline,'start':details['start'],'end':details['end'],
        'boundary_wrapper_version':VERSION,'boundary_method':'query_similarity_contiguous_component_cut_limited',
        'raw_frozen_wrapper_interval':[baseline['start'],baseline['end']],
        'v4_boundary_evidence':details,'reviewed_interval_used_as_model_input':False,
        'semantic_v3_algorithm_sha256':ALGORITHM_SHA256,'automatic_acceptance':False}
    return baseline,prediction


class CachedRuntime:
    """Exact-input text cache around the existing frozen CLIP runtime."""
    def __init__(self,runtime,directory,encoder_sha,config=DEFAULTS):
        self.runtime=runtime;self.directory=Path(directory);self.encoder_sha=encoder_sha;self.config=config
        self.text_calls=0;self.text_hits=0;self.image_calls=0;self.image_frames=0
    def encode_text(self,query):
        key=key_for('text',encoder_sha=self.encoder_sha,config=self.config,query=query)
        path=self.directory/(key+'.npz')
        if path.exists():
            self.text_hits+=1
            with np.load(path) as z:return z['features'].copy()
        self.text_calls+=1;value=self.runtime.encode_text(query)
        if value.shape!=(1,512) or not np.isfinite(value).all():raise ValueError('Invalid encoded text')
        self.directory.mkdir(parents=True,exist_ok=True);atomic_npz(path,features=value)
        return value
    def encode_images(self,paths,**kwargs):
        self.image_calls+=1;self.image_frames+=len(paths)
        return self.runtime.encode_images(paths,**kwargs)


def uniform_query_only(video_id,query,*,cache,runtime,duration):
    with np.load(Path(cache)/video_id/'clip_features.npz') as z:
        times=z['timestamps'];mask=np.isclose(np.mod(times,2),0,atol=1e-6)
        if not np.allclose(times[mask],np.arange(0,duration-.05,2)):raise ValueError('Reviewer needs complete fixed uniform source coverage')
        peak=float(times[mask][int(np.argmax(z['features'][mask]@runtime.encode_text(query)[0]))])
    return {'query':query,'interval':[max(0,peak-2),min(duration,peak+2)],
        'protocol':'Uniform0,2,4,... full source; only query+uniform features; computed before generator localization',
        'generator_bounds_received':False,'generator_dense_samples_received':False,
        'distinct_model':False,'same_clip_weights':True,'human_verified':False}


def review_errors(review,prediction,duration):
    """Sample-supported AI onset/offset brackets, not independent human GT."""
    errors=[]
    if review.get('human_verified') is not False:errors.append('New review must disclose AI provenance')
    for field in ('actual_overview_inspected','actual_qc_inspected','semantic_match','unique_event','complete_occurrence'):
        if review.get(field) is not True:errors.append('Missing or failed '+field)
    if not review.get('reason'):errors.append('Missing actual adjudication reason')
    for name,value in [('onset_bracket_seconds',prediction['start']),('offset_bracket_seconds',prediction['end'])]:
        bracket=review.get(name,[])
        if len(bracket)!=2 or not all(isinstance(x,(int,float)) and math.isfinite(x) for x in bracket) or not 0<=bracket[0]<=bracket[1]<=duration:
            errors.append('Invalid '+name);continue
        if not bracket[0]-1e-6<=value<=bracket[1]+1e-6:errors.append('Proposal outside observed '+name)
    errors.extend(interval_errors(prediction['start'],prediction['end'],duration))
    return errors
