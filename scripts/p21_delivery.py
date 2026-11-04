"""Reconcile executed P21 evidence, quantitative reports and a new sealed manifest."""
import json,sys,math,statistics,shutil
from collections import Counter,defaultdict
from pathlib import Path
from datetime import datetime,timezone
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from p21_audit import O,E,R,ROOT,csvout
from data_process.lumae_scale.models import read_json,digest,atomic_json
from data_process.lumae_scale.p21_quality_registry import load_exclusions,effective_records

def jl(path,rows):
 # JSONL requires one complete JSON value per physical line.
 path.write_text(''.join(json.dumps(r,sort_keys=True,ensure_ascii=False,allow_nan=False)+'\n' for r in rows),encoding='utf-8')
 assert [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]==rows
def md(path,text):path.write_text(text,encoding='utf-8')
def wilson(k,n):
 if not n:return [None,None]
 z=1.959963984540054;p=k/n;den=1+z*z/n;c=(p+z*z/(2*n))/den;d=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
 return [max(0,c-d),min(1,c+d)]
def accepted(stage):return [read_json(p) for p in sorted((R/'work'/('p21_'+stage)/'accepted').glob('*.json'))]
def main():
 exclusions=load_exclusions(O/'validation/final_quality_exclusions.json')
 exposure=read_json(O/'validation/fresh_source_exposure_audit.json');overlap_ids={r['source_video_id'] for r in exposure['wave_preexisting_source_overlap']}
 stages=['dev01','reserved01','fresh01'];observations={};decisions={};reviews={};proposals=[];profiles={};ablation=[]
 for stage in stages:
  run=E/'runs'/stage;profiles[stage]=read_json(run/'process_profile.json')
  for folder in (run/'sources').iterdir():
   sid=folder.name;obs=read_json(folder/'observations.json');d=read_json(folder/'acceptance_decision.json');v=read_json(folder/'visual_review.json')
   observations[sid]=(stage,obs)
   if sid in exclusions:
    original_status=d['status'];d={**d,'initial_trial_status':original_status,'status':'WITHDRAWN_FINAL_BOUNDARY_AUDIT','selected_variant':None,'final_quality_audit':exclusions[sid],
     'variants':[{**x,'initial_status':x['status'],'status':'REJECTED','errors':x.get('errors',[])+['FINAL_QC: confirmed earlier action omitted'],'final_reason':exclusions[sid]['reason']} for x in d.get('variants',[])]}
    v={**v,'final_quality_audit':exclusions[sid]}
   decisions[sid]=d;reviews[sid]=v
   if (folder/'proposals.json').exists():
    props=read_json(folder/'proposals.json');proposals += [dict(p,stage=stage) for p in props]
   else:props=[]
   byvariant={p['variant']:p for p in props};verdict={p['variant']:p for p in d.get('variants',[])}
   for variant in ('B0','B1','B2','B3','B4','B5'):
    p=byvariant.get(variant,{});judge=verdict.get(variant,{})
    ablation.append({'stage':stage,'source_id':sid,'variant':variant,'executed':bool(p),'query':p.get('query'),
      'start':p.get('prediction',{}).get('start'),'end':p.get('prediction',{}).get('end'),'localization_seconds':p.get('seconds'),
      'ai_bracket_pass':judge.get('status')=='AI_VISUAL_BRACKETS_PASS','initial_ai_bracket_pass':judge.get('initial_status',judge.get('status'))=='AI_VISUAL_BRACKETS_PASS','final_qc_withdrawn':sid in exclusions,'errors':'|'.join(judge.get('errors',[])),
      'assessment':'AI_UNBLINDED_NOT_GT','fresh_preexisting_shared_raw_overlap':sid in overlap_ids,'human_verified':False})
 recovery_ids={r['source_id'] for r in read_json(O/'audit/recovery_inventory.json')};assert len(recovery_ids)==61
 provisional=accepted('dev01')+accepted('reserved01')+accepted('fresh01');new=effective_records(provisional,exclusions)
 recovered=[r for r in new if r['p21_stage']!='fresh01'];fresh=[r for r in new if r['p21_stage']=='fresh01']
 previous=[json.loads(l) for l in (ROOT/'research_outputs/p16_p2_20261103_01/accepted_new_pairs.jsonl').read_text().splitlines()]
 all_new=previous+new
 for key in ('source_video_id','source_video_sha256','qid'):assert len({r[key] for r in all_new})==len(all_new)
 status=Counter(decisions[s]['status'] for s in recovery_ids)
 jl(O/'recovery/recovered_query_candidates.jsonl',[dict(e,source_id=sid,stage=stage) for sid,(stage,o) in observations.items() if sid in recovery_ids for e in o['events']])
 jl(O/'recovery/recovered_temporal_proposals.jsonl',[p for p in proposals if p['source_id'] in recovery_ids])
 jl(O/'recovery/recovery_review_results.jsonl',[reviews[s] for s in sorted(recovery_ids)])
 jl(O/'recovery/recovery_acceptance_decisions.jsonl',[decisions[s] for s in sorted(recovery_ids)])
 jl(O/'recovery/recovered_accepted_pairs.jsonl',recovered)
 jl(O/'manifests/accepted_additions_pool.jsonl',all_new)
 jl(O/'manifests/provisional_trial_acceptances.jsonl',provisional)
 csvout(O/'manifests/accepted_pair_index.csv',[{k:r.get(k) for k in ('qid','source_video_id','source_video_sha256','query','window','selected_ablation_variant','p21_stage','human_verified','split_eligibility')} for r in all_new])
 metrics={'attempted_sources':61,'technical_valid_sources':61,'new_recovered_accepted':len(recovered),'initial_recovery_trial_acceptances':5,'withdrawn_after_final_boundary_audit':len(exclusions),'statuses':dict(status),'recovery_yield':len(recovered)/61,'wilson95':wilson(len(recovered),61),
   'original18_accepted_unchanged':True,'raw_downloads_for_recovery':0,'human_verified':0,'IAA':None,'selected_variants':dict(Counter(r['selected_ablation_variant'] for r in recovered)),
   'reserved_not_genuinely_unseen':True,'selection_biased_recovery_cohort':True,'no_final_d1000_freeze':True}
 atomic_json(O/'recovery/recovery_metrics.json',metrics)
 # Distinguish new actual inspection findings from prior ledger hypotheses.
 taxonomy=[];distribution=Counter()
 for row in read_json(O/'audit/recovery_inventory.json'):
  sid=row['source_id'];d=decisions[sid];obs=observations[sid][1];v=reviews[sid];causes=set()
  if not obs['events']:causes.add('F8')
  if row['duplicate_candidates']:causes.add('F7')
  for review in v.get('event_reviews',[]):
   reason=review['reason'].lower()
   if not review['semantic_match']:causes.add('F1')
   if not review['unique_event']:
    causes.add('F6' if any(t in reason for t in ('repeat','multiple','two motions','several')) else 'F2')
   relevant=[x for x in d.get('variants',[]) if x.get('query')==review['query']]
   if review['semantic_match'] and review['unique_event'] and any(x['status']=='REJECTED' for x in relevant):
    causes.add('F4')
    a,b=review['onset_bracket_seconds'],review['offset_bracket_seconds']
    if b[0]-a[1]<=2:causes.add('F5')
    if any(x.get('window') and (x['window'][1]<a[1] or x['window'][0]>b[0]) for x in relevant):causes.add('F3')
  for c in causes:distribution[c]+=1
  taxonomy.append({'source_id':sid,'historical_causes':'|'.join(row['initial_failure_causes']),'new_observed_causes':'|'.join(sorted(causes)),
    'final_status':d['status'],'selected_variant':d.get('selected_variant'),'actual_reason':' || '.join(x['reason'] for x in v.get('event_reviews',[])) or v['reason'],
    'evidence':'ACTUAL_AI_OVERVIEW_AND_DENSE_REVIEW_NOT_HUMAN_GT','final_withdrawal_reason':exclusions.get(sid,{}).get('reason')})
 csvout(O/'recovery/annotation_failure_taxonomy.csv',taxonomy)
 atomic_json(O/'recovery/failure_distribution.json',{'multi_label_new_observation_counts':dict(sorted(distribution.items())),'source_denominator':61,'categories_can_overlap':True,'not_human_gt':True})
 md(O/'recovery/annotation_root_cause_analysis.md',f'''# Root cause analysis\n\nAll61 sources received a bounded new attempt; recovered{len(recovered)}. Distribution: {dict(distribution)}. Categories overlap and are actual AI judgments, not human GT. Prior diagnoses remain in a separate CSV column.\n\nF1 is evidenced by wrong objects/actions in generated queries: robot/props called a child with pens; cigarette called a cocktail; a filled glass placement called pouring; chopsticks called a fork. Coarse overviews need dense semantic verification. F2 concerns incomplete/inferred actions and event discovery. F3 is an actual nonoverlap/incorrect-shot retrieval finding. F4 is onset/offset disagreement under the same protocol. F5 marks short actions requiring dense resolution; it does not mean every short action is recoverable. F6 preserves repeated-event ambiguity. F7 remains campaign/duplicate uncertainty, never resolved from hash inequality. F8 is explicit abstention on static/repetitive sources.\n\nThe historical wrapper adds1s context and a4s floor; evidence-start max can retain a late onset. Removing padding in v4 reduces overshoot but its peak-minus0.03 component can collapse into a fraction of an action. Actual counterexamples and raw predictions remain in the ablation table. Adaptive sampling adds evidence and compute, but cannot repair incorrect semantics or guarantee full-action boundaries. CLIP/V3 confidence is not a calibrated correctness probability. No threshold was tuned on reserved/fresh results.\n''')
 csvout(O/'annotation_v4/ablation_results.csv',ablation)
 aggregate=[]
 for stage in stages:
  for variant in ('B0','B1','B2','B3','B4','B5'):
   rows=[r for r in ablation if r['stage']==stage and r['variant']==variant];k=sum(r['ai_bracket_pass'] for r in rows)
   aggregate.append({'stage':stage,'variant':variant,'sources':len(rows),'executed_localizations':sum(r['executed'] for r in rows),'ai_bracket_pass':k,'rate':k/len(rows),'wilson95':wilson(k,len(rows)),
      'localization_seconds_sum':sum(r['localization_seconds'] or 0 for r in rows),'human_gt_validated':False})
 csvout(O/'benchmarks/baseline_vs_optimized.csv',aggregate)
 for variant in ('B0','B1','B2','B3','B4','B5'):
  rows=[r for r in ablation if r['stage']=='fresh01' and r['variant']==variant and not r['fresh_preexisting_shared_raw_overlap']];passes=sum(r['ai_bracket_pass'] for r in rows)
  aggregate.append({'stage':'fresh01_genuinely_new_only','variant':variant,'sources':len(rows),'executed_localizations':sum(r['executed'] for r in rows),'ai_bracket_pass':passes,'rate':passes/len(rows),'wilson95':wilson(passes,len(rows)),
   'localization_seconds_sum':sum(r['localization_seconds'] or 0 for r in rows),'human_gt_validated':False})
 csvout(O/'benchmarks/baseline_vs_optimized.csv',aggregate)
 transitions=[]
 for sid,(stage,obs) in observations.items():
  flags={r['variant']:r['ai_bracket_pass'] for r in ablation if r['source_id']==sid}
  transitions.append({'source_id':sid,'stage':stage,'B0_pass':flags['B0'],'B3_pass':flags['B3'],'B4_pass':flags['B4'],
    'B0_to_B4':'BOTH_PASS' if flags['B0'] and flags['B4'] else 'REJECTED_TO_PASS' if flags['B4'] else 'PASS_TO_REJECTED' if flags['B0'] else 'BOTH_REJECTED',
    'selected_pipeline_status':decisions[sid]['status'],'evidence':'AI_BRACKET_AGREEMENT_NOT_HUMAN_GT'})
 csvout(O/'benchmarks/paired_source_transitions.csv',transitions)
 quality='''# Actual paired quality comparison\n\n'''+ '\n'.join(f"- {r['stage']} {r['variant']}: {r['ai_bracket_pass']}/{r['sources']} source-level AI bracket agreement; {r['executed_localizations']} actual localizations." for r in aggregate)+'''
\nThe denominator includes explicit source abstentions. B0/B1 are identical controls because the old provider was already event aware. B2 differs only where a genuinely different event candidate existed. B3 versus B4 shares query and adaptive frames, isolating the wrapper. B5 repeats B4 exactly with cached text and no image re-encoding. Neither B5 timing nor shared preprocessing can establish full pipeline speedup. Human verification=0; IAA=null. No Test GT or publication-grade temporal-IoU result is created. Recovery sources were selected for previous failure and exposed in prior tasks, so their historical zero acceptance is not an unbiased baseline. The whole fresh wave mixes24 genuinely new sources with1 preexisting shared-raw source. Only the filtered genuinely-new subgroup is source-held-out evidence; campaign independence remains unknown. Source-wise regressions are retained, not hidden by aggregate gains. No production promotion is justified.\n'''
 md(O/'annotation_v4/quality_comparison.md',quality);md(O/'annotation_v4/annotation_quality_comparison.md',quality)
 md(O/'benchmarks/quality_regression_audit.md',quality)
 quality += f"\nFinal QA correction:6 provisional additions became5 effective additions after original-resolution onset review withdrew the SPAR record. All6 provisional records and initial review outcomes remain immutable; final_quality_exclusions.json is authoritative for exports/counts. Never export unfiltered provisional builder ledgers. Whole fresh wave also contains1 preexisting shared-raw source; exclude it from genuinely new-source subgroup. No method thresholds or source queries were revised.\n"
 for p in ('annotation_v4/quality_comparison.md','annotation_v4/annotation_quality_comparison.md','benchmarks/quality_regression_audit.md'):md(O/p,quality)
 runtime={'stages':profiles,'interactive_development_seconds':None,'development_timing_reason':'Not separately measured; no retrospective reconstruction','fresh_clock':read_json(O/'benchmarks/fresh_pipeline_clock_complete.json'),
  'model_interface_revision':'NOT_EXPOSED','ai_model_calls_billable':None,'monetary_cost':None,'clip_model_sha256':read_json(O/'audit/audit_before_change.json')['clip_weight_sha256'],'offline_machine_contention':'Regression tests and read-only authoring overlapped some offline runs; do not claim isolated production speedup'}
 runtime['timing_interpretation']={'event_discovery':'Includes initial semantic query formulation; cognitive operations cannot be fully separated by interface wall-clock timers',
   'query_generation':'The4.1-second recorded phase measures final query transcription/serialization only, not all query formulation',
   'fresh_selection':'Precomputed during setup; time unmeasured, not asserted zero',
   'fresh_reconciliation':'One original query failed sanitation before localization. Explicit fail-closed manual reconciliation is included in measured save_validation phase; postrun adapter reproduces the same records without changing frozen inference or acceptance.'}
 runtime['timing_interpretation']['postrun_QA']='Original-resolution final QA and research-integrity audits ran after the1464.05s primary workflow clock. The primary-loop accepted/hour rate is not a validated production rate; full production cost including final audit is unknown.'
 atomic_json(O/'benchmarks/timing_interpretation.json',runtime['timing_interpretation'])
 warm=read_json(E/'runs/devwarm01/process_profile.json');runtime['actual_warm_cache_replay']=warm
 equality=[]
 for folder in (E/'runs/dev01/sources').iterdir():
  if not (folder/'proposals.json').exists():continue
  a=read_json(folder/'proposals.json');b=read_json(E/'runs/devwarm01/sources'/folder.name/'proposals.json')
  same=[{k:p.get(k) for k in ('variant','query','prediction','status')} for p in a]==[{k:p.get(k) for k in ('variant','query','prediction','status')} for p in b]
  assert same; equality.append({'source_id':folder.name,'all_numerical_predictions_equal':same})
 runtime['warm_replay_numerical_equality']=equality
 atomic_json(O/'annotation_v4/runtime_profile.json',runtime)
 breakdown=[]
 for stage,p in profiles.items():
  for s in p['profiles']:
   for field in ('probe_decode_seconds','frame_sampling_seconds','clip_extraction_seconds','total_computation_wall_seconds','event_discovery_seconds','query_generation_seconds','ai_visual_review_seconds'):
    breakdown.append({'stage':stage,'source_id':s['source_id'],'cost':field,'seconds':s.get(field),'scope':'COMPUTATION_ONLY' if s.get(field) is not None else 'NOT_SEPARATELY_MEASURED'})
 for phase,p in runtime['fresh_clock']['interactive_phases'].items():breakdown.append({'stage':'fresh01','source_id':'BATCH','cost':phase,'seconds':p['wall_seconds'],'scope':p['scope']})
 wave=read_json(E/'fresh_wave/wave_sealed.json');overview=read_json(O/'benchmarks/fresh_overview_profile.json')
 breakdown.extend([{'stage':'fresh01','source_id':'BATCH','cost':'download_and_probe_wall_seconds','seconds':wave['download_and_probe_wall_seconds'],'scope':'Actual existing crawler wall time, includes probe/fingerprint and persisted ledger'},
   {'stage':'fresh01','source_id':'BATCH','cost':'overview_preparation_wall_seconds','seconds':overview['wall_seconds'],'scope':'Actual full-source frame extraction and overview assembly, reused by later indexing'},
   {'stage':'fresh01','source_id':'BATCH','cost':'runtime_warmup_seconds','seconds':profiles['fresh01']['runtime_warmup_seconds'],'scope':'Included within process wall, do not add twice'}])
 csvout(O/'benchmarks/throughput_breakdown.csv',breakdown)
 fresh_records=[read_json(p) for p in (E/'fresh_wave/records').glob('*.json')];n=len(fresh_records);k=len(fresh);technical=sum(r.get('technical',{}).get('pass',False) for r in fresh_records)
 duration=runtime['fresh_clock']['actual_experimental_end_to_end_seconds'];arms={r['source_video_id']:r['arm'] for r in read_json(O/'candidate_ranking/fresh25_selection.json')['selected']}
 arm_result={}
 for arm in ('priority','control'):
  ar=[r for r in fresh_records if arms[r['source_video_id']]==arm];ak=sum(arms[r['source_video_id']]==arm for r in fresh)
  arm_result[arm]={'attempted':len(ar),'technical_valid':sum(r.get('technical',{}).get('pass',False) for r in ar),'accepted':ak,'wilson95':wilson(ak,len(ar))}
 genuinely_new=[r for r in fresh_records if r['source_video_id'] not in overlap_ids];gn=len(genuinely_new);gk=sum(r['source_video_id'] not in overlap_ids for r in fresh)
 fresh_result={'one_real_wave_executed':True,'attempted':n,'technical_valid':technical,'accepted':k,'accepted_yield':k/n if n else None,'wilson95':wilson(k,n),'arms':arm_result,
  'genuinely_new_attempted':gn,'genuinely_new_technical_valid':sum(r.get('technical',{}).get('pass',False) for r in genuinely_new),'genuinely_new_accepted':gk,'genuinely_new_yield':gk/gn,'genuinely_new_wilson95':wilson(gk,gn),
  'preexisting_shared_raw_overlap':exposure['wave_preexisting_source_overlap'],'protocol_deviation_disclosed':True,
  'rank_advantage_identifiable':False,'whole_wave_genuinely_unseen':False,'genuinely_new_subgroup_source_held_out':True,'campaign_independence_proven':False,'new_human_gt':0,'experimental_end_to_end_seconds':duration,'accepted_per_hour':k*3600/duration,
  'rate_scope':'Measured primary experimental loop; postrun original-resolution QA/integrity cost excluded, not production throughput',
  'seconds_per_accepted':duration/k if k else None,'includes_baseline_ablation_and_actual_AI_inspection':True,'setup_excluded':True}
 fresh_result['source_outcomes']=[{'source_video_id':r['source_video_id'],'technical_pass':r.get('technical',{}).get('pass',False),
  'technical_reason':r.get('technical',{}).get('reason'),'duration_seconds':r.get('technical',{}).get('probe',{}).get('duration'),
  'final_status':decisions.get(r['source_video_id'],{}).get('status',r.get('status')),
  'exposure_group':'PREEXISTING_SHARED_RAW' if r['source_video_id'] in overlap_ids else 'GENUINELY_NEW_SOURCE'} for r in fresh_records]
 atomic_json(O/'benchmarks/fresh_wave_results.json',fresh_result)
 benchmark={'offline_ablation':aggregate,'fresh':fresh_result,'runtime_artifact':'../annotation_v4/runtime_profile.json','production_end_to_end_speedup_established':False,
  'historical_50min_comparison_used':False,'shared_preprocessing_not_double_charged':True,'true_interactive_times_not_inferred_from_tokens':True}
 atomic_json(O/'benchmarks/throughput_benchmark.json',benchmark)
 md(O/'benchmarks/candidate_selection_comparison.md',f"# Selection comparison\n\n{json.dumps(arm_result,indent=2)}\n\nAll predownload scores were zero because legitimate metadata is absent. Both arms therefore use the same seeded fallback. Differences are sampling noise, not evidence of ranking superiority. No statistical superiority is claimed.\n")
 total=500+len(all_new);remaining=1000-total;unused=1037-n;wavepoint=k/n;wavelo,wavehi=wilson(k,n);point=gk/gn;lo,hi=wilson(gk,gn)
 mean_bytes=statistics.mean(r['bytes'] for r in genuinely_new if r.get('bytes')) if any(r.get('bytes') for r in genuinely_new) else None
 scenarios=[]
 for name,p in [('Wilson_lower',lo),('point',point),('Wilson_upper',hi)]:
  attempts=math.ceil(remaining/p) if p else None
  scenarios.append({'scenario':name,'yield':p,'remaining_candidate_capacity':math.floor(unused*p) if p is not None else None,'attempts_needed':attempts,'fits_pool':attempts is not None and attempts<=unused,
   'projected_raw_download_bytes':attempts*mean_bytes if attempts is not None and mean_bytes else None,'projected_hours_primary_loop_optimistic':attempts*duration/gn/3600 if attempts is not None and gn else None})
 free=shutil.disk_usage(R).free;required_yield=remaining/unused if unused else None
 feasibility={'accepted_unique_pool':total,'historical_d500_frozen_rows':500,'original_accepted18':18,'recovered_accepted':len(recovered),'fresh_accepted':k,'remaining_to1000':remaining,'unused_candidates':unused,
  'required_future_yield':required_yield,'fresh_yield':point,'fresh_wilson95':[lo,hi],'scenarios':scenarios,'E_free_bytes':free,'minimum_free_bytes':15*2**30,'processing_reserve_bytes':2*2**30,'mean_fresh_raw_bytes':mean_bytes,
  'yield_basis':'Genuinely new subgroup1/24; whole actual wave1/25 separately reported','remaining_cached_requires_revalidation':exposure['remaining_already_cached'],'remaining_network_candidates':exposure['remaining_not_in_shared_raw'],
  'required_new_yield_even_if_all_cached_accepted':max(0,remaining-exposure['remaining_already_cached'])/exposure['remaining_not_in_shared_raw'],
  'usable_bytes_after_reserves':max(0,free-17*2**30),'forecast_assumptions':'Stationary yield and source size, serial measured experimental workflow; candidate availability and campaign uniqueness not guaranteed; tiny wave uncertainty is material',
  'decision':'NO-GO','production_promotion':False,'reason':'Component truncation and source-wise quality regressions remain; metadata cannot identify better candidates; measured evidence does not validate sustainable D1000 throughput. No unbounded acquisition.'}
 atomic_json(O/'reports/feasibility.json',feasibility)
 csvout(O/'reports/feasibility_scenarios.csv',scenarios)
 validation=read_json(O/'validation/records_and_resume.json');tests=read_json(O/'validation/test_summary.json');historical=read_json(O/'validation/historical_final.json')
 assert validation['pass'] and tests['pass'] and historical['pass']
 md(O/'annotation_v4/validation_report.md',f"Actual contract/cache/resume validation PASS. Tests: {tests['tests']}; accepted additions{len(new)}. Experimental source-wise regressions remain: no production promotion. Human verification0; IAA=null. See full regression and integrity JSON.\n")
 md(O/'benchmarks/throughput_validation_report.md',f"Actual fresh wave{n} sources, accepted{k}; end-to-end{duration:.2f}s, accepted/hour{fresh_result['accepted_per_hour']:.3f}. Includes real AI interaction and paired ablation overhead. No production baseline end-to-end speedup established. Metadata arms are equal priors. Setup/report writing excluded; per-source AI times remain null.\n")
 files=['data_process/lumae_scale/annotation_v4.py','data_process/lumae_scale/candidate_priority_v1.py','data_process/lumae_scale/p21_write_guard.py','data_process/lumae_scale/p21_quality_registry.py','data_process/lumae_scale/source_inventory_guard_v2.py','data_process/tests/test_annotation_v4.py','data_process/tests/test_candidate_priority_v1.py','data_process/tests/test_p21_review_reconcile.py','data_process/tests/test_p21_quality_registry.py','data_process/tests/test_source_inventory_guard_v2.py','research_outputs/p21_20261103_01/validation/python_guard/sitecustomize.py']+[str(p.relative_to(ROOT)).replace('\\','/') for p in sorted((ROOT/'scripts').glob('p21_*.py'))]
 purposes={'annotation_v4.py':'versioned event selection, adaptive sampling, component proposal, query-only procedure and exact-input cache','candidate_priority_v1.py':'metadata-safe deterministic equal-prior ranking','p21_write_guard.py':'prevent tests writing sealed P16/P2 state','test_annotation_v4.py':'boundary/cache/provenance/abstention tests','test_candidate_priority_v1.py':'ranking determinism, metadata leakage and domain tests','p21_audit.py':'read-only inventory and evaluation split','p21_process.py':'actual CPU paired ablation and runtime profiling','p21_review.py':'actual image review import and strict acceptance','p21_dev_review_notes.py':'transcribe actual development image judgments','p21_reserved_notes.py':'transcribe actual reserved overview events','p21_reserved_review_notes.py':'transcribe actual reserved boundary judgments','p21_rank_freeze.py':'freeze selection policy and experimental code','p21_fresh_wave.py':'bounded one-wave adapter around existing crawler','p21_fresh_prepare.py':'fresh overview evidence and batch clock','p21_validate.py':'historical, test, cache, acceptance and resume checks','p21_delivery.py':'reconcile actual metrics, feasibility and seal'}
 purposes.update({'p21_fresh_notes.py':'convert actual fresh event/query notes into frozen contracts; generic review helper limitation is handled separately',
  'p21_quality_registry.py':'apply evidence/identity-bound final withdrawals to the effective pool without rewriting provisional trial ledgers',
  'test_p21_quality_registry.py':'two tests prove withdrawals remove only exact new trial identities and cannot mutate records or silently target history',
  'source_inventory_guard_v2.py':'postrun offline guard separates cached assets, attempted sources and network-eligible sources across all registered raw roots',
  'test_source_inventory_guard_v2.py':'two tests prevent duplicate-network selection and fail closed on unavailable storage roots',
  'p21_review_reconcile.py':'postrun fail-closed reconciliation of actual sanitation abstention; original timed manual procedure preserved',
  'test_p21_review_reconcile.py':'two tests prove abstention cannot become acceptance and replay is immutable/idempotent',
  'sitecustomize.py':'inherit historical-write protection into Streamlit and other test subprocesses'})
 validations={
  'p21_quality_registry.py':'2 registry tests PASS; actual1 withdrawal and5 retained additions; raw record/evidence hashes verified',
  'test_p21_quality_registry.py':'2 PASS in quality_registry_tests.xml',
  'source_inventory_guard_v2.py':'2 tests PASS; actual remaining pool partition990 network/22 cached/25 attempted; not live evaluated',
  'test_source_inventory_guard_v2.py':'2 PASS in source_inventory_guard_tests.xml',
  'annotation_v4.py':'test_annotation_v4.py:7 PASS; real67-source localization and18-source warm replay',
  'candidate_priority_v1.py':'test_candidate_priority_v1.py:2 PASS;1037 actual equal-prior rows and frozen25 selection',
  'p21_write_guard.py':'Full327-case regression under guard:326 pass and1 UI idle retry pass; historical final full hash pass',
  'sitecustomize.py':'Full regression and UI idle retry inherited protection;44387 historical files unchanged',
  'test_annotation_v4.py':'7 PASS in prewave/full suite','test_candidate_priority_v1.py':'2 PASS in prewave/full suite',
  'test_p21_review_reconcile.py':'2 PASS in reconcile_tests_final.xml; initial import errors retained in failed XML',
  'p21_review_reconcile.py':'2 fail-closed/idempotence tests PASS; actual23 bundles+1 machine abstention reproduced byte-exact',
  'p21_audit.py':'61-source raw identity inventory, grouped20/41 freeze;44387-file full preflight',
  'p21_dev_review_notes.py':'Actual20 overview inspections and18 dense reviews;2 provisional passes,1 withdrawn after final native-resolution audit',
  'p21_reserved_notes.py':'Actual41 overview inspections;30 localization attempts and11 no-event abstentions',
  'p21_reserved_review_notes.py':'Actual30 dense source reviews; strict import recovered3',
  'p21_process.py':'Actual dev20/reserved41/fresh24;402 B0-B5 proposals;18 warm numerical-equality checks',
  'p21_rank_freeze.py':'1037-row eligibility reconstruction;25 deterministic selected IDs frozen before network/images',
  'p21_review.py':'6 provisional records pass structural contract;5 retained after1 final QC withdrawal;85 valid sources reconciled and3 builder resumes unchanged',
  'p21_fresh_notes.py':'24 actual fresh overviews,20 original queries;19 localized and1 sanitation abstention, no post-evaluation revision',
  'p21_fresh_prepare.py':'24 actual uniform evidence pages; actual1464.05-second whole-workflow clock',
  'p21_fresh_wave.py':'25 actual requests/25 source records, quotas preserved; sealed-wave refusal checked with network denied',
  'p21_validate.py':'Actual record/cache/raw/frozen hash validation;333 effective test cases, initial failures retained',
  'p21_delivery.py':'Executed reconciliation, Wilson/scenario arithmetic and final artifact/manifest verification'}
 changed=[{'path':str(ROOT/f),'sha256':digest(ROOT/f),'purpose':purposes[Path(f).name],'why_necessary':'Isolate P21 work from sealed history; '+purposes[Path(f).name],
   'change':'NEW_FILE_ONLY','validation':validations[Path(f).name]} for f in files]
 csvout(O/'reports/changed_files.csv',changed)
 code_details='\n'.join(f"- `{x['path']}`: {x['purpose']}. Necessary because: {x['why_necessary']}. Validation: {x['validation']}." for x in changed)+'\n'
 md(O/'reports/code_changes.md',code_details)
 report=f'''# A. Executive Summary\n\n**NO-GO cho mở rộng D1000 và promotion v4.** Pool accepted hiện{total}, còn thiếu{remaining}. Phục hồi thật{len(recovered)}/61 nguồn cũ; fresh{k}/{n}. D500 và18 accepted cũ giữ nguyên; đây là pool nghiên cứu, chưa đóng băng D1000.\n\n# B. Research State Audit\n\nĐã đối chiếu trạng thái518 ban đầu,61 nguồn chưa có annotation, toàn bộ80 raw cũ và manifest/code/evidence đóng băng. Kiểm tra{historical['checked']} file lịch sử đọc được pass; D48 ACL vẫn hạn chế và nội dung ZIP cloud chưa xác minh. CPU-only, CLIP local; không huấn luyện/GPU/ZIP/migration.\n\n# C. Recovery61\n\nThử lại đủ61, không tải lại raw. Trạng thái{dict(status)}. Nguồn được nhận có window nguyên từ proposal, review hình ảnh thực và validator hiện có. Campaign không suy ra từ hash; human_verified=false, Test bị hạn chế. Chi tiết từng nguồn: recovery_acceptance_decisions.jsonl và taxonomy CSV.\n\n# D. Annotation optimization\n\nB0–B5 đã chạy thật trên20 dev và41 reserved; số liệu chi tiết ở ablation_results.csv. V4 bỏ padding/floor nhưng nhiều trường hợp cắt mất hành động. Baseline có trường hợp tốt hơn; chỉ giữ fallback có lineage rõ. B0/B1 không tách được tác động O1; B5 chỉ kiểm tra cache/repeat, không chứng minh tốc độ toàn pipeline. Không tuning trên reserved/fresh sau freeze. Reserved61 đã từng phơi lộ ở tác vụ trước; không gọi là unseen GT.\n\n# E. Candidate prioritization\n\n1.037 ứng viên trước wave thiếu title/description/category/duration/brand/campaign; score đều0, tie-break SHA256(seed:ID). source_split/GT/CLIP sau tải không được dùng. Hai arm cùng fallback nên không có bằng chứng ranking tốt hơn.\n\n# F. Throughput benchmark\n\nMột wave thật{n} nguồn; technical-valid{technical}, accepted{k}; yield{point:.3%}, Wilson95%[{lo:.3%},{hi:.3%}]. Thời gian workflow thực{duration/60:.2f} phút; {fresh_result['accepted_per_hour']:.3f} accepted/giờ. Bao gồm tải/probe/frames/CLIP/ablations và tương tác AI thực; không gồm coding/setup/report. Thời gian AI theo batch, không bịa theo từng nguồn. Chi phí USD/model revision không được interface cung cấp nên để null/NOT_EXPOSED.\n\n# G. D1000 feasibility\n\nCòn{remaining} accepted và{unused} ứng viên chưa thử; cần yield tương lai{required_yield:.3%}. Dự báo Wilson lower/point/upper trong feasibility_scenarios.csv, không dùng yield recovery có selection bias để thay yield fresh. E trống{free/2**30:.2f}GiB, giữ15GiB+2GiB reserve; dự báo raw theo kích thước fresh thực. Không có bảo đảm URL/campaign còn hợp lệ hoặc yield ổn định.\n\n# H. Research integrity\n\nCanonical/splits/checkpoint/ranker/frozen manifests cũ không sửa. Tất cả record mới là AI pseudo, human0, IAA=null, chưa phải publication Test GT. Không sửa lịch sử kết quả nghiên cứu, không tạo D1000 freeze. Review AI không độc lập mô hình và không blind; mọi giới hạn được ghi rõ.\n\n# I. Code changes\n\nTất cả file code mới, mục đích và kiểm tra được liệt kê đầy đủ trong reports/code_changes.md và changed_files.csv, kèm SHA256. Không sửa file code được manifest cũ pin.\n\n# J. Validation\n\n{tests['tests']} test pass; record/cache/raw hashes/unique IDs/unique bytes/resume offline pass. Không có HTTP trong bước validation. Warm-cache replay kiểm tra numerical prediction equality trên{len(equality)} nguồn có event. Crawler một wave, <=25 nguồn/50 request, quota/rate/concurrency và reserve được kiểm tra.\n\n# K. Blockers\n\nChưa giải quyết đầy đủ action completeness, short-event boundaries và source-wise quality regression. Không có metadata giúp ranking nội dung, không có calibrated independent semantic verifier, campaign independence chưa chứng minh. Những điều này chưa được che bằng HIGH confidence hoặc human giả. Cloud backup content còn unverified.\n\n# L. Validated next stage\n\nKhông có lệnh scale được phê duyệt kỹ thuật từ kết quả này. Wave P21 đã sealed; lệnh wave sẽ từ chối chạy thêm. Chỉ chạy lại kiểm tra offline: `python scripts/p21_validate.py records` và `python scripts/p21_validate.py historical`. Cần phiên bản thuật toán mới giải quyết regression, protocol đánh giá độc lập phù hợp và feasibility đạt trên nguồn fresh trước khi xem xét scope acquisition tiếp theo. Không tự tăng ngân sách hay thay quality gate.\n'''
 for old_title,new_title in {
  '# B. Research State Audit':'# B. Audit Before Change',
  '# C. Recovery61':'# C. P2.1a — Recovery of Existing 61 Sources',
  '# D. Annotation optimization':'# D. P2.1b — Annotation Optimization',
  '# E. Candidate prioritization':'# E. P2.1c — Candidate Prioritization',
  '# F. Throughput benchmark':'# F. P2.1d — Controlled Throughput Benchmark',
  '# G. D1000 feasibility':'# G. Updated D1000 Feasibility',
  '# H. Research integrity':'# H. Research Integrity',
  '# I. Code changes':'# I. Code Changes',
  '# J. Validation':'# J. Validation Evidence',
  '# K. Blockers':'# K. Remaining Blockers',
  '# L. Validated next stage':'# L. Next Execution Stage'}.items():report=report.replace(old_title,new_title)
 report=report.replace('# J. Validation Evidence',code_details+'\n# J. Validation Evidence')
 transitions_summary=dict(Counter(r['B0_to_B4'] for r in transitions))
 timing_note=f"\n\nFresh timing interpretation: discovery{runtime['fresh_clock']['interactive_phases']['event_discovery']['wall_seconds']:.2f}s includes initial query formulation. The query_generation clock{runtime['fresh_clock']['interactive_phases']['query_generation']['wall_seconds']:.2f}s measures final wording/data transcription only; do not interpret it as total semantic query-generation cost. Visual QC{runtime['fresh_clock']['interactive_phases']['ai_visual_review']['wall_seconds']:.2f}s; save/validation{runtime['fresh_clock']['interactive_phases']['save_validation']['wall_seconds']:.2f}s. Download/probe{wave['download_and_probe_wall_seconds']:.2f}s; overview{overview['wall_seconds']:.2f}s; frozen computation{profiles['fresh01']['total_process_wall_seconds']:.2f}s including warmup. Remaining elapsed time is real orchestration/wait overhead. Development interactive times remain unknown.\n\nPaired B0→B4 source transitions across85 technically valid sources: {transitions_summary}; see paired_source_transitions.csv. These are AI-bracket proxy outcomes, not human GT. The one invalid fresh source is included in acquisition yield1/25 but excluded from annotation ablation denominator24.\n\n"
 report=report.replace('# G. Updated D1000 Feasibility',timing_note+'# G. Updated D1000 Feasibility')
 evidence_note=f"\n\nTest resolution:326 full-run PASS,1 UI timeout PASS on idle retry,2 reconciliation tests PASS =329 effective cases. The initial16 sandbox failures and adapter import failures are preserved in XML; not a single clean329-case full run. Read-only validation confirms all80 original raw MP4 hashes,67 used visual indices,3 unchanged builder resumes, and source/hash uniqueness.\n\nValidated commands: `python scripts/p21_validate.py tests`; `python scripts/p21_validate.py historical --full`; `python scripts/p21_validate.py records`; actual processing used `python scripts/p21_process.py --stage dev01 --events research_outputs/p21_20261103_01/recovery/development_events.json`, corresponding reserved01/fresh01 input files, and devwarm01 replay. Actual single-wave invocation was `python scripts/p21_fresh_wave.py wave`; its configuration is now sealed and cannot authorize a second wave.\n\nFrozen code/config binding SHA256: `{digest(O/'manifests/experimental_pipeline_freeze.json')}`. CLIP weight SHA256: `{runtime['clip_model_sha256']}`. Previous execution manifest SHA256: `{read_json(O/'audit/audit_before_change.json')['p2_execution_manifest_sha256']}`. Final P21 manifest SHA is in manifests/p21_execution_manifest.sha256.\n\n"
 report=report.replace('# K. Remaining Blockers',evidence_note+'# K. Remaining Blockers')
 blocker_note='''\n\n- Model: coarse-frame semantic errors, action-direction mistakes and component truncation; no calibrated independent verifier demonstrated.\n- Annotation: incomplete/ambiguous actions and strict short-event boundaries; one valid-looking plant query failed the unchanged sanitation subject rule. Frozen inference was not revised on fresh data. A separate fail-closed reconciliation adapter reproduces the timed manual abstention handling and is not an evaluated annotation-method improvement.\n- Candidate supply: missing semantic metadata; future yield required exceeds measured evidence.\n- Campaigns: fingerprint warnings and unknown campaign relations remain unresolved; every addition remains split-restricted.\n- Human verification:0 new genuine human reviews, IAA=null; publication Test GT unavailable.\n- Storage: current reserves pass; predicted attempts at low yield cannot fit candidate supply and would consume much more raw storage/time.\n- Permissions: current one-wave operator authorization was used on the existing allowlist; all25 requests succeeded. No independent license document was verified, and no access restriction was bypassed.\n- External dependencies: cloud ZIP content still unverified; D48 ACL scope explicitly excluded from full-readable-file claim. GPU/Colab training remains out of scope.\n\n'''
 report=report.replace('# L. Next Execution Stage',blocker_note+'# L. Next Execution Stage')
 report=report.replace(f'yield{point:.3%}, Wilson95%[{lo:.3%},{hi:.3%}]',f'whole-wave yield{wavepoint:.3%}, Wilson95%[{wavelo:.3%},{wavehi:.3%}]; genuinely-new subgroup{gk}/{gn}, yield{point:.3%}, Wilson95%[{lo:.3%},{hi:.3%}]')
 report=report.replace('2 reconciliation tests PASS =329 effective cases','2 reconciliation +2 final-quality-registry +2 offline-inventory tests PASS =333 effective cases').replace('single clean329-case full run','single clean333-case full run')
 integrity_note=f"\n\nFinal quality audit:6 provisional P21 acceptances were reviewed at original resolution. SPAR was already hugging before model onset16.375, so that record was withdrawn via a hash-bound exclusion registry. Proposals and original trial ledgers were preserved. Effective additions are{len(new)}: recovery{len(recovered)}+fresh{k}. Only the filtered accepted_additions_pool.jsonl may be used downstream.\n\n"
 report=report.replace('# D. P2.1b — Annotation Optimization',integrity_note+'# D. P2.1b — Annotation Optimization')
 exposure_note=f"\n\nProtocol deviation:1/25 selected URLs already existed byte-exact in shared raw616 (26,050,422 bytes). It was outside prior80 and not accepted, but cannot count as genuinely unseen. Pure new-source cohort has{gn} candidates and{fresh_result['genuinely_new_technical_valid']} technically valid videos. No replacement download occurred. Of{unused} remaining candidates,{exposure['remaining_already_cached']} are cached and require reuse/revalidation;{exposure['remaining_not_in_shared_raw']} are network-new. A new inventory guard was tested offline after this finding; it was not the evaluated selection policy and has no live performance claim.\n\nTiming boundary:1464.05s covers the primary experimental workflow through first acceptance persistence. Subsequent original-resolution QA and final integrity checks are outside that clock. The reported primary-loop rate is not validated production throughput.\n\n"
 report=report.replace('# G. Updated D1000 Feasibility',exposure_note+'# G. Updated D1000 Feasibility')
 report=report.replace('# L. Next Execution Stage','\nThe next stage must additionally enforce the hash-bound withdrawal registry and the all-raw-root inventory guard. The frozenV1 selection missed23 preexisting cached candidates in its1037-row pool; only1 was selected in this wave. Do not treat the postrun offline fix as a fresh-validated method.\n\n# L. Next Execution Stage')
 report=report.replace('# E. P2.1c — Candidate Prioritization',f"Warm-cache compute replay: {profiles['dev01']['total_process_wall_seconds']:.2f}s cold versus {warm['total_process_wall_seconds']:.2f}s warm;18 event-source predictions remained numerically identical. This is an observed computation comparison under shared-machine conditions, not causal or production end-to-end speedup.\n\n# E. P2.1c — Candidate Prioritization")
 report=report.replace('# G. Updated D1000 Feasibility','Technical rejection: source11ea25826b1b10bd425aae873a52d3fd has duration176s, exceeding the unchanged150s limit; it was downloaded/probed but not annotated. Per-source outcomes are in fresh_wave_results.json.\n\n# G. Updated D1000 Feasibility')
 md(O/'reports/p21_final_report.md',report)
 md(O/'candidate_ranking/report.md',f"# Candidate prioritization report\n\n1037 V1 candidates had no legitimate semantic metadata; all scores0 and both selection arms used the same seeded fallback. No ranking advantage is established.\n\nV1 eligibility missed23 already-cached shared raw sources. The wave selected1 of those, downloading26,050,422 duplicate bytes. It is excluded from genuinely new-source evaluation. This deviation invalidates a claim that all25 sources were genuinely fresh. The unchanged original policy/selection remain preserved.\n\nPostrun offline inventory guard separates990 network-new candidates,22 cached candidates requiring reuse/revalidation, and25 attempted. Its2 tests pass, but it has not been live evaluated and does not authorize another wave.\n")
 for folder,pairs in {
  'annotation_v4':{'design.md':'annotation_v4_design.md','ablation_results.csv':'annotation_ablation_results.csv','runtime_profile.json':'annotation_runtime_profile.json','validation_report.md':'annotation_v4_validation_report.md'},
  'candidate_ranking':{'policy.md':'candidate_prioritization_policy.md','ranking_validation.json':'candidate_ranking_validation.json','report.md':'candidate_prioritization_report.md'}}.items():
  for src,dst in pairs.items():(O/folder/dst).write_bytes((O/folder/src).read_bytes())
 atomic_json(O/'candidate_ranking/candidate_ranking_validation.json',{'original_frozen_V1_validation':read_json(O/'candidate_ranking/ranking_validation.json'),'cache_eligibility_complete_in_V1':False,'observed_cached_overlap_in_wave':1,'genuinely_new_sources':gn,
   'offline_V2_guard_counts':read_json(O/'candidate_ranking/next_pool_cache_audit.json')['counts'],'offline_V2_live_evaluated':False,'ranking_superiority_established':False,'production_promotion':False})
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,ax=plt.subplots(figsize=(8,2.8));ax.errorbar([point*100],[0],xerr=[[(point-lo)*100],[(hi-point)*100]],fmt='o',capsize=6,color='#176c9c',markersize=8)
 ax.axvline(required_yield*100,color='#b34235',linewidth=2)
 ax.text(point*100,.13,f'{gk}/{gn} genuinely new sources: {point:.1%}',color='#176c9c',fontsize=10)
 ax.text(required_yield*100+1,-.08,f'Needed\n{required_yield:.1%}',color='#b34235',fontsize=10)
 ax.set(xlim=(0,60),ylim=(-.35,.35),yticks=[],xlabel='Accepted source yield (%)',title='Fresh-source yield versus D1000 requirement')
 ax.grid(axis='x',alpha=.2);ax.spines[['top','right','left']].set_visible(False)
 fig.text(.12,.015,'Wilson95% interval; one cached source excluded. Whole wave1/25. No production promotion.',fontsize=8,color='#555555')
 fig.tight_layout(rect=(0,.05,1,1));fig.savefig(O/'reports/fresh_yield_feasibility.png',dpi=180);plt.close(fig)
 manifest={'created_utc':datetime.now(timezone.utc).isoformat(),'decision':'NO-GO','accepted_unique_pool':total,'provisional_p21_acceptances':len(provisional),'withdrawn_p21_acceptances':len(exclusions),'effective_p21_acceptances':len(new),'d1000_complete':False,'d1000_frozen':False,'human_verified':0,'old_execution_manifest_sha256':read_json(O/'audit/audit_before_change.json')['p2_execution_manifest_sha256'],
  'artifact_sha256':{str(p.relative_to(O)).replace('\\','/'):digest(p) for p in O.rglob('*') if p.is_file() and p.name not in ('p21_execution_manifest.json','p21_execution_manifest.sha256','recovery_manifest.json')},
  'processing_code_sha256':{x['path']:x['sha256'] for x in changed},'evidence_sha256':{str(p):digest(p) for p in E.rglob('*') if p.is_file() and p.name!='run.lock'},
  'accepted_record_sha256':{str(p):digest(p) for stage in stages for p in (R/'work'/('p21_'+stage)/'accepted').glob('*.json')},
  'builder_state_sha256':{str(p):digest(p) for stage in stages for p in (R/'work'/('p21_'+stage)).rglob('*') if p.is_file()}}
 atomic_json(O/'recovery/recovery_manifest.json',{'sources':{r['source_id']:r['sha256'] for r in read_json(O/'audit/recovery_inventory.json')},'attempted':61,'accepted':len(recovered),'human_verified':False,'original18_preserved':True,'evaluation_protocol_sha256':digest(O/'manifests/evaluation_protocol.json'),'accepted_pairs_sha256':digest(O/'recovery/recovered_accepted_pairs.jsonl')})
 manifest['artifact_sha256']['recovery/recovery_manifest.json']=digest(O/'recovery/recovery_manifest.json')
 atomic_json(O/'manifests/p21_execution_manifest.json',manifest);(O/'manifests/p21_execution_manifest.sha256').write_text(digest(O/'manifests/p21_execution_manifest.json')+'\n')
 print(json.dumps({'decision':manifest['decision'],'accepted_pool':total,'recovered':len(recovered),'fresh':k,'remaining':remaining,'manifest_sha256':digest(O/'manifests/p21_execution_manifest.json')}))
if __name__=='__main__':main()
