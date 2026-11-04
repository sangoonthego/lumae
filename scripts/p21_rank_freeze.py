"""Freeze predownload fallback policy and experimental annotation before holdout."""
import json,sys
from datetime import datetime,timezone
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from p21_audit import O,R,ROOT,csvout
from data_process.lumae_scale.models import read_json,digest
from data_process.lumae_scale.acquisition import Pilot
from data_process.lumae_scale.builder import BuildConfig
from data_process.lumae_scale.incremental import immutable_write,json_bytes
from data_process.lumae_scale.candidate_priority_v1 import rank_candidates

def main():
 c=read_json(ROOT/'research_outputs/p16_p2_20261103_01/p2_config.json')
 p=Pilot(c,BuildConfig.load(ROOT/c['builder_config'],ROOT),read_json(ROOT/c['source_manifest'])['videos'],live=False)
 used={r['source_video_id'] for r in p.records()}
 pool=[r for r in p.plan['candidates'] if r['source_video_id'] not in used]
 ranking=rank_candidates(pool);assert len(ranking)==1037
 csvout(O/'candidate_ranking/candidate_features_manifest.csv',[{'source_video_id':r['source_video_id'],'source_url':r['source_url'],**{k:r.get(k) for k in ('title','description','category','brand','campaign','duration')}} for r in pool])
 csvout(O/'candidate_ranking/candidate_priority_scores.csv',ranking)
 immutable_write(O/'candidate_ranking/ranking_validation.json',json_bytes({'pool_count':len(pool),'all_equal_priors':True,'metadata_quality_ranking_supported':False,'source_split_used':False,'postdownload_features_used':False,'seed':20261103,'missing_metadata_not_inferred':True}))
 policy='''# Candidate policy v1\n\nOnly HTTPS video.adsoftheworld.com URLs already in the pinned source pool are eligible. Exclude D500, original pilot30, and all50 P2 attempted source IDs before ranking. Every remaining record exposes only ID, URL, target_name and upstream source_split. No title, description, category, brand, campaign or duration exists. source_split is forbidden in scoring.\n\nAll scores are exactly zero. SHA256(seed:source_id), seed20261103, resolves ties deterministically. This is a seeded fallback, not evidence of semantic intelligence. No CLIP, downloaded frames, queries, labels, accept/reject results, or Test information enters ranking. Optional metadata availability never predicts quality without calibration.\n\nIf the single validation wave gate passes, freeze the first25 IDs before downloads. Alternating assignment priority13/control12 is operational bookkeeping only: with equal priors, the policies coincide and no ranking advantage can be inferred. No new candidates are added after failures.\n'''
 (O/'candidate_ranking/policy.md').write_text(policy,encoding='utf-8')
 (O/'candidate_ranking/report.md').write_text('1037 eligible unused candidates. Metadata ranking is blocked by absent metadata; implemented and validated deterministic fallback. No improvement claimed.\n',encoding='utf-8')
 immutable_write(O/'candidate_ranking/fresh25_selection.json',json_bytes({'seed':20261103,'selected':[dict(r,arm='priority' if i%2==0 else 'control') for i,r in enumerate(ranking[:25])],'max_sources':25,'no_ranking_advantage_identifiable':True}))
 files=['data_process/lumae_scale/annotation_v4.py','data_process/lumae_scale/candidate_priority_v1.py','scripts/p21_process.py','scripts/p21_review.py']
 immutable_write(O/'manifests/experimental_pipeline_freeze.json',json_bytes({'created_utc':datetime.now(timezone.utc).isoformat(),'code_sha256':{f:digest(ROOT/f) for f in files},'config_sha256':digest(O/'annotation_v4/config.json'),'evaluation_protocol_sha256':digest(O/'manifests/evaluation_protocol.json'),'candidate_policy_sha256':digest(O/'candidate_ranking/policy.md'),'selection_sha256':digest(O/'candidate_ranking/fresh25_selection.json'),'status':'EXPERIMENTAL_NOT_PROMOTED','human_gt':False,'no_more_tuning_on_reserved_or_fresh':True}))
 print('FROZEN',len(pool))
if __name__=='__main__':main()
