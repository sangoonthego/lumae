"""Bind reviewed P1 preflight/tests to a bounded immutable pilot configuration."""
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from p1_preflight import compare
from data_process.lumae_scale.models import ROOT,read_json,atomic_json,digest
from data_process.lumae_scale.incremental import immutable_write,json_bytes

out=ROOT/'research_outputs/p1_20261103_01'
integrity=read_json(out/'d500_integrity.json');storage=read_json(out/'storage_preflight.json')
protected=compare(read_json(out/'protected_asset_snapshot.json'));atomic_json(out/'protected_pre_live_validation.json',protected)
suites=list(ET.parse(out/'targeted_validation_fixed.xml').getroot().iter('testsuite'))
tests_pass=bool(suites) and all(int(s.attrib.get('failures',0))+int(s.attrib.get('errors',0))==0 for s in suites)
source='local_data/raw/adsqa/source_manifest.json';h=digest(ROOT/source)
gate={'protected_assets_pass':protected['pass'] and integrity['pass'],'storage_pass':storage['pass'],'source_permissions_attested':read_json(out/'permission_attestation.json')['status']=='OPERATOR_ATTESTED','crawler_tests_pass':tests_pass,'source_manifest_sha256':h,'preflight_evidence':str(out),'cloud_sha_verified':False,'cleanup_allowed':False}
immutable_write(out/'operational_gate.json',json_bytes(gate))
cfg={'run_id':'20261103_01','mode':'pilot','builder_config':'data_process/config/d1000_v1.json','data_root':'E:/LUMAE','source_manifest':source,'source_manifest_sha256':h,
     'baseline_fingerprints':str(out/'d500_fingerprint_baseline.json'),'operational_gate':str(out/'operational_gate.json'),
     'target_accepted':25,'max_candidates':30,'max_http_requests':45,'max_download_bytes':2*2**30,'max_file_bytes':100*2**20,'max_runtime_seconds':3600,'max_staging_bytes':int(2.5*2**30),'processing_reserve_bytes':256*2**20,'min_free_bytes':15*2**30,'max_consecutive_failures':3,'max_domain_error_rate':.6,'domain_error_min_requests':5,'retries_per_source':2,'timeout_seconds':30,'request_interval_seconds':2,'concurrency':1,'network_execution_authorized':True,'source_permission':read_json(out/'permission_attestation.json')}
immutable_write(out/'pilot_config.json',json_bytes(cfg))
annotation=read_json(out/'annotation_readiness.json');annotation['cpu_smoke']=read_json(out/'model_dependency_smoke.json');atomic_json(out/'annotation_readiness.json',annotation)
crawler=read_json(out/'crawler_readiness.json');crawler.update({'p1_targeted_tests':{'tests':sum(int(s.attrib['tests']) for s in suites),'pass':tests_pass},'network_preflight':read_json(out/'network_preflight.json'),'live_gate':gate,'controls_implementation':'data_process/lumae_scale/acquisition.py'});atomic_json(out/'crawler_readiness.json',crawler)
(out/'p1_preflight_report.md').write_text('''# P1.0 — Operational Preflight

VERIFIED: D500 canonical/split checksums, 500 video SHA-256, local ZIP SHA-256 and M0 identity checked afresh. 44,387 readable protected files snapshotted; the previously inaccessible D48 directory remains explicitly reported. P0/audit outputs are protected. Pre-live stat comparison against the fresh full-SHA snapshot passes.

VERIFIED: E: currently has 54,578,716,672 bytes free. Projected pilot peak 2.75 GiB plus 15 GiB reserve passes. E: remains working storage on the same physical disk as C:/D:, not independent backup. No D500 migration is needed.

VERIFIED: 1,117 metadata-eligible candidates after parent/source/URL/inherited rejection exclusions. The 116 existing raw files outside D500 are not imported or counted accepted. Source candidate audit records prior states and reasons. First permitted-host video HEAD returns HTTP 200, 68,589,242 bytes, video/mp4. robots.txt returns 403; no video access controls will be bypassed.

OPERATOR_ATTESTED: User confirmed existing acquisition permission in this chat. A separate license document has not been independently verified. Upstream acquisition guidance: https://github.com/TsinghuaC3I/AdsQA#data-acquisition . AOTW legal framework: https://www.adsoftheworld.com/legal and https://clios.com/clio-awards-llc-terms-of-service/ . These sources are not treated as a blanket research license.

VERIFIED: bounded P1 collector and real generated-video tests pass, default planning denies HTTP, concurrency=1, rate interval=2s, retry/backoff, stream budgets, Range+If-Range, response validation, atomic/durable manifests and access-denial terminal states are implemented. Pilot config pins source and parent. Start with --max-wave 3.

VERIFIED: CLIP local CPU image/text smoke is finite (1,512), no CUDA, frozen Semantic V3 source matches. TODO: actual event query inspection, semantic interval verification and genuine independent human labels. Compact CPU VLM abstains; no unattended annotation is claimed. Technically valid acquired files are quarantined pending full D500/pilot campaign review, not automatically accepted.

Gate: controlled acquisition permitted by operator attestation and passed local storage/integrity/tests. No destructive cleanup, final D1000 freeze, full-scale campaign or GPU experiment authorized. Cloud ZIP content remains UNVERIFIED.
''',encoding='utf-8')
print(json.dumps(gate))
