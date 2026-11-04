import pytest
from data_process.lumae_scale.source_inventory_guard_v2 import inventory_mp4,partition_candidates

def test_all_registered_roots_prevent_duplicate_network_and_never_auto_accept(tmp_path):
 a=tmp_path/'shared';b=tmp_path/'previous';a.mkdir();b.mkdir();(a/'cached.mp4').write_bytes(b'fixture');(b/'attempted.mp4').write_bytes(b'fixture')
 inv=inventory_mp4([a,b]);rows=[{'source_video_id':s,'source_split':'not-used'} for s in ('new','cached','attempted')]
 result=partition_candidates(rows,inv,{'attempted'})
 assert [r['source_video_id'] for r in result['network_eligible']]==['new']
 assert result['cached_requires_revalidation'][0]['automatic_acceptance'] is False
 assert [r['source_video_id'] for r in result['already_attempted']]==['attempted']

def test_missing_root_blocks_instead_of_assuming_empty(tmp_path):
 with pytest.raises(FileNotFoundError):inventory_mp4([tmp_path/'missing'])
