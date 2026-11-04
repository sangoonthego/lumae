import os
from pathlib import Path
import pytest

def test_real_research_corpus_write_is_blocked_before_mutation(tmp_path):
    target=Path(__file__).resolve().parents[2]/'local_data/manifests/semantic_v3_clean_query_blind5_query_freeze.json'
    before=target.read_bytes()
    with pytest.raises(AssertionError,match='protected historical write'):target.write_text('must not be written')
    with pytest.raises(AssertionError,match='protected historical write'):os.open(target,os.O_WRONLY|os.O_TRUNC)
    assert target.read_bytes()==before
    fixture=tmp_path/'allowed.txt';fixture.write_text('new synthetic fixture');assert fixture.read_text()=='new synthetic fixture'
