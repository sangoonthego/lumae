"""Regression tests must never write into the real research corpus."""
import builtins
import io
import os
import inspect
from pathlib import Path
import pytest

REPO_ROOT=Path(__file__).resolve().parents[2]
PROTECTED=[REPO_ROOT/'local_data'/name for name in ('raw','datasets','canonical','annotations','intermediate','manifests','exports','reports','cache')]
PROTECTED.extend([REPO_ROOT/'audit_outputs',REPO_ROOT/'research_outputs/p0_20261103_01'])
PROTECTED.extend([REPO_ROOT/'research_outputs/p1_20261103_01',Path('E:/LUMAE/work/d1000/pilot_20261103_01'),Path('E:/LUMAE/cache/p1_20261103_01')])
for sealed_version in ('p14_p15_20261103_01','p14_p15_20261103_02'):
    PROTECTED.extend([REPO_ROOT/'research_outputs'/sealed_version,
                      Path('E:/LUMAE/work/d1000')/sealed_version,
                      Path('E:/LUMAE/work')/sealed_version])

def protected_path(value):
    if isinstance(value,int):return False
    try:p=Path(os.fsdecode(os.fspath(value))).resolve()
    except (TypeError,ValueError):return False
    return any(p.is_relative_to(root) for root in PROTECTED)

@pytest.fixture(autouse=True)
def forbid_historical_research_writes(monkeypatch):
    original_builtin=builtins.open;original_io=io.open;original_os=os.open
    def check(path,mode):
        if any(x in mode for x in 'wax+') and protected_path(path):
            raise AssertionError(f'Test attempted a protected historical write: {path}; use tmp_path/output_dir')
    def builtin_open(file,mode='r',*args,**kwargs):check(file,mode);return original_builtin(file,mode,*args,**kwargs)
    def io_open(file,mode='r',*args,**kwargs):check(file,mode);return original_io(file,mode,*args,**kwargs)
    def os_open(file,flags,*args,**kwargs):
        if flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND) and protected_path(file):
            raise AssertionError(f'Test attempted a protected historical write: {file}; use tmp_path/output_dir')
        return original_os(file,flags,*args,**kwargs)
    monkeypatch.setattr(builtins,'open',builtin_open);monkeypatch.setattr(io,'open',io_open);monkeypatch.setattr(os,'open',os_open)

@pytest.fixture
def isolated_benchmark_output(tmp_path,monkeypatch):
    """Reuse frozen benchmark source unmodified, against a small metadata copy.

    No videos/features/checkpoints are copied. This also preserves algorithm
    source hashes, unlike changing a frozen evaluator to add an output flag.
    """
    root=tmp_path/'research_fixture'
    for rel in ('local_data/annotations/lumae_ads','local_data/manifests','local_data/reports/lumae_ads','data_process/adsqa'):
        for source in (REPO_ROOT/rel).rglob('*'):
            if source.is_file() and source.suffix in ('.json','.csv','.md','.py') and '__pycache__' not in source.parts:
                dest=root/source.relative_to(REPO_ROOT);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(source.read_bytes())
    def execute(function):
        if 'repo_root' in inspect.signature(function).parameters:return function(repo_root=root)
        monkeypatch.setitem(function.__globals__,'REPO_ROOT',root)
        return function()
    return execute
