"""Optional pytest plugin: additionally protect the latest sealed P16/P2 state."""
import builtins
import io
import os
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[2]
PROTECTED=[ROOT/'research_outputs/p16_p2_20261103_01',Path('E:/LUMAE/work/d1000/p16_p2_20261103_01'),
    Path('E:/LUMAE/work/d1000/pilot_p2_20261103_01'),Path('E:/LUMAE/work/p1_p2_20261103_01'),Path('E:/LUMAE/cache/p1_p2_20261103_01')]

@pytest.fixture(autouse=True)
def p21_protect_sealed_previous_state(monkeypatch):
    def protected(value):
        if isinstance(value,int):return False
        try:path=Path(os.fsdecode(os.fspath(value))).resolve()
        except (TypeError,ValueError):return False
        return any(path.is_relative_to(p) for p in PROTECTED)
    old_builtin,old_io,old_os=builtins.open,io.open,os.open
    def checked(file,mode):
        if any(c in mode for c in 'wax+') and protected(file):raise AssertionError('P21 protected previous state write')
    def builtin(file,mode='r',*args,**kwargs):checked(file,mode);return old_builtin(file,mode,*args,**kwargs)
    def io_open(file,mode='r',*args,**kwargs):checked(file,mode);return old_io(file,mode,*args,**kwargs)
    def os_open(file,flags,*args,**kwargs):
        if flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND) and protected(file):raise AssertionError('P21 protected previous state write')
        return old_os(file,flags,*args,**kwargs)
    monkeypatch.setattr(builtins,'open',builtin);monkeypatch.setattr(io,'open',io_open);monkeypatch.setattr(os,'open',os_open)
