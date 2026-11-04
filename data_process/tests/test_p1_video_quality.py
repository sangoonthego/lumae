"""Real ffmpeg fixtures; never read or write historical video/cache outputs."""
import subprocess
from data_process.lumae_scale.pilot_quality import validate_video
from data_process.lumae_scale.visual_index import build_visual_index
from data_process.lumae_scale.models import digest

def test_real_video_decode_and_isolated_visual_index(tmp_path):
    source=tmp_path/'source.mp4'
    subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','testsrc2=duration=6:size=160x90:rate=10','-c:v','libx264','-pix_fmt','yuv420p',str(source)],check=True,timeout=30)
    info=validate_video(source)
    assert info['pass'] and info['sample_frames_decoded']==3 and info['initial_timestamps_consistent']
    out=tmp_path/'cache';work=tmp_path/'work'
    index=build_visual_index(source,6,digest(source),work=work,strict=True,visual_cache=out,root=tmp_path)
    assert index['video_path']=='source.mp4' and len(index['frames'])==3
    assert all((out/'source'/s).is_file() for s in index['contact_sheets'])
    before=(out/'source/metadata.json').read_bytes()
    assert build_visual_index(source,6,digest(source),work=work,strict=True,visual_cache=out,root=tmp_path)==index
    assert (out/'source/metadata.json').read_bytes()==before

def test_html_named_mp4_is_rejected(tmp_path):
    source=tmp_path/'invalid.mp4';source.write_text('<html>Forbidden</html>')
    assert validate_video(source)['pass'] is False
