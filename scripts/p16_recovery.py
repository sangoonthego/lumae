"""New recovery evidence only. Never import the frozen, executable P14 scripts."""
import argparse
import csv
import math
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_process.lumae_scale.models import ROOT, atomic_json, digest, read_json

VERSION = 'p16_p2_20261103_01'
O = ROOT / 'research_outputs' / VERSION
R = Path('E:/LUMAE')
E = R / 'work/d1000' / VERSION
OLD = ROOT / 'research_outputs/p14_p15_20261103_02'
A = R / 'work/d1000/p14_p15_20261103_01'
B = A.with_name('p14_p15_20261103_02')

# Written after actual full-overview visual inspection in this recovery session.
# These events differ from the previous failed queries, not just their timestamps.
ALTERNATIVES = {
    1: ('A man and a woman sing facing one another while holding microphones.', 19.5, 22, 'man and woman', 'sing facing each other', 'microphones'),
    2: ('A child walks toward the camera along an indoor corridor.', 24, 27.5, 'child', 'walks toward camera', 'indoor corridor'),
    3: ('A car is pulled onto the back of a recovery truck.', 24, 27.5, 'car', 'is pulled onto', 'recovery truck'),
    4: ('An armored character swings a weapon during a fight in a forest.', 20, 24, 'armored character', 'swings', 'weapon in forest'),
    6: ('Three worker portraits appear together inside a yellow graphic border.', 12.5, 16, 'three worker portraits', 'appear together', 'yellow graphic border'),
    10: ('A person holds up a newspaper showing a football player.', 9.5, 13, 'person', 'holds up', 'newspaper showing football player'),
    11: ('A boy reaches into an open refrigerator to take a packet.', 30, 33.5, 'boy', 'reaches into', 'open refrigerator'),
    19: ('A person sits on a couch watching a television across the room.', 13.5, 21, 'person', 'sits watching', 'television across room'),
    23: ('A person applies lipstick to the lower lip in a close-up.', 33, 35.5, 'person', 'applies', 'lipstick to lower lip'),
    24: ('A child sleeps with the head tilted against a car seat.', 31, 33.5, 'child', 'sleeps', 'car seat'),
    25: ('A group of friends raises glasses together in a pub.', 25, 29.5, 'group of friends', 'raises', 'glasses in pub'),
    27: ('A paramedic carrying a bag walks down a hospital corridor.', 2.5, 5.5, 'paramedic', 'walks carrying', 'bag in hospital corridor'),
    28: ('An ambulance drives away from the camera along a dark street.', 21.5, 27.5, 'ambulance', 'drives away', 'dark street'),
}

# Explicit observed event intervals are AI visual estimates, not human GT.
# Raw model envelopes below remain unchanged even when excluded here.
PRIOR_REVIEWS = {
    5: (False, [11.5, 13], 'Repeated open-hand gestures; raw11-14 includes the next close-up and is not a uniquely bounded occurrence.'),
    7: (True, [100.5, 102.5], 'Unique goggles removal;99-103 contains surfacing context then the full movement with no unrelated shot.'),
    8: (True, [25.5, 30.072], 'Girl is lifted onto a back and remains riding through the clip end; endpoint is source truncation, not observed completion.'),
    9: (False, [43.5, 46.5], 'Static room description; raw42.5-47 also includes close seat view and a different mall shot.'),
    12: (False, [19, 23], 'Static grid description; raw18.5-23.5 begins with four tiles and ends in a single-person shot.'),
    13: (False, [2, 5.5], 'Static bird/bottle description; raw1-6 includes title card and a different close-up.'),
    14: (False, [21.5, 24], 'Holding a glass is a scene description; raw21.5-24.5 includes unrelated horse shot.'),
    15: (False, [12.5, 18], 'Stage action exists but raw11.5-18.5 includes outdoor rehearsal and car shot on opposite sides.'),
    16: (True, [0, 1.8], 'Crackers visibly pour into bowl from opening frame to cut near2s; no earlier source context exists.'),
    17: (False, [15, 17.5], 'Static stone surface description without an occurrence; raw15-18 crosses into purple stone.'),
    18: (True, [28.5, 31.5], 'Sequential rising from bed is a visible state change, fully covered by raw28-32.'),
    20: (False, [19.5, 25.5], 'Generic sustained operator scene, lacks a distinct event; raw19-26 overlaps adjacent graphics.'),
    21: (True, [50.5, 52], 'Booklet changes from closed brown cover to visible black-and-white pages;49.5-52 contains the opening.'),
    26: (True, [67, 70], 'Animation shows strap moving into attachment and assembled shield; raw67-71 contains the state change.'),
    29: (True, [38, 38.5], 'Background visibly changes white to furnished; raw37-39.5 includes clear before and after states.'),
    30: (False, [14.5, 17.5], 'Static bag display only; raw13.5-18 crosses from text to display and then a close-up.'),
}


def inventory():
    return list(csv.DictReader((OLD / 'pilot30_inventory.csv').open(encoding='utf-8-sig')))


def sheet(frames, target):
    from PIL import Image, ImageDraw
    canvas = Image.new('RGB', (1320, 150 * math.ceil(len(frames) / 6)))
    draw = ImageDraw.Draw(canvas)
    for j, frame in enumerate(frames):
        with Image.open(frame['path']) as im:
            im = im.convert('RGB'); im.thumbnail((218, 122))
            x, y = j % 6 * 220, j // 6 * 150
            canvas.paste(im, (x, y)); draw.text((x + 2, y + 125), str(round(frame['timestamp'], 3)), fill='white')
    canvas.save(target, quality=90)


def extract(video, times, folder):
    folder.mkdir(parents=True, exist_ok=True)
    frames = []
    for j, t in enumerate(times):
        path = folder / f'{j:04d}.jpg'
        if not path.exists():
            subprocess.run(['ffmpeg', '-v', 'error', '-ss', str(t), '-i', str(video), '-frames:v', '1', '-vf', 'scale=320:-2', '-q:v', '3', str(path)], check=True, capture_output=True, timeout=30)
        frames.append({'timestamp':float(t), 'path':str(path), 'sha256':digest(path)})
    return frames


def prepare():
    import numpy as np
    import torch
    from data_process.lumae_scale.clip_runtime import ClipRuntime
    from data_process.lumae_scale.visual_index import extract_clip_features
    from data_process.lumae_scale.temporal_labeler import localize
    if (O / 'pilot30_recovery_freeze_manifest.json').exists():
        raise PermissionError('Recovery frozen; use a new version')
    E.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    weights = ROOT / 'local_data/cache/clip_weights/ViT-B-32.pt'
    runtime = ClipRuntime(str(weights), device='cpu', download_root=E / 'model_cache')
    for n, values in ALTERNATIVES.items():
        r = inventory()[n-1]; sid = r['source_id']; query, start, end, subject, action, obj = values
        out = E / 'annotations' / sid; out.mkdir(parents=True, exist_ok=True)
        if (out / 'qc.json').exists():
            continue
        # Whole-video 2s context plus dense alternative-event evidence. No old event-only crop.
        times = sorted(set(np.arange(0, float(r['duration'])-.05, 2).tolist() + np.arange(max(0,start-2), min(float(r['duration'])-.05,end+2), .5).tolist()))
        cache = E / 'visual_index' / sid; frames = extract(r['path'], times, cache)
        overview = E / f'pilot30_{n:02d}_alternative.jpg'; sheet(frames, overview)
        metadata = {'video_id':sid, 'video_path':r['path'], 'video_sha256':r['sha256'], 'duration':float(r['duration']), 'frames':[{'timestamp':f['timestamp'], 'path':Path(f['path']).name, 'sha256':f['sha256']} for f in frames], 'contact_sheets':[str(overview)], 'clip_features_status':'pending', 'sampling_note':'Whole source2s plus exact-seek0.5s alternative-event context'}
        atomic_json(cache / 'metadata.json', metadata)
        extract_clip_features(sid, clip_runtime=runtime, visual_cache=E / 'visual_index')
        # Numerically blind to generator bounds; same CLIP model, not model independence.
        with np.load(cache / 'clip_features.npz') as z:
            t = float(z['timestamps'][int(np.argmax(z['features'] @ runtime.encode_text(query)[0]))])
        atomic_json(out / 'blind_first.json', {'query':query, 'interval':[max(0,t-2), min(float(r['duration']),t+2)], 'procedure':'Query-only full-source retrieval before localize; no phase/bounds input', 'same_clip_weights':True, 'weights_sha256':digest(weights), 'human_verified':False})
        event = {'query':query, 'coarse_start':start, 'coarse_end':end, 'evidence_timestamps':[t for t in times if start <= t < end], 'visible_subject':subject, 'visible_action':action, 'visible_object':obj}
        atomic_json(out / 'generation_input.json', event)
        pred = localize(sid, query, [event], float(r['duration']), 0, clip_runtime=runtime, visual_cache=E / 'visual_index')
        atomic_json(out / 'proposal.json', {'query':query, 'prediction':pred, 'event':event})
        qc = extract(r['path'], np.arange(max(0,pred['start']-1), min(float(r['duration'])-.05,pred['end']+1), .5), out / 'qc')
        sheet(qc, E / f'pilot30_{n:02d}_alternative_qc.jpg'); atomic_json(out / 'qc.json', qc)
        print(n, pred['start'], pred['end'], flush=True)


if __name__ == '__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    prepare()
