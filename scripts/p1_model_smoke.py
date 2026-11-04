"""Local CPU-only CLIP dependency smoke; no label or human-verification claim."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
import time
from data_process.lumae_scale.models import ROOT,atomic_json,digest
from data_process.annotation.assisted_deployment import verify_frozen_ranker,ALGORITHM_SHA256

ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();start=time.monotonic()
result={'device':'cpu','gpu_experiment':False,'semantic_v3_algorithm_sha256':ALGORITHM_SHA256,'scope':'Dependency smoke on historical image; not a new pilot annotation or acceptance'}
try:
    verify_frozen_ranker(ROOT)
    import torch,clip
    from PIL import Image
    torch.set_num_threads(4)
    weights=ROOT/'local_data/cache/clip_weights/ViT-B-32.pt'
    model,preprocess=clip.load(str(weights),device='cpu',jit=False,download_root=str(a.output.parent/'model_cache'))
    image=next((ROOT/'local_data/cache/lumae_visual_index').glob('*/sampled_frames/frame_0001.jpg'))
    with Image.open(image) as frame:inputs=preprocess(frame).unsqueeze(0)
    with torch.no_grad():
        features=model.encode_image(inputs);text=model.encode_text(clip.tokenize(['A person holds a product.']))
    result.update(status='PASS',weights_sha256=digest(weights),image_sha256=digest(image),image_features=list(features.shape),text_features=list(text.shape),finite=bool(torch.isfinite(features).all() and torch.isfinite(text).all()),torch_version=torch.__version__,cuda_available=torch.cuda.is_available(),semantic_ranker_frozen=True)
except Exception as e:result.update(status='BLOCKED',error_type=type(e).__name__,reason=str(e)[:300])
result['elapsed_seconds']=time.monotonic()-start
atomic_json(a.output,result);print(result)
