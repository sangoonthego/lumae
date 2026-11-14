# D1500 Test-450 initial audit

## Verified before implementation

- Repository `D:/DUT/Se07/CV/lumae`, clean at `f2c532decbe47448ad28ee3c82881f412893c469` on `feat/ui_v1`. Research Python lives in `data_process`, `scripts` and `experiments`. The frontend has a separate Node test suite. No frontend changes are required.
- `research_outputs/d1500_split_freeze_20261110_01` contains the frozen 20261110 split. All eight entries of `SHA256SUMS.txt` match. The canonical dataset at `research_outputs/d1500_agent_20261106_01/lumae_ads_d1500/lumae_ads_d1500_all.jsonl` matches `f386b4e3e3a717dbd151f67abdce71e6366123c760753b7687f766ade7b38e6a`.
- The 1,200/150/150 partitions match the sealed VID-to-QID assignments. The combined split has 1,500 unique VIDs and QIDs, with one valid interval per query. The Test subset manifests contain 30 legacy and 120 additional D1500 originals. All 150 Test sources have a local path and matching annotation SHA-256 and returned a video stream in FFprobe. Forty-eight Train/Val annotation rows lack an inline source hash. The frozen integrity report records the historical source-hash audit separately.
- All 3,000 NPZ files referenced in `research_outputs/d1500_colab_audit_20261110_01/d1500_colab_upload_manifest.json` opened without pickle and matched the audited SHA-256, keys and finite sequence shapes. There are 1,500 video files (`features[T,512]`, one float16) and 1,500 text files (`last_hidden_state[L,512]`, all float32). Cache roots are recorded per artifact across the E-drive historical, D1000 and D1500 stores. No cache contains TEF. Original NPZ hashes are available in the sealed input inventory rather than copied into this report.
- M0 at `research_outputs/p0_20261103_01/assets/M0_pretrained.ckpt` matched `ad5facec03008800536466c7d33f9232c3db2b71a1bd445073957bd6f4d0edbf`. Strict CPU load against the pinned upstream snapshot in that run loaded 92 tensors. Projection matrices are `(256,514)` for video and `(256,512)` for text. Span supervision is L1 center/width and saliency loss is zero in the cached adaptation runner. The historical report references E0 and D200 weights on Drive. No fine-tuned checkpoint was found in the inspected local research/cache trees.
- `experiments/M1_lumae_ads/cached_runner.py` iterates annotation rows and resolves video by VID and text by QID. Repeated VID is already supported. Its `preflight` function accepts an explicit recipe. The historical CLI defaults to the pinned D500 recipe. The pinned `standalone_eval/eval.py` is the authoritative MR evaluator. Keep the legacy runner and schema intact.
- `scripts/d1000_agent_run.py` and `scripts/d1500_agent_run.py` implement the earlier visual-evidence and agent review workflow. The D1500 text extractor under `E:/LUMAE/work/d1500/d1500_agent_20261106_01/feature_extractor_source` uses pinned Moment-DETR code, CLIP ViT-B/32, context length 77 and a 32-token cap including boundary tokens. It stores per-token hidden states. `local_data/cache/clip_weights/ViT-B-32.pt` matches `40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af`.

| Frozen split | SHA-256 |
|---|---|
| Train | `dfae4bb52d9f0366ae08eef1e386e441941946b84094726a011ef9c0d3eb698f` |
| Validation | `de937c59a67d548d9b3f2a2819072ff5af84acd1a70acda014033d6628723f0f` |
| Test | `50a309a839c998b746042afe28f48a573aa82f4cf8d2082dda3e5ba334d00b6c` |
| Legacy Test-30 | `19a3ef35674f79777153e0b6f39fe282e848c556ce52f17d3c73e3889967c1ea` |
| Other Test-120 | `2b391c568a02006e9c517d35f83d9604defd3be6e6d336328e6fe8de298f2482` |

## Inferences, assumptions and risks

- Drive confirmed candidate names and sizes for QR-DETR `videoonly.ckpt` (91,131,233 bytes) and UniVTG `model_best.ckpt` (151,708,747 bytes) and `opt.json` (3,052 bytes). Local strict load, tensors and candidate identity remain unverified at this checkpoint in the audit.
- The official QD-DETR video recipe uses CLIP and SlowFast, TEF and saliency supervision. Its MR outputs can be converted to ranked seconds. The intended QR-DETR candidate's actual feature dimensions and training labels need validation. [QD-DETR upstream](https://github.com/wjun0830/QD-DETR)
- UniVTG publishes CLIP-only and SlowFast+CLIP variants. The referenced QVHighlights recipe uses 2,816 visual channels before TEF. Temporal labels can provide weak saliency under an explicit training-only policy where supported. Candidate config, text inputs, span decoding and exact required features must be checked before creating runnable commands. [UniVTG model zoo](https://github.com/showlab/UniVTG/blob/main/model.md) · [UniVTG QVHighlights recipe](https://github.com/showlab/UniVTG/blob/main/scripts/qvhl_pretrain.sh)
- The local PyTorch runtime reports CUDA unavailable. The host detects an MX330 with 2 GB VRAM. GPU execution and reproducibility on the target runtime remain unverified.
- The historical report classifies campaign membership as unknown and records 203 unverified near-duplicate edges crossing split boundaries. Test-30 has prior evaluation exposure. All frozen Test and Validation labels are AI pseudo-labels. Query-level expansion cannot establish campaign independence or human agreement.
- Two independent visual estimates are required for each added query. Use a configurable 0.70 tIoU threshold and additional evidence and uniqueness gates. Sufficient distinct localized events across all 150 videos remain to be verified visually.

## Planned changes

- Add a D1500-specific workflow under `data_process/test450`, with audit and parent bindings, timestamped evidence and blind review intake, publication and additional text extraction, training orchestration and shared evaluation. Add focused regression tests under `data_process/tests` and a runbook and final report under `docs/reproducibility`.
- Use `research_outputs/d1500_test450_ai_v1` for incremental evidence, per-video state, cache, manifests and validation reports. Preserve the parent Test JSONL bytes in the expanded release. Use sidecar provenance for new rows, and distinct feature bindings for the training and evaluation entry points. Stage training bindings over Train/Val only and use the existing cached runner with an explicit D1500 recipe.
- Version release and run outputs independently. Keep protected dataset, feature, checkpoint and historical result directories read-only during this task. Recheck the pinned split, canonical and feature inventory after implementation.
