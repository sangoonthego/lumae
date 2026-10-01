"""Lazy singleton CLIP runtime provider for process-wide model reuse.

Eliminates redundant CLIP model loads across feature extraction,
boundary refinement, consistency scoring, and verification.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

from .models import ROOT

logger = logging.getLogger(__name__)

# Global registry of initialized runtimes: (model_name, device, download_root) -> ClipRuntime
_RUNTIMES: Dict[tuple, ClipRuntime] = {}


class ClipRuntime:
    """Encapsulates a loaded CLIP model, tokenizer, and preprocessor.

    Loaded once per process/device configuration and shared across stages.
    """

    def __init__(
        self,
        model_name: str = "ViT-B/32",
        device: Optional[str] = None,
        download_root: Optional[Union[Path, str]] = None,
    ) -> None:
        self.model_name = model_name
        self.download_root = Path(download_root or (ROOT / "local_data/cache/clip_weights"))
        self._model = None
        self._preprocess = None
        self._init_count = 0
        self._lock = None

        if device is None:
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def init_count(self) -> int:
        return self._init_count

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        import clip
        self.download_root.mkdir(parents=True, exist_ok=True)
        model, preprocess = clip.load(
            self.model_name,
            device=self.device,
            jit=False,
            download_root=str(self.download_root),
        )
        model.eval()
        self._model = model
        self._preprocess = preprocess
        self._init_count += 1
        logger.info(
            "Initialized ClipRuntime: model=%s device=%s init_count=%d",
            self.model_name,
            self.device,
            self._init_count,
        )

    @property
    def model(self) -> Any:
        self._ensure_loaded()
        return self._model

    @property
    def preprocess(self) -> Any:
        self._ensure_loaded()
        return self._preprocess

    def encode_text(
        self,
        queries: Union[str, Sequence[str]],
        normalize: bool = True,
    ) -> np.ndarray:
        """Encode text query string(s) into normalized float32 numpy vectors."""
        import clip
        import torch

        self._ensure_loaded()

        if isinstance(queries, str):
            query_list = [queries]
        else:
            query_list = list(queries)

        tokens = clip.tokenize(query_list).to(self.device)
        with torch.inference_mode():
            vector = self._model.encode_text(tokens).float()
            if normalize:
                vector /= vector.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        return vector.cpu().numpy()

    def encode_images(
        self,
        images_or_paths: Sequence[Any],
        batch_size: int = 32,
        normalize: bool = True,
    ) -> np.ndarray:
        """Encode PIL images or image paths into normalized float32 numpy vectors."""
        from PIL import Image
        import torch

        self._ensure_loaded()

        vectors: List[np.ndarray] = []
        num_images = len(images_or_paths)

        with torch.inference_mode():
            for start_idx in range(0, num_images, batch_size):
                batch_items = images_or_paths[start_idx : start_idx + batch_size]
                tensors = []
                for item in batch_items:
                    if isinstance(item, (str, Path)):
                        with Image.open(item) as img:
                            tensors.append(self._preprocess(img.convert("RGB")))
                    else:
                        tensors.append(self._preprocess(item.convert("RGB")))
                
                batch_tensor = torch.stack(tensors).to(self.device)
                encoded = self._model.encode_image(batch_tensor).float()
                if normalize:
                    encoded /= encoded.norm(dim=-1, keepdim=True).clamp(min=1e-12)
                vectors.append(encoded.cpu().numpy())

        if not vectors:
            return np.empty((0, 512), dtype=np.float32)
        return np.concatenate(vectors, axis=0)


def get_clip_runtime(
    model_name: str = "ViT-B/32",
    device: Optional[str] = None,
    download_root: Optional[Union[Path, str]] = None,
) -> ClipRuntime:
    """Return a singleton ClipRuntime instance for the specified configuration."""
    resolved_root = str(Path(download_root or (ROOT / "local_data/cache/clip_weights")).resolve())
    chosen_device = device or "cpu"
    key = (model_name, chosen_device, resolved_root)

    if key not in _RUNTIMES:
        _RUNTIMES[key] = ClipRuntime(
            model_name=model_name,
            device=chosen_device,
            download_root=resolved_root,
        )
    return _RUNTIMES[key]


def reset_clip_runtimes() -> None:
    """Reset all cached runtimes (used for testing)."""
    _RUNTIMES.clear()
