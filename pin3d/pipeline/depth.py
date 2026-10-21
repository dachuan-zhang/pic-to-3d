"""Monocular depth backends (ONNX Runtime — no torch, works on Apple Silicon CPU).

Depth Anything V2 Small, exported to ONNX, is the default: ~99 MB, no PyTorch, and the
only thing it needs is a (H, W, 3) uint8 image. It is an **affine-invariant** model, so its
output has to be aligned to metric scale before it means anything in metres — see
``pin3d.pipeline.real.affine_fit``.

The preprocessing mirrors the model's own ``preprocessor_config.json`` (resize so the long
side is 518, keep aspect, dims rounded to a multiple of 14, ImageNet normalisation).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

try:  # pragma: no cover - optional dependency
    from PIL import Image
    _HAVE_PIL = True
except Exception:  # pragma: no cover
    _HAVE_PIL = False

DEFAULT_MODEL = "models/depth_anything_v2_small.onnx"
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _resize(img: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Bilinear resize of an (H,W) or (H,W,3) array to (h,w)."""
    if not _HAVE_PIL:
        raise ImportError("Pillow is required for image resizing: pip install pillow")
    h, w = size
    if img.ndim == 2:
        return np.array(Image.fromarray(img).resize((w, h), Image.BILINEAR))
    return np.array(Image.fromarray(img).resize((w, h), Image.BILINEAR))


def target_size(h: int, w: int, long_side: int = 518, multiple_of: int = 14) -> Tuple[int, int]:
    scale = long_side / max(h, w)
    th = max(multiple_of, int(round(h * scale / multiple_of)) * multiple_of)
    tw = max(multiple_of, int(round(w * scale / multiple_of)) * multiple_of)
    return th, tw


@dataclass
class DepthModelInfo:
    path: str
    providers: List[str]
    long_side: int


class OnnxDepth:
    """Depth Anything V2 (ONNX). Output is affine-invariant depth, larger = farther."""

    def __init__(
        self,
        path: str = DEFAULT_MODEL,
        long_side: int = 518,
        providers: Optional[Sequence[str]] = None,
    ) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} not found. Fetch it with:\n"
                f"  bash scripts/fetch_models.sh\n"
                f"(or curl the file from huggingface.co/onnx-community/depth-anything-v2-small)"
            )
        import onnxruntime as ort

        available = ort.get_available_providers()
        chosen = [p for p in (providers or ["CoreMLExecutionProvider", "CPUExecutionProvider"])
                  if p in available] or ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(path, providers=chosen)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.long_side = long_side
        self.info = DepthModelInfo(path=path, providers=list(self.session.get_providers()),
                                   long_side=long_side)

    def predict(self, rgb: np.ndarray) -> np.ndarray:
        """(H, W, 3) uint8 -> (H, W) float32 depth, same size as the input."""
        img = np.asarray(rgb, dtype=np.uint8)
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        h, w = img.shape[:2]
        th, tw = target_size(h, w, self.long_side)

        resized = _resize(img, (th, tw)).astype(np.float32) / 255.0
        norm = (resized - IMAGENET_MEAN) / IMAGENET_STD
        chw = np.transpose(norm, (2, 0, 1))[None].astype(np.float32)

        out = self.session.run([self.output_name], {self.input_name: chw})[0]
        pred = np.asarray(out, dtype=np.float32).reshape(out.shape[-2], out.shape[-1])
        return _resize(pred, (h, w)).astype(np.float32)


class PassthroughDepth:
    """Wraps ground-truth depth so the real-data harness is model-agnostic."""

    def __init__(self, depth: np.ndarray) -> None:
        self.depth = np.asarray(depth, dtype=np.float32)
        self.info = DepthModelInfo(path="<ground truth>", providers=["none"], long_side=0)

    def predict(self, rgb: np.ndarray) -> np.ndarray:  # noqa: ARG002
        return self.depth
