"""Tiny image helpers for local inspection (no matplotlib)."""
from __future__ import annotations

import os
from typing import Optional

import numpy as np

try:
    from PIL import Image
    _HAVE_PIL = True
except Exception:  # pragma: no cover
    _HAVE_PIL = False

#: perceptual-ish colormap: near = warm/bright, far = cool/dark
_STOPS = np.array([
    [255, 245, 200], [255, 200, 90], [250, 130, 60], [200, 60, 90],
    [110, 50, 140], [40, 40, 90], [15, 15, 30],
], dtype=float)


def colorize_depth(depth: np.ndarray, lo: Optional[float] = None, hi: Optional[float] = None) -> np.ndarray:
    """(H,W) depth -> (H,W,3) uint8. Invalid (<=0 / non-finite) become black."""
    d = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(d) & (d > 0)
    if not valid.any():
        return np.zeros((*d.shape, 3), dtype=np.uint8)
    if lo is None:
        lo = float(np.percentile(d[valid], 2))
    if hi is None:
        hi = float(np.percentile(d[valid], 98))
    t = np.clip((d - lo) / max(hi - lo, 1e-6), 0, 1)
    pos = t * (len(_STOPS) - 1)
    i = np.clip(np.floor(pos).astype(int), 0, len(_STOPS) - 2)
    frac = (pos - i)[..., None]
    rgb = _STOPS[i] * (1 - frac) + _STOPS[i + 1] * frac
    out = rgb.astype(np.uint8)
    out[~valid] = 0
    return out


def save_rgb(path: str, rgb: np.ndarray) -> str:
    if not _HAVE_PIL:
        raise ImportError("Pillow required")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    Image.fromarray(np.asarray(rgb, dtype=np.uint8)).save(path)
    return path


def save_depth_png(path: str, depth: np.ndarray, lo: Optional[float] = None,
                   hi: Optional[float] = None) -> str:
    return save_rgb(path, colorize_depth(depth, lo, hi))
