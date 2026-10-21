"""Multi-view: SfM poses, metric scale recovery, and cloud fusion.

Why this exists
---------------
Our single-view measurements (§15b of the design doc) show a room cannot be recovered from one
frame: 29% area error, ceiling unobservable in 150/150 scenes, no doors. The claim is that
multiple overlapping views fix this. This module is the apparatus to test that claim rather
than assert it.

The awkward part is scale. SfM is scale-free: COLMAP returns camera centres in unknown units,
while each depth map is already metric. So a fused cloud is

    P_world = R_i @ P_cam(i) + s * t_i

with one unknown scalar ``s`` for the whole reconstruction. ``estimate_scale`` solves for it by
requiring overlapping clouds to agree — no ground truth needed, and no assumption that the
depth model is metrically perfect.
"""
from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class SfMResult:
    ok: bool
    registered: List[str] = field(default_factory=list)
    rotations: Dict[str, np.ndarray] = field(default_factory=dict)   # world->cam
    translations: Dict[str, np.ndarray] = field(default_factory=dict)  # SfM units
    intrinsics: Optional[np.ndarray] = None
    reason: str = ""
    points: int = 0

    @property
    def n(self) -> int:
        return len(self.registered)


def run_colmap(
    images: Sequence[Tuple[str, np.ndarray]],
    workdir: Optional[str] = None,
    camera_model: str = "PINHOLE",
) -> SfMResult:
    """Exhaustive-matching incremental SfM on a handful of frames (CPU).

    Returns whatever registered, plus the reason it did not when it fails: on sparse
    wide-baseline captures, failure is a result worth recording, not an exception.
    """
    try:
        import pycolmap
    except Exception as exc:  # pragma: no cover
        return SfMResult(False, reason=f"pycolmap unavailable: {exc}")

    from PIL import Image

    tmp = workdir or tempfile.mkdtemp(prefix="pin3d-sfm-")
    own_tmp = workdir is None
    img_dir = os.path.join(tmp, "images")
    db = os.path.join(tmp, "db.sqlite")
    out = os.path.join(tmp, "sparse")
    try:
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(out, exist_ok=True)
        for name, rgb in images:
            Image.fromarray(np.asarray(rgb, dtype=np.uint8)).save(os.path.join(img_dir, name))
        if os.path.exists(db):
            os.remove(db)

        pycolmap.extract_features(db, img_dir)
        pycolmap.match_exhaustive(db)
        recs = pycolmap.incremental_mapping(db, img_dir, out)
        if not recs:
            return SfMResult(False, reason="no reconstruction")
        rec = max(recs.values(), key=lambda r: r.num_reg_images())
        if rec.num_reg_images() < 2:
            return SfMResult(
                False,
                registered=[rec.images[i].name for i in rec.images],
                reason=f"only {rec.num_reg_images()} image(s) registered",
            )

        rotations: Dict[str, np.ndarray] = {}
        translations: Dict[str, np.ndarray] = {}
        intrinsics = None
        for image in rec.images.values():
            # pycolmap 3.13: cam_from_world is a method, not a property
            pose = image.cam_from_world()
            rotations[image.name] = np.asarray(pose.rotation.matrix(), dtype=float)
            translations[image.name] = np.asarray(pose.translation, dtype=float)
            cam = rec.cameras[image.camera_id]
            k = cam.calibration_matrix
            intrinsics = np.asarray(k() if callable(k) else k, dtype=float)

        return SfMResult(
            True,
            registered=sorted(rotations),
            rotations=rotations,
            translations=translations,
            intrinsics=intrinsics,
            points=int(rec.num_points3D()),
        )
    except Exception as exc:  # pragma: no cover - COLMAP raises in many ways
        return SfMResult(False, reason=f"{type(exc).__name__}: {exc}")
    finally:
        if own_tmp:
            shutil.rmtree(tmp, ignore_errors=True)


def _subsample(points: np.ndarray, n: int, seed: int = 0) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if len(pts) <= n:
        return pts
    idx = np.random.default_rng(seed).choice(len(pts), n, replace=False)
    return pts[idx]


def _mean_nn_distance(a: np.ndarray, b: np.ndarray, n: int = 900) -> float:
    """Symmetric mean nearest-neighbour distance between two clouds (subsampled)."""
    a = _subsample(a, n, 1)
    b = _subsample(b, n, 2)
    if not len(a) or not len(b):
        return float("inf")
    total = 0.0
    for src, dst in ((a, b), (b, a)):
        d = np.linalg.norm(src[:, None, :] - dst[None, :, :], axis=2)
        total += float(d.min(axis=1).mean())
    return total / 2.0


def fuse(
    clouds: Dict[str, np.ndarray],
    rotations: Dict[str, np.ndarray],
    translations: Dict[str, np.ndarray],
    scale: float,
) -> np.ndarray:
    """Camera-frame clouds + poses -> one world cloud. ``scale`` converts SfM units to metres."""
    out = []
    for name, pts in clouds.items():
        if name not in rotations:
            continue
        r = rotations[name]
        t = translations[name] * scale
        # einsum, not @: numpy 2 + Accelerate BLAS emits spurious FP warnings on
        # (N,3) @ (3,3) even for finite input (same quirk noted in geometry.py)
        out.append(np.einsum("ij,kj->ik", np.asarray(pts, dtype=float), r) + t)
    return np.vstack(out) if out else np.zeros((0, 3))


def estimate_scale(
    clouds: Dict[str, np.ndarray],
    rotations: Dict[str, np.ndarray],
    translations: Dict[str, np.ndarray],
    candidates: Optional[np.ndarray] = None,
) -> Tuple[float, float]:
    """Pick the scale at which independently-unprojected clouds agree best.

    The search is 1-D and coarse-to-fine. The objective (mean nearest-neighbour distance
    between overlapping clouds) needs no ground truth; it only assumes the views overlap,
    which is the thing under test.
    """
    names = [n for n in clouds if n in rotations]
    if len(names) < 2:
        return 1.0, float("inf")

    if candidates is None:
        candidates = np.geomspace(1e-2, 1e3, 48)

    def cost(s: float) -> float:
        total, pairs = 0.0, 0
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a = fuse({names[i]: clouds[names[i]]}, rotations, translations, s)
                b = fuse({names[j]: clouds[names[j]]}, rotations, translations, s)
                total += _mean_nn_distance(a, b, n=500)
                pairs += 1
        return total / max(pairs, 1)

    best_s, best_c = None, float("inf")
    for s in candidates:
        c = cost(float(s))
        if c < best_c:
            best_s, best_c = float(s), c
    if best_s is None:
        return 1.0, float("inf")

    # refine around the winner
    for _ in range(2):
        lo, hi = best_s / 1.6, best_s * 1.6
        for s in np.geomspace(lo, hi, 24):
            c = cost(float(s))
            if c < best_c:
                best_s, best_c = float(s), c
    return best_s, best_c


def coverage_confidence(cloud: np.ndarray, polygon: np.ndarray, floor_y: float,
                        max_dist: float = 0.20) -> float:
    """Fraction of torso-band points lying on the fitted box boundary."""
    from .planes import band_points, distance_to_polygon_boundary

    shifted = np.asarray(cloud, dtype=float).copy()
    shifted[:, 1] -= floor_y
    band = band_points(shifted, 0.8, 1.6)
    if len(band) < 20:
        return 0.0
    d = distance_to_polygon_boundary(band[:, [0, 2]], polygon)
    return float(np.mean(d <= max_dist))
