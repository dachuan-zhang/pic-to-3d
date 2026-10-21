"""Synthetic capture simulation: render a room as a point cloud from one or more viewpoints.

Used to test room models under *controlled visibility* — the thing real data never gives you.
A bounding-box fit and a wall-plane fit differ only when the view is partial or the room is
not a rectangle, so both cases have to be constructible on demand.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .geometry import as_poly, polygon_centroid, wall_segments


@dataclass
class Capture:
    polygon: np.ndarray            # floorplan polygon (x, z)
    camera_xy: Tuple[float, float]
    azimuth_deg: float = 0.0
    hfov_deg: float = 68.0
    height: float = 2.6
    density: float = 70.0          # points per m^2 of wall
    noise: float = 0.012
    floor_ceiling: bool = True


def _in_fov(points_xz: np.ndarray, cam: np.ndarray, azimuth: float, hfov: float) -> np.ndarray:
    d = points_xz - cam[None, :]
    rng = np.linalg.norm(d, axis=1)
    ok = rng > 0.25
    az = np.arctan2(d[:, 1], d[:, 0])
    delta = np.arctan2(np.sin(az - azimuth), np.cos(az - azimuth))
    return ok & (np.abs(delta) <= math.radians(hfov) / 2.0)


def capture_cloud(cap: Capture, seed: int = 0) -> np.ndarray:
    """Sample the walls a camera can actually see, plus floor and ceiling.

    Walls outside the horizontal FOV are dropped: that is what makes a single view partial,
    and it is exactly the case where a bounding rectangle invents walls it never observed.
    """
    rng = np.random.default_rng(seed)
    cam = np.asarray(cap.camera_xy, dtype=float)
    az = math.radians(cap.azimuth_deg)

    pts: List[np.ndarray] = []
    for p0, p1 in wall_segments(cap.polygon):
        length = float(np.linalg.norm(p1 - p0))
        if length < 1e-6:
            continue
        n = max(8, int(length * cap.height * cap.density))
        along = rng.uniform(0.0, 1.0, n)
        vert = rng.uniform(0.0, cap.height, n)
        xz = p0[None, :] + along[:, None] * (p1 - p0)[None, :]
        keep = _in_fov(xz, cam, az, cap.hfov_deg)
        if keep.sum() == 0:
            continue
        pts.append(np.stack([xz[keep, 0], vert[keep], xz[keep, 1]], axis=1))

    if cap.floor_ceiling:
        for y in (0.0, cap.height):
            n = max(64, int(abs(_poly_area(cap.polygon)) * cap.density * 0.4))
            xs = rng.uniform(cap.polygon[:, 0].min(), cap.polygon[:, 0].max(), n * 4)
            zs = rng.uniform(cap.polygon[:, 1].min(), cap.polygon[:, 1].max(), n * 4)
            from .geometry import points_in_polygon
            inside = points_in_polygon(np.stack([xs, zs], axis=1), cap.polygon)
            take = min(n, int(inside.sum()))
            if take:
                pts.append(np.stack([xs[inside][:take], np.full(take, y), zs[inside][:take]], axis=1))

    if not pts:
        return np.zeros((0, 3))
    cloud = np.vstack(pts)
    return cloud + rng.normal(0.0, cap.noise, cloud.shape)


def _poly_area(poly: np.ndarray) -> float:
    p = as_poly(poly)
    x, y = p[:, 0], p[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def add_clutter(cloud: np.ndarray, polygon: np.ndarray, n_boxes: int = 2,
                seed: int = 1, size: Tuple[float, float] = (0.8, 1.4)) -> np.ndarray:
    """Add furniture at torso height: the thing that makes bounding boxes wrong."""
    rng = np.random.default_rng(seed)
    from .geometry import points_in_polygon

    lo = polygon.min(axis=0) + 0.6
    hi = polygon.max(axis=0) - 0.6
    extra = []
    for _ in range(n_boxes):
        c = rng.uniform(lo, hi)
        if not points_in_polygon(c[None, :], polygon)[0]:
            continue
        w = rng.uniform(*size) / 2
        d = rng.uniform(*size) / 2
        h = rng.uniform(0.8, 1.6)
        n = max(40, int(4 * (w + d) * h * 60))
        side = rng.integers(0, 4, n)
        u = rng.uniform(-1, 1, n)
        v = rng.uniform(0, h, n)
        x = np.where(side == 0, c[0] + w, np.where(side == 1, c[0] - w,
                    np.where(side == 2, c[0] + u * w, c[0] + u * w)))
        z = np.where(side == 2, c[1] + d, np.where(side == 3, c[1] - d,
                    np.where(side < 2, c[1] + u * d, c[1] + u * d)))
        extra.append(np.stack([x, v, z], axis=1))
    if not extra:
        return cloud
    return np.vstack([cloud] + extra)
