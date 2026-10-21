"""Point cloud -> Manhattan room box / wall segments.

Convention: 3D points are ``(N, 3)`` with **+Y up**, metres, right-handed (X/Z horizontal).
Everything here is deterministic and dependency-light so it can be unit tested without a
model; see ``docs/BRAINSTORM.md`` §5 (Level 2 = room box) and §9 (risks).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..geometry import (
    as_poly,
    dot_rows,
    ensure_ccw,
    min_area_rect,
    polygon_area,
    rectify_polygon,
    rot_points,
    snap_angle,
    wall_segments,
)

UP_AXIS = 1


@dataclass
class RoomBox:
    """Level-2 reconstructed room: an oriented Manhattan rectangle."""

    polygon: np.ndarray  # (4,2) in floorplan coords (X,Z)
    angle: float  # radians, normalised to [0, pi/2)
    ceiling: float  # metres
    floor_y: float  # world Y of the floor plane
    confidence: float  # 0..1, wall-support fraction
    n_points: int
    method: str = "box_fit"
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def area(self) -> float:
        return polygon_area(self.polygon)

    @property
    def size(self) -> Tuple[float, float]:
        p = as_poly(self.polygon)
        return float(np.ptp(p[:, 0])), float(np.ptp(p[:, 1]))


def distance_to_polygon_boundary(points: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Per-point distance to the polygon's edges."""
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    if len(pts) == 0 or len(poly) < 2:
        return np.full(len(pts), np.inf)
    best = np.full(len(pts), np.inf)
    for a, b in wall_segments(poly):
        ab = b - a
        denom = float(ab @ ab)
        if denom < 1e-12:
            d = np.linalg.norm(pts - a, axis=1)
        else:
            t = np.clip(dot_rows(pts - a, ab) / denom, 0.0, 1.0)
            proj = a[None, :] + t[:, None] * ab[None, :]
            d = np.linalg.norm(pts - proj, axis=1)
        best = np.minimum(best, d)
    return best


def estimate_floor_ceiling(
    points: np.ndarray, frac: float = 0.10
) -> Tuple[float, Optional[float]]:
    """Robust floor Y, and ceiling Y if the cloud actually reaches one.

    Same outer-window-median reasoning as the wall planes: the floor and ceiling are spikes
    at the extremes of the Y distribution, so their window medians are unbiased, while a
    min/max would be inflated by ~2 sigma of depth noise.
    """
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    if len(pts) == 0:
        return 0.0, None
    floor, top = wall_plane_bounds(pts[:, UP_AXIS], frac)
    ceiling = top if (top - floor) > 1.5 else None
    return floor, ceiling


def band_points(
    points: np.ndarray, lo: float = 0.8, hi: float = 1.6
) -> np.ndarray:
    """Horizontal slice at torso height: avoids furniture, floor and ceiling clutter."""
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    y = pts[:, UP_AXIS]
    return pts[(y >= lo) & (y <= hi)]


def density_filter(
    points2d: np.ndarray, cell: float = 0.25, min_count: int = 3
) -> np.ndarray:
    """Drop isolated points (outlier clutter) by grid occupancy.

    Min-area rectangles are not robust to even a fraction of a percent of outliers, and every
    real point cloud has some. A 25 cm cell with >=3 points keeps walls (dense) and discards
    flyers (sparse) in O(N log N), without a radius search.
    """
    pts = np.asarray(points2d, dtype=float).reshape(-1, 2)
    if len(pts) == 0:
        return pts
    keys = np.floor(pts / cell).astype(np.int64)
    kx = keys[:, 0] - keys[:, 0].min()
    kz = keys[:, 1] - keys[:, 1].min()
    code = kx * int(kz.max() + 1) + kz
    _, inv, counts = np.unique(code, return_inverse=True, return_counts=True)
    return pts[counts[inv] >= min_count]


def _wall_plane(values: np.ndarray, side: str, frac: float = 0.15) -> float:
    """Estimate one wall plane from the outer slice of a coordinate distribution.

    Why not min/max: noise biases the extreme outward by ~2 sigma. Why not percentiles: the
    coordinate distribution is a *spike at each wall plus a uniform interior* (walls on the
    perpendicular axis span the whole range), so a percentile trim removes the wrong thing.
    In the outer ``frac`` slice the wall spike dominates by point count, so its median is an
    unbiased wall-plane estimate. Assumes the input is already outlier-filtered.
    """
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        return 0.0
    lo, hi = float(v.min()), float(v.max())
    span = hi - lo
    if span <= 1e-9:
        return lo
    window = frac * span
    sel = v[v <= lo + window] if side == "lo" else v[v >= hi - window]
    if len(sel) == 0:
        return lo if side == "lo" else hi
    return float(np.median(sel))


def wall_plane_bounds(values: np.ndarray, frac: float = 0.15) -> Tuple[float, float]:
    return _wall_plane(values, "lo", frac), _wall_plane(values, "hi", frac)


def fit_room_box(
    points: np.ndarray,
    band: Tuple[float, float] = (0.8, 1.6),
    max_wall_dist: float = 0.20,
    snap: bool = True,
    apply_floor: bool = True,
    min_points: int = 40,
    cell: float = 0.25,
    cell_min_count: int = 3,
) -> Optional[RoomBox]:
    """Fit a Manhattan rectangle to the wall support seen in a point cloud.

    Strategy (deliberate, see §12 of the design doc): take a torso-height slice so
    furniture does not bias the fit, compute the minimum-area rectangle, snap its angle to
    the Manhattan lattice, then re-fit the extents in the snapped frame. Confidence is the
    fraction of band points that actually sit near the rectangle boundary -- a room whose
    walls were only partially seen reports low confidence instead of lying.
    """
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    if len(pts) < min_points:
        return None

    floor_y, ceiling_y = estimate_floor_ceiling(pts)
    offset = floor_y if apply_floor else 0.0
    shifted = pts.copy()
    shifted[:, UP_AXIS] -= offset

    band_pts = band_points(shifted, *band)
    if len(band_pts) < min_points:
        band_pts = shifted[shifted[:, UP_AXIS] > 0.2]
    if len(band_pts) < min_points:
        return None

    pts2 = band_pts[:, [0, 2]]
    dense = density_filter(pts2, cell=cell, min_count=cell_min_count)
    if len(dense) < max(min_points, 20):
        # sparse or uniform cloud: filtering would remove the room itself
        dense = pts2
    fit_points = dense if len(dense) >= 20 else pts2

    if snap:
        _, raw_angle = min_area_rect(fit_points)
        angle = snap_angle(raw_angle)
        # Re-fit extents in the snapped frame so snapping does not shrink the room.
        c, s = math.cos(-angle), math.sin(-angle)
        rot = rot_points(fit_points, np.array([[c, -s], [s, c]]))
        x_lo, x_hi = wall_plane_bounds(rot[:, 0])
        z_lo, z_hi = wall_plane_bounds(rot[:, 1])
        box2 = np.array([
            [x_lo, z_lo], [x_hi, z_lo], [x_hi, z_hi], [x_lo, z_hi],
        ])
        c, s = math.cos(angle), math.sin(angle)
        poly = rot_points(box2, np.array([[c, -s], [s, c]]))
    else:
        poly, angle = min_area_rect(fit_points)

    poly = ensure_ccw(poly)
    if snap:
        poly = rectify_polygon(poly, tol=0.02)

    d = distance_to_polygon_boundary(pts2, poly)
    confidence = float(np.mean(d <= max_wall_dist)) if len(d) else 0.0

    ceiling = (ceiling_y - offset) if ceiling_y is not None else 2.7

    return RoomBox(
        polygon=poly,
        angle=float(angle),
        ceiling=float(np.clip(ceiling, 1.5, 6.0)),
        floor_y=float(offset),
        confidence=confidence,
        n_points=int(len(pts)),
        extra={
            "band_points": int(len(band_pts)),
            "dense_points": int(len(dense)),
            "wall_dist_mean": float(np.mean(np.minimum(d, 5.0))),
        },
    )


# --------------------------------------------------------------------------
# Wall segments (general case: non-rectangular rooms, opening refinement)
# --------------------------------------------------------------------------
@dataclass
class WallSegment:
    p0: np.ndarray  # (2,) in floorplan coords
    p1: np.ndarray
    inliers: int
    normal: np.ndarray
    y_min: float
    y_max: float

    @property
    def length(self) -> float:
        return float(np.linalg.norm(self.p1 - self.p0))

    @property
    def midpoint(self) -> np.ndarray:
        return 0.5 * (self.p0 + self.p1)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "p0": [round(float(v), 3) for v in self.p0],
            "p1": [round(float(v), 3) for v in self.p1],
            "length": round(self.length, 3),
            "inliers": int(self.inliers),
            "y_range": [round(self.y_min, 3), round(self.y_max, 3)],
        }


def detect_wall_segments(
    points: np.ndarray,
    band: Tuple[float, float] = (0.5, 1.8),
    eps: float = 0.04,
    min_inliers: int = 40,
    iterations: int = 400,
    max_walls: int = 12,
    min_length: float = 0.5,
    seed: int = 0,
) -> List[WallSegment]:
    """RANSAC 2D line extraction on a torso-height slice, iteratively removing inliers.

    v0 scope: produces wall segments only. Turning a segment soup into a closed polygon
    (corner pairing, loop closure) is deliberately not attempted here -- see §9 item 4.
    """
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    if len(pts) < min_inliers:
        return []
    band_pts = band_points(pts, *band)
    if len(band_pts) < min_inliers:
        return []

    rng = np.random.default_rng(seed)
    remaining = band_pts[:, [0, 2]].copy()
    remaining_y = band_pts[:, UP_AXIS].copy()
    segments: List[WallSegment] = []

    for _ in range(max_walls):
        n = len(remaining)
        if n < min_inliers:
            break
        best_mask: Optional[np.ndarray] = None
        best_count = 0
        for _ in range(iterations):
            i, j = rng.integers(0, n, size=2)
            if i == j:
                continue
            a, b = remaining[i], remaining[j]
            d = b - a
            norm = float(np.linalg.norm(d))
            if norm < 1e-6:
                continue
            normal = np.array([-d[1], d[0]]) / norm
            mask = np.abs(dot_rows(remaining - a, normal)) <= eps
            count = int(np.count_nonzero(mask))
            if count > best_count:
                best_count, best_mask = count, mask
        if best_mask is None or best_count < min_inliers:
            break

        inl = remaining[best_mask]
        mean = inl.mean(axis=0)
        # PCA refine
        _, _, vt = np.linalg.svd(inl - mean, full_matrices=False)
        direction = vt[0]
        normal = np.array([-direction[1], direction[0]])
        t = dot_rows(inl - mean, direction)
        p0 = mean + t.min() * direction
        p1 = mean + t.max() * direction
        if float(np.linalg.norm(p1 - p0)) < min_length:
            remaining = remaining[~best_mask]
            remaining_y = remaining_y[~best_mask]
            continue

        ys = remaining_y[best_mask]
        segments.append(WallSegment(
            p0=p0, p1=p1, inliers=best_count, normal=normal,
            y_min=float(ys.min()), y_max=float(ys.max()),
        ))
        remaining = remaining[~best_mask]
        remaining_y = remaining_y[~best_mask]

    segments.sort(key=lambda s: -s.inliers)
    return segments
