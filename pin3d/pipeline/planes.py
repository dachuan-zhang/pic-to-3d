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
    convex_hull,
    dot_rows,
    ensure_ccw,
    min_area_rect,
    polygon_area,
    polygon_from_halfplanes,
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
    floor_y: Optional[float] = None,
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

    if floor_y is not None:
        # A known floor (e.g. metric scale recovered from a camera-height prior) is far more
        # reliable than estimating it, and it pins the torso band where it belongs.
        est_floor, ceiling_y = estimate_floor_ceiling(pts)
        offset = float(floor_y)
        ceiling_y = (ceiling_y - est_floor + offset) if ceiling_y is not None else None
    else:
        est_floor, ceiling_y = estimate_floor_ceiling(pts)
        offset = est_floor if apply_floor else 0.0
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


# --------------------------------------------------------------------------
# wall-plane reconstruction (replaces the bounding-box room model)
# --------------------------------------------------------------------------
@dataclass
class WallPlane:
    """A detected vertical wall, oriented so the room lies on the ``n . x <= d`` side."""

    normal: np.ndarray      # (2,) inward unit normal, floorplan coords
    offset: float           # n . x = offset
    midpoint: np.ndarray    # a point on the plane
    support: int            # inlier points
    length: float           # extent of the observed run
    y_min: float
    y_max: float

    @property
    def angle_deg(self) -> float:
        return float(np.degrees(np.arctan2(self.normal[1], self.normal[0])))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "angle_deg": round(self.angle_deg, 2),
            "offset": round(float(self.offset), 3),
            "midpoint": [round(float(v), 3) for v in self.midpoint],
            "support": int(self.support),
            "length": round(float(self.length), 2),
            "y_range": [round(float(self.y_min), 2), round(float(self.y_max), 2)],
        }


@dataclass
class RoomPolygon:
    polygon: np.ndarray
    floor_y: float
    ceiling: float
    planes: List[WallPlane]
    binding: List[int]
    bounded: bool
    support: float
    n_points: int
    method: str = "wall_planes"

    @property
    def area(self) -> float:
        return polygon_area(self.polygon)

    @property
    def size(self) -> Tuple[float, float]:
        p = as_poly(self.polygon)
        return float(np.ptp(p[:, 0])), float(np.ptp(p[:, 1]))


def planes_from_segments(
    segments: Sequence[WallSegment], camera_xy: Sequence[float] = (0.0, 0.0)
) -> List[WallPlane]:
    """Turn 2D wall segments into inward-oriented half-planes."""
    c = np.asarray(camera_xy, dtype=float)
    out: List[WallPlane] = []
    for seg in segments:
        n = np.asarray(seg.normal, dtype=float)
        norm = float(np.linalg.norm(n))
        if norm < 1e-9:
            continue
        n = n / norm
        mid = seg.midpoint
        d = float(n @ mid)
        # orient so the camera is on the room side of the wall
        if float(n @ c) > d:
            n = -n
            d = -d
        out.append(WallPlane(
            normal=n, offset=d, midpoint=mid, support=int(seg.inliers),
            length=float(seg.length), y_min=float(seg.y_min), y_max=float(seg.y_max),
        ))
    return out


def merge_coplanar(
    planes: Sequence[WallPlane], angle_eps_deg: float = 6.0, offset_eps: float = 0.14
) -> List[WallPlane]:
    """Merge planes that are the same physical wall (split by a door, or re-detected)."""
    merged: List[WallPlane] = []
    for p in sorted(planes, key=lambda q: -q.support):
        placed = False
        for i, q in enumerate(merged):
            dang = abs(((p.angle_deg - q.angle_deg + 90.0) % 180.0) - 90.0)
            if dang > angle_eps_deg:
                continue
            if abs(float(p.normal @ q.midpoint) - p.offset) > offset_eps:
                continue
            total = q.support + p.support
            w = p.support / max(total, 1)
            merged[i] = WallPlane(
                normal=(1 - w) * q.normal + w * p.normal,
                offset=(1 - w) * q.offset + w * p.offset,
                midpoint=(1 - w) * q.midpoint + w * p.midpoint,
                support=total,
                length=q.length + p.length,
                y_min=min(q.y_min, p.y_min), y_max=max(q.y_max, p.y_max),
            )
            placed = True
            break
        if not placed:
            merged.append(p)
    for m in merged:
        m.normal = m.normal / max(float(np.linalg.norm(m.normal)), 1e-9)
    return merged


def snap_manhattan(
    planes: Sequence[WallPlane], angle_tol_deg: float = 22.0
) -> Tuple[List[WallPlane], float]:
    """Snap wall azimuths to a common orthogonal lattice.

    The lattice angle is the circular mean of ``4 * theta`` over the planes, which is the
    standard trick for averaging directions modulo 90 degrees.
    """
    if not planes:
        return [], 0.0
    thetas = np.array([np.arctan2(p.normal[1], p.normal[0]) for p in planes])
    weights = np.array([max(p.support, 1) for p in planes], dtype=float)
    acc = np.sum(weights * np.exp(4j * thetas))
    dominant = (np.angle(acc) / 4.0) if abs(acc) > 1e-12 else 0.0
    out: List[WallPlane] = []
    for p in planes:
        theta = np.arctan2(p.normal[1], p.normal[0])
        dev = ((np.degrees(theta - dominant) + 45.0) % 90.0) - 45.0
        if abs(dev) > angle_tol_deg:
            continue                       # not part of the Manhattan structure
        snapped = theta - np.radians(dev)
        n = np.array([np.cos(snapped), np.sin(snapped)])
        out.append(WallPlane(
            normal=n, offset=float(n @ p.midpoint), midpoint=p.midpoint,
            support=p.support, length=p.length, y_min=p.y_min, y_max=p.y_max,
        ))
    return out, float(np.degrees(dominant) % 90.0)


def plane_support(points: np.ndarray, polygon: np.ndarray, floor_y: float,
                  band: Tuple[float, float] = (0.8, 1.6), max_dist: float = 0.20) -> float:
    """Fraction of torso-band points lying on the polygon boundary."""
    shifted = np.asarray(points, dtype=float).copy()
    shifted[:, UP_AXIS] -= floor_y
    b = band_points(shifted, *band)
    if len(b) < 20:
        return 0.0
    d = distance_to_polygon_boundary(b[:, [0, 2]], polygon)
    return float(np.mean(d <= max_dist))


def fit_room_polygon(
    points: np.ndarray,
    camera_xy: Optional[Sequence[float]] = None,
    floor_y: Optional[float] = None,
    band: Tuple[float, float] = (0.5, 1.9),
    min_planes: int = 3,
    min_support: int = 40,
    extent_cap: float = 25.0,
    snap: bool = True,
) -> Optional[RoomPolygon]:
    """Reconstruct a room from detected vertical wall planes.

    Why this replaces ``fit_room_box``: a bounding rectangle of a torso slice is fitted to
    whatever happens to be there (furniture, a plant, the far room through a doorway) and it
    cannot generalise — measured held-out support was 0.08 on real data. Wall planes carry
    *orientation*, so each one constrains which side the room is on, and the room is their
    half-plane intersection. Walls that were never seen leave the polygon unbounded on that
    side, which is reported instead of being invented.
    """
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    if len(pts) < 200:
        return None
    est_floor, est_ceiling = estimate_floor_ceiling(pts)
    fy = float(floor_y) if floor_y is not None else est_floor
    ceiling = (est_ceiling - est_floor + fy) if est_ceiling is not None else 2.7
    shifted = pts.copy()
    shifted[:, UP_AXIS] -= fy

    c = np.asarray(camera_xy, dtype=float) if camera_xy is not None else np.array([0.0, 0.0])
    horiz = np.linalg.norm(shifted[:, [0, 2]] - c[None, :], axis=1)
    extent = float(np.clip(1.5 * np.percentile(horiz, 95), 6.0, extent_cap))

    segments = detect_wall_segments(shifted, band=band, min_inliers=min_support)
    if not segments:
        return None

    # A wall spans the band; a table top or a wardrobe face does not. Without this the
    # furniture planes cut the room into pieces (measured: IoU 1.00 -> 0.33 on a clean room
    # once two boxes of furniture were added).
    band_height = float(band[1] - band[0])
    segments = [sg for sg in segments
                if (sg.y_max - sg.y_min) >= 0.7 * band_height and sg.length >= 0.8]
    if not segments:
        return None
    planes = planes_from_segments(segments, camera_xy=c)
    planes = merge_coplanar(planes)
    if snap:
        planes, _ = snap_manhattan(planes)
    if not planes:
        return None

    halfplanes = [(p.normal, p.offset) for p in planes]

    # Bound the room by the floor we actually observed. Without this, an open view (walls
    # seen on one or two sides) leaves the half-plane intersection unbounded and produces a
    # 96 m2 "room" whose boundary still scores well on support. The floor footprint is a
    # crude regulariser, but it is observed rather than invented.
    floor_pts = shifted[shifted[:, UP_AXIS] < 0.25][:, [0, 2]]
    if len(floor_pts) >= 120:
        hull = convex_hull(floor_pts)
        # Only trust the floor footprint when it is actually a room-sized patch. A desk or a
        # rug occludes the floor and leaves a 1-2 m2 hull, which would clip the room away
        # (observed on TUM: a 5.8 m2 room collapsing to 0.6 m2).
        if len(hull) < 4 or polygon_area(hull) < 3.0:
            hull = None
        if hull is not None:
            cen = hull.mean(axis=0)
            for i in range(len(hull)):
                p0, p1 = hull[i], hull[(i + 1) % len(hull)]
                d = p1 - p0
                nrm = np.array([-d[1], d[0]])
                ln = float(np.linalg.norm(nrm))
                if ln < 1e-9:
                    continue
                nrm = nrm / ln
                if float(nrm @ cen) > float(nrm @ p0):   # ensure interior is the <= side
                    nrm = -nrm
                halfplanes.append((nrm, float(nrm @ p0) + 0.15))

    poly, binding = polygon_from_halfplanes(halfplanes, centre=c, extent=extent)
    binding = [b for b in binding if b < len(planes)]
    if len(poly) < 3 or polygon_area(poly) < 0.5:
        return None

    return RoomPolygon(
        polygon=ensure_ccw(poly),
        floor_y=fy,
        ceiling=float(np.clip(ceiling, 1.5, 6.0)),
        planes=planes,
        binding=binding,
        bounded=len(binding) >= min_planes,
        support=plane_support(pts, poly, fy),
        n_points=int(len(pts)),
    )
