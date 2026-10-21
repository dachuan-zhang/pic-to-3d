"""2D geometry primitives used by the floorplan pipeline.

Everything is metres and works on ``(N, 2)`` float arrays. shapely is used when available
(for exact IoU on non-convex rooms); a raster fallback keeps the library dependency-free.
"""
from __future__ import annotations

import math
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:  # pragma: no cover - exercised implicitly
    from shapely.geometry import Polygon as _ShapelyPolygon
    _HAVE_SHAPELY = True
except Exception:  # pragma: no cover
    _HAVE_SHAPELY = False

Poly = np.ndarray  # (N,2)


def as_poly(poly: Iterable[Sequence[float]]) -> Poly:
    return np.asarray(list(poly), dtype=float).reshape(-1, 2)


# --------------------------------------------------------------------------
# BLAS-free small-vector helpers
# --------------------------------------------------------------------------
# On macOS numpy 2.0 + Accelerate BLAS, ``(N, 2) @ (2,)`` sets spurious floating-point
# status flags and triggers RuntimeWarnings/FloatingPointErrors for perfectly finite input.
# These helpers do the same arithmetic without touching BLAS. Results are identical.
def dot_rows(a: np.ndarray, v: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float).reshape(-1, 2)
    v = np.asarray(v, dtype=float)
    return a[:, 0] * v[0] + a[:, 1] * v[1]


def dot2(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(a[0] * b[0] + a[1] * b[1])


def rot_points(points: np.ndarray, r: np.ndarray) -> Poly:
    """Apply a 2x2 rotation matrix to (N,2) points (BLAS-free)."""
    p = np.asarray(points, dtype=float).reshape(-1, 2)
    r = np.asarray(r, dtype=float)
    return np.stack([
        p[:, 0] * r[0, 0] + p[:, 1] * r[0, 1],
        p[:, 0] * r[1, 0] + p[:, 1] * r[1, 1],
    ], axis=1)


def rot_vec(r: np.ndarray, v: np.ndarray) -> np.ndarray:
    r = np.asarray(r, dtype=float)
    v = np.asarray(v, dtype=float)
    return np.array([
        r[0, 0] * v[0] + r[0, 1] * v[1],
        r[1, 0] * v[0] + r[1, 1] * v[1],
    ])


def _signed_area(poly: Poly) -> float:
    """Shoelace area, positive for CCW. Also the orientation test."""
    p = as_poly(poly)
    if len(p) < 3:
        return 0.0
    x, y = p[:, 0], p[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def polygon_area(poly: Poly) -> float:
    return abs(_signed_area(poly))


def ensure_ccw(poly: Poly) -> Poly:
    p = as_poly(poly)
    return p if _signed_area(p) >= 0 else p[::-1].copy()


def polygon_centroid(poly: Poly) -> np.ndarray:
    """Area centroid (falls back to vertex mean for degenerate input)."""
    p = as_poly(poly)
    a = _signed_area(p)
    if len(p) < 3 or abs(a) < 1e-12:
        return p.mean(axis=0) if len(p) else np.zeros(2)
    x, y = p[:, 0], p[:, 1]
    x1, y1 = np.roll(x, -1), np.roll(y, -1)
    cross = x * y1 - x1 * y
    return np.array([np.sum((x + x1) * cross), np.sum((y + y1) * cross)]) / (6.0 * a)


def wall_segments(poly: Poly) -> List[Tuple[np.ndarray, np.ndarray]]:
    p = as_poly(poly)
    if len(p) < 2:
        return []
    return [(p[i], p[(i + 1) % len(p)]) for i in range(len(p))]


def edge_lengths(poly: Poly) -> np.ndarray:
    p = as_poly(poly)
    d = np.roll(p, -1, axis=0) - p
    return np.linalg.norm(d, axis=1)


def edge_directions(poly: Poly) -> np.ndarray:
    p = as_poly(poly)
    d = np.roll(p, -1, axis=0) - p
    n = np.linalg.norm(d, axis=1, keepdims=True)
    return d / np.where(n > 1e-12, n, 1.0)


def side_lengths(poly: Poly) -> Tuple[float, float]:
    """Width/height of the axis-aligned bounding box."""
    p = as_poly(poly)
    lo, hi = p.min(axis=0), p.max(axis=0)
    return float(hi[0] - lo[0]), float(hi[1] - lo[1])


# --------------------------------------------------------------------------
# hull / rectangle fitting
# --------------------------------------------------------------------------
def convex_hull(points: np.ndarray) -> np.ndarray:
    """Andrew monotone chain, CCW, no repeated endpoint."""
    pts = np.unique(np.asarray(points, dtype=float).reshape(-1, 2), axis=0)
    if len(pts) <= 2:
        return pts
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def half(seq: np.ndarray) -> List[np.ndarray]:
        out: List[np.ndarray] = []
        for p in seq:
            while len(out) >= 2:
                o, a = out[-2], out[-1]
                if (a[0] - o[0]) * (p[1] - o[1]) - (a[1] - o[1]) * (p[0] - o[0]) <= 0:
                    out.pop()
                else:
                    break
            out.append(p)
        return out

    lower = half(pts)
    upper = half(pts[::-1])
    return np.array(lower[:-1] + upper[:-1])


def min_area_rect(points: np.ndarray) -> Tuple[Poly, float]:
    """Minimum-area enclosing rectangle (rotating calipers).

    Returns ``(corners, angle)`` where ``angle`` is the rectangle's rotation in radians,
    normalised to ``[0, pi/2)``. Corners are CCW.
    """
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    hull = convex_hull(pts)
    if len(hull) < 3:
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        w, h = hi - lo
        corners = np.array([[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]])
        return corners, 0.0

    best: Optional[Tuple[float, float, np.ndarray, np.ndarray]] = None
    for i in range(len(hull)):
        d = hull[(i + 1) % len(hull)] - hull[i]
        ang = math.atan2(d[1], d[0])
        c, s = math.cos(-ang), math.sin(-ang)
        r = np.array([[c, -s], [s, c]])
        rot = hull @ r.T
        lo, hi = rot.min(axis=0), rot.max(axis=0)
        area = float((hi[0] - lo[0]) * (hi[1] - lo[1]))
        if best is None or area < best[0] - 1e-9:
            best = (area, ang, lo, hi)
    assert best is not None
    _, ang, lo, hi = best
    box = np.array([
        [lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]],
    ])
    c, s = math.cos(ang), math.sin(ang)
    corners = box @ np.array([[c, -s], [s, c]]).T
    ang = ang % (math.pi / 2)
    return ensure_ccw(corners), float(ang)


def snap_angle(angle: float, period: float = math.pi / 2) -> float:
    """Snap to the nearest multiple of ``period`` (Manhattan rectification)."""
    return round(angle / period) * period


def rect_corners(cx: float, cy: float, w: float, h: float) -> Poly:
    return np.array([
        [cx - w / 2, cy - h / 2],
        [cx + w / 2, cy - h / 2],
        [cx + w / 2, cy + h / 2],
        [cx - w / 2, cy + h / 2],
    ])


# --------------------------------------------------------------------------
# IoU / distances
# --------------------------------------------------------------------------
def _raster_iou(a: Poly, b: Poly, step: float = 0.02) -> float:
    """Pixel-sampling IoU fallback for arbitrary (non-convex) polygons."""
    pts = np.vstack([as_poly(a), as_poly(b)])
    lo, hi = pts.min(axis=0) - step, pts.max(axis=0) + step
    nx = max(2, int(math.ceil((hi[0] - lo[0]) / step)))
    ny = max(2, int(math.ceil((hi[1] - lo[1]) / step)))
    if nx * ny > 4_000_000:  # keep the fallback bounded
        step *= math.sqrt(nx * ny / 4_000_000)
        nx = max(2, int(math.ceil((hi[0] - lo[0]) / step)))
        ny = max(2, int(math.ceil((hi[1] - lo[1]) / step)))
    xs = lo[0] + (np.arange(nx) + 0.5) * (hi[0] - lo[0]) / nx
    ys = lo[1] + (np.arange(ny) + 0.5) * (hi[1] - lo[1]) / ny
    gx, gy = np.meshgrid(xs, ys)
    in_a = points_in_polygon(np.column_stack([gx.ravel(), gy.ravel()]), a)
    in_b = points_in_polygon(np.column_stack([gx.ravel(), gy.ravel()]), b)
    inter = float(np.count_nonzero(in_a & in_b))
    union = float(np.count_nonzero(in_a | in_b))
    return inter / union if union > 0 else 0.0


def polygon_iou(a: Poly, b: Poly) -> float:
    pa, pb = as_poly(a), as_poly(b)
    if len(pa) < 3 or len(pb) < 3:
        return 0.0
    if _HAVE_SHAPELY:
        sa, sb = _ShapelyPolygon(pa), _ShapelyPolygon(pb)
        if not sa.is_valid:
            sa = sa.buffer(0)
        if not sb.is_valid:
            sb = sb.buffer(0)
        inter = sa.intersection(sb).area
        union = sa.union(sb).area
        return float(inter / union) if union > 0 else 0.0
    return _raster_iou(pa, pb)


def polygon_overlap_area(a: Poly, b: Poly) -> float:
    pa, pb = as_poly(a), as_poly(b)
    if len(pa) < 3 or len(pb) < 3:
        return 0.0
    if _HAVE_SHAPELY:
        sa, sb = _ShapelyPolygon(pa), _ShapelyPolygon(pb)
        if not sa.is_valid:
            sa = sa.buffer(0)
        if not sb.is_valid:
            sb = sb.buffer(0)
        return float(sa.intersection(sb).area)
    return _raster_iou(pa, pb) * max(polygon_area(pa), polygon_area(pb))


def chamfer_corner_distance(a: Poly, b: Poly) -> float:
    """Symmetric mean nearest-vertex distance (metres)."""
    pa, pb = as_poly(a), as_poly(b)
    if not len(pa) or not len(pb):
        return float("inf")
    d = np.linalg.norm(pa[:, None, :] - pb[None, :, :], axis=2)
    return float(0.5 * (d.min(axis=1).mean() + d.min(axis=0).mean()))


def points_in_polygon(points: np.ndarray, poly: Poly) -> np.ndarray:
    """Even-odd ray casting, vectorised over ``points``."""
    p = as_poly(poly)
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    if len(p) < 3:
        return np.zeros(len(pts), dtype=bool)
    x, y = pts[:, 0], pts[:, 1]
    inside = np.zeros(len(pts), dtype=bool)
    j = len(p) - 1
    for i in range(len(p)):
        xi, yi = p[i]
        xj, yj = p[j]
        crosses = (yi > y) != (yj > y)
        with np.errstate(divide="ignore", invalid="ignore"):
            x_int = xi + (y - yi) * (xj - xi) / (yj - yi)
        inside ^= crosses & (x < x_int)
        j = i
    return inside


# --------------------------------------------------------------------------
# Manhattan rectification
# --------------------------------------------------------------------------
def _cluster_1d(values: np.ndarray, tol: float) -> np.ndarray:
    """Merge values within ``tol`` of the running cluster mean."""
    v = np.asarray(values, dtype=float)
    order = np.argsort(v)
    out = v.copy()
    group: List[int] = []
    for idx in order:
        if group and abs(v[idx] - float(np.mean(v[group]))) > tol:
            m = float(np.mean(v[group]))
            out[group] = m
            group = []
        group.append(idx)
    if group:
        out[group] = float(np.mean(v[group]))
    return out


def rectify_polygon(poly: Poly, tol: float = 0.08) -> Poly:
    """Snap near-duplicate x/y coordinates so near-Manhattan polygons become exact.

    Returns a new polygon; orientation is preserved.
    """
    p = as_poly(poly)
    if len(p) < 3:
        return p
    out = p.copy()
    out[:, 0] = _cluster_1d(p[:, 0], tol)
    out[:, 1] = _cluster_1d(p[:, 1], tol)
    return out


def rectilinear_score(poly: Poly, angle_tol_deg: float = 3.0) -> float:
    """Fraction of edges that are (near) horizontal or vertical."""
    p = as_poly(poly)
    if len(p) < 2:
        return 0.0
    d = np.roll(p, -1, axis=0) - p
    ang = np.abs(np.degrees(np.arctan2(d[:, 1], d[:, 0]))) % 90.0
    ang = np.minimum(ang, 90.0 - ang)
    return float(np.mean(ang <= angle_tol_deg))


def snap_to_grid(poly: Poly, step: float = 0.05) -> Poly:
    return np.round(as_poly(poly) / step) * step


def simplify_collinear(poly: Poly, tol: float = 1e-6) -> Poly:
    """Drop vertices that lie on the straight line between their neighbours."""
    p = as_poly(poly)
    if len(p) < 4:
        return p
    keep = []
    n = len(p)
    for i in range(n):
        a, b, c = p[(i - 1) % n], p[i], p[(i + 1) % n]
        cross = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        if abs(cross) > tol:
            keep.append(i)
    return p[keep] if len(keep) >= 3 else p


def transform_polygon(poly: Poly, theta: float, tx: float, ty: float) -> Poly:
    c, s = math.cos(theta), math.sin(theta)
    r = np.array([[c, -s], [s, c]])
    return rot_points(poly, r) + np.array([tx, ty], dtype=float)


def wall_normal(p0: np.ndarray, p1: np.ndarray) -> np.ndarray:
    """Unit normal pointing to the left of the p0->p1 direction."""
    d = np.asarray(p1, dtype=float) - np.asarray(p0, dtype=float)
    n = np.linalg.norm(d)
    d = d / n if n > 1e-12 else np.array([1.0, 0.0])
    return np.array([-d[1], d[0]])


# --------------------------------------------------------------------------
# half-plane intersection (used to turn wall planes into a room polygon)
# --------------------------------------------------------------------------
def clip_halfplane(poly: Poly, normal: np.ndarray, offset: float, eps: float = 1e-9) -> Poly:
    """Clip a convex polygon to ``normal . x <= offset`` (Sutherland-Hodgman)."""
    p = as_poly(poly)
    if len(p) < 3:
        return p
    n = np.asarray(normal, dtype=float)
    n = n / max(float(np.linalg.norm(n)), 1e-12)
    d = float(offset) / max(float(np.linalg.norm(normal)), 1e-12)
    vals = dot_rows(p, n) - d
    out: List[np.ndarray] = []
    for i in range(len(p)):
        j = (i - 1) % len(p)
        cur, prev = p[i], p[j]
        vc, vp = vals[i], vals[j]
        if vc <= eps:                      # current inside
            if vp > eps:                   # entering
                t = vp / (vp - vc)
                out.append(prev + t * (cur - prev))
            out.append(cur)
        elif vp <= eps:                    # leaving
            t = vp / (vp - vc)
            out.append(prev + t * (cur - prev))
    return np.array(out) if len(out) >= 3 else np.zeros((0, 2))


def polygon_from_halfplanes(
    planes: Sequence[Tuple[np.ndarray, float]],
    centre: Sequence[float] = (0.0, 0.0),
    extent: float = 12.0,
) -> Tuple[Poly, List[int]]:
    """Intersect half-planes ``n_i . x <= d_i`` into a bounded polygon.

    Starts from a generous square around ``centre`` and clips. Returns the polygon plus the
    indices of planes that actually cut it — a plane that never binds is a wall that was
    detected but is not a room boundary (e.g. a wardrobe face), and the caller should not
    treat the room as closed on that side.
    """
    c = np.asarray(centre, dtype=float)
    poly = np.array([
        [c[0] - extent, c[1] - extent], [c[0] + extent, c[1] - extent],
        [c[0] + extent, c[1] + extent], [c[0] - extent, c[1] + extent],
    ])
    binding: List[int] = []
    for i, (n, d) in enumerate(planes):
        before = poly
        poly = clip_halfplane(poly, n, d)
        if len(poly) < 3:
            return np.zeros((0, 2)), []
        if len(poly) != len(before) or not np.allclose(poly, before):
            binding.append(i)
    return ensure_ccw(poly), binding


def polygon_perimeter(poly: Poly) -> float:
    return float(edge_lengths(poly).sum())
