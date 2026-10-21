"""Shared synthetic-cloud helpers for the test suite."""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np


def make_room_cloud(
    lo: Sequence[float] = (0.0, 0.0, 0.0),
    hi: Sequence[float] = (3.6, 2.6, 4.0),
    density: float = 60.0,
    gap: Optional[Tuple[int, float, float, float]] = None,
    noise: float = 0.01,
    outliers: float = 0.0,
    seed: int = 0,
) -> np.ndarray:
    """Sample the interior surfaces of an axis-aligned box, optionally with a doorway gap.

    ``gap = (axis, at, span_lo, span_hi)`` removes points on the plane ``x[axis] == at``
    whose coordinate on the *other* horizontal axis lies in ``[span_lo, span_hi]``, below
    ``GAP_HEIGHT``. +Y is up.
    """
    rng = np.random.default_rng(seed)
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    out = []

    for axis in (0, 2):
        other = 2 if axis == 0 else 0
        area = (hi[other] - lo[other]) * (hi[1] - lo[1])
        n = max(24, int(area * density))
        along = rng.uniform(lo[other], hi[other], n)
        vert = rng.uniform(lo[1], hi[1], n)
        for val in (lo[axis], hi[axis]):
            pts = np.zeros((n, 3))
            pts[:, axis] = val
            pts[:, other] = along
            pts[:, 1] = vert
            out.append(pts)

    area = (hi[0] - lo[0]) * (hi[2] - lo[2])
    n = max(64, int(area * density * 0.5))
    xs = rng.uniform(lo[0], hi[0], n)
    zs = rng.uniform(lo[2], hi[2], n)
    for val in (lo[1], hi[1]):
        out.append(np.stack([xs, np.full(n, val), zs], axis=1))

    pts = np.vstack(out)
    if gap is not None:
        axis, at, s0, s1 = gap
        other = 2 if axis == 0 else 0
        cut = (
            (np.abs(pts[:, axis] - at) < 0.05)
            & (pts[:, other] >= s0)
            & (pts[:, other] <= s1)
            & (pts[:, 1] <= GAP_HEIGHT)
        )
        pts = pts[~cut]

    pts = pts + rng.normal(0.0, noise, pts.shape)
    if outliers > 0:
        n_out = max(1, int(outliers * len(pts)))
        pts = np.vstack([pts, rng.uniform(lo, hi, (n_out, 3)) * 1.8])
    return pts


GAP_HEIGHT = 2.05
