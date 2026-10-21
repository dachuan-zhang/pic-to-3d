#!/usr/bin/env python3
"""Wall planes vs. bounding box, judged by held-out support.

The box model is fitted to the min-area rectangle of whatever is in the torso band, so its own
"support" number is self-fulfilling. The only honest question is: does a room model fitted on
some views explain a view it never saw? That is the number here.

    python scripts/planes_vs_box.py --tum /tmp/tum/rgbd_dataset_freiburg1_room
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional, Sequence

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pin3d.geometry import polygon_area                                  # noqa: E402
from pin3d.pipeline.openings import GapOpeningDetector                   # noqa: E402
from pin3d.pipeline.planes import (                                      # noqa: E402
    band_points, distance_to_polygon_boundary, fit_room_box, fit_room_polygon,
)
from multiview_experiment import load_tum, pick_group, world_cloud        # noqa: E402


def _support(polygon: np.ndarray, floor_y: float, cloud: np.ndarray,
             max_dist: float = 0.20) -> float:
    shifted = np.asarray(cloud, dtype=float).copy()
    shifted[:, 1] -= floor_y
    band = band_points(shifted, 0.8, 1.6)
    if len(band) < 50:
        return float("nan")
    d = distance_to_polygon_boundary(band[:, [0, 2]], polygon)
    return float(np.mean(d <= max_dist))


def fit(method: str, cloud: np.ndarray, camera_xy: Sequence[float]) -> Optional[Dict]:
    if method == "box":
        b = fit_room_box(cloud)
        if b is None:
            return None
        return {"polygon": b.polygon, "floor_y": b.floor_y, "ceiling": b.ceiling,
                "area": b.area, "n_planes": None, "bounded": True,
                "doors": len(GapOpeningDetector().detect(
                    cloud, b.polygon, floor_y=b.floor_y, ceiling=b.ceiling))}
    r = fit_room_polygon(cloud, camera_xy=camera_xy)
    if r is None:
        return None
    return {"polygon": r.polygon, "floor_y": r.floor_y, "ceiling": r.ceiling,
            "area": r.area, "n_planes": len(r.planes), "bounded": r.bounded,
            "doors": len(GapOpeningDetector().detect(
                cloud, r.polygon, floor_y=r.floor_y, ceiling=r.ceiling))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tum", default="/tmp/tum/rgbd_dataset_freiburg1_room")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--views", type=int, default=5)
    args = ap.parse_args()

    frames = load_tum(args.tum, limit=args.limit)
    idx = pick_group(frames, len(frames) // 2, args.views)
    seed = frames[idx[0]]
    cam = (float(seed.t[0]), float(seed.t[2]))
    print(f"frames {idx} · baseline {[round(float(np.linalg.norm(frames[i].t - seed.t)), 2) for i in idx]} m")

    clouds = {i: world_cloud(frames[i], frames[i].depth) for i in idx}
    held = idx[-1]
    fit_views = [i for i in idx if i != held]

    print(f"\n{'method':<8} {'fit on':<14} {'area m2':>8} {'support(fit)':>13} "
          f"{'held-out':>9} {'bounded':>8} {'doors':>6}")
    out = {}
    for method in ("box", "planes"):
        for label, views in (("1 view", fit_views[:1]), ("%d views" % len(fit_views), fit_views)):
            cloud = np.vstack([clouds[i] for i in views])
            r = fit(method, cloud, cam)
            if r is None:
                print(f"{method:<8} {label:<14} {'FAILED':>8}")
                continue
            ho = _support(r["polygon"], r["floor_y"], clouds[held])
            print(f"{method:<8} {label:<14} {r['area']:8.1f} "
                  f"{_support(r['polygon'], r['floor_y'], cloud):13.2f} "
                  f"{ho:9.2f} {str(r['bounded']):>8} {r['doors']:6d}")
            out[(method, label)] = {"area": r["area"], "heldout": ho}
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
