#!/usr/bin/env python3
"""Box vs. wall planes under controlled visibility, scored against known ground truth.

Real captures cannot settle this: TUM's room view sees mostly one wall, so neither model can
close the room and the comparison is uninformative. Here visibility is controlled, so the
cases that matter can be constructed:

  * full view of a rectangle    -> box should be fine, planes should match
  * partial view (1-3 walls)    -> box must invent walls; planes should report it
  * clutter at torso height     -> box is fitted to furniture; planes should ignore it
  * L-shaped room               -> a rectangle cannot represent it at all
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pin3d.geometry import polygon_area, polygon_centroid, polygon_iou   # noqa: E402
from pin3d.pipeline.planes import fit_room_box, fit_room_polygon          # noqa: E402
from pin3d.sim import Capture, add_clutter, capture_cloud                 # noqa: E402

RECT = np.array([[0, 0], [4.0, 0], [4.0, 3.0], [0, 3.0]], dtype=float)
LROOM = np.array([[0, 0], [5.0, 0], [5.0, 2.2], [2.4, 2.2], [2.4, 4.2], [0, 4.2]], dtype=float)


def run(name, polygon, cam, az, hfov=68.0, clutter=0, seed=0):
    cap = Capture(polygon=polygon, camera_xy=cam, azimuth_deg=az, hfov_deg=hfov)
    cloud = capture_cloud(cap, seed=seed)
    if clutter:
        cloud = add_clutter(cloud, polygon, n_boxes=clutter, seed=seed + 1)
    truth_area = abs(polygon_area(polygon))

    b = fit_room_box(cloud)
    r = fit_room_polygon(cloud, camera_xy=cam)

    def fmt(model, poly, extra=""):
        if poly is None:
            return f"{model:<7} FAILED"
        iou = polygon_iou(poly, polygon)
        return (f"{model:<7} IoU {iou:5.3f}  area {abs(polygon_area(poly)):6.1f} "
                f"(truth {truth_area:5.1f}){extra}")

    print(f"\n{name}  (hfov {hfov:.0f}deg, {clutter} clutter boxes)")
    print("  " + fmt("box", None if b is None else b.polygon))
    print("  " + fmt("planes", None if r is None else r.polygon,
                     "" if r is None else
                     f"  planes {len(r.planes)} binding {len(r.binding)} bounded {r.bounded}"))


def main() -> int:
    c = tuple(polygon_centroid(RECT))
    print("=" * 74)
    print("rectangle 4.0 x 3.0 m")
    # corner camera, 120 deg: what a wide-angle listing photo actually gets
    corner = (0.9, 0.9)
    run("panorama (all walls)", RECT, c, 0.0, hfov=350.0, seed=1)
    run("corner view 120deg (3 walls)", RECT, corner, 45.0, hfov=120.0, seed=2)
    run("narrow view 68deg (2 walls)", RECT, c, 0.0, hfov=68.0, seed=3)
    run("panorama + furniture", RECT, c, 0.0, hfov=350.0, clutter=2, seed=4)
    run("corner view + furniture", RECT, corner, 45.0, hfov=120.0, clutter=2, seed=5)

    cl = tuple(polygon_centroid(LROOM))
    print("\n" + "=" * 74)
    print("L-shaped room (5.0 x 4.2 m outline, 15.3 m2)")
    run("panorama (all walls)", LROOM, cl, 0.0, hfov=350.0, seed=6)
    run("corner view 120deg", LROOM, (1.0, 1.0), 45.0, hfov=120.0, seed=7)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
