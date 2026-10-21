from __future__ import annotations

import unittest

import numpy as np

from pin3d.geometry import clip_halfplane, polygon_area, polygon_from_halfplanes, polygon_iou
from pin3d.pipeline.planes import (
    fit_room_box, fit_room_polygon, merge_coplanar, planes_from_segments, snap_manhattan,
)
from pin3d.sim import Capture, add_clutter, capture_cloud

RECT = np.array([[0, 0], [4.0, 0], [4.0, 3.0], [0, 3.0]], dtype=float)
SQUARE_PLANES = [
    (np.array([1.0, 0.0]), 2.0), (np.array([-1.0, 0.0]), 2.0),
    (np.array([0.0, 1.0]), 1.5), (np.array([0.0, -1.0]), 1.5),
]


class TestHalfPlane(unittest.TestCase):
    def test_clip_keeps_inside(self):
        poly = np.array([[-5, -5], [5, -5], [5, 5], [-5, 5]], dtype=float)
        out = clip_halfplane(poly, np.array([1.0, 0.0]), 2.0)
        self.assertAlmostEqual(polygon_area(out), 70.0)   # x in [-5, 2] (7) x y in [-5, 5] (10)

    def test_clip_fully_outside_returns_empty(self):
        poly = np.array([[-5, -5], [-4, -5], [-4, 5], [-5, 5]], dtype=float)
        out = clip_halfplane(poly, np.array([1.0, 0.0]), -10.0)
        self.assertEqual(len(out), 0)

    def test_closed_square(self):
        poly, binding = polygon_from_halfplanes(SQUARE_PLANES, centre=(0, 0))
        self.assertAlmostEqual(polygon_area(poly), 12.0, places=6)
        self.assertEqual(len(binding), 4)

    def test_open_square_reports_fewer_binding(self):
        poly, binding = polygon_from_halfplanes(SQUARE_PLANES[:3], centre=(0, 0))
        self.assertEqual(len(binding), 3)
        self.assertGreater(polygon_area(poly), 12.0)


class TestPlaneModel(unittest.TestCase):
    @staticmethod
    def _cloud(polygon, cam, az, hfov, clutter=0, seed=0):
        cap = Capture(polygon=polygon, camera_xy=cam, azimuth_deg=az, hfov_deg=hfov)
        cloud = capture_cloud(cap, seed=seed)
        if clutter:
            cloud = add_clutter(cloud, polygon, n_boxes=clutter, seed=seed + 1)
        return cloud

    def test_panorama_recovers_rectangle(self):
        cloud = self._cloud(RECT, (2.0, 1.5), 0.0, 350.0)
        room = fit_room_polygon(cloud, camera_xy=(2.0, 1.5))
        self.assertIsNotNone(room)
        self.assertTrue(room.bounded)
        self.assertGreaterEqual(len(room.planes), 4)
        self.assertAlmostEqual(room.area, 12.0, delta=0.6)
        self.assertGreater(polygon_iou(room.polygon, RECT), 0.95)

    def test_panorama_survives_furniture(self):
        cloud = self._cloud(RECT, (2.0, 1.5), 0.0, 350.0, clutter=2, seed=4)
        room = fit_room_polygon(cloud, camera_xy=(2.0, 1.5))
        self.assertIsNotNone(room)
        self.assertGreater(polygon_iou(room.polygon, RECT), 0.9)

    def test_partial_view_beats_box(self):
        # the case that matters for listing photos: a 120 degree corner view
        cam = (0.9, 0.9)
        cloud = self._cloud(RECT, cam, 45.0, 120.0, seed=2)
        room = fit_room_polygon(cloud, camera_xy=cam)
        box = fit_room_box(cloud)
        self.assertIsNotNone(room)
        self.assertIsNotNone(box)
        iou_planes = polygon_iou(room.polygon, RECT)
        iou_box = polygon_iou(box.polygon, RECT)
        self.assertGreater(iou_planes, 0.8, msg=f"planes IoU {iou_planes:.3f}")
        self.assertGreater(iou_planes, iou_box,
                           msg=f"planes {iou_planes:.3f} should beat box {iou_box:.3f}")

    def test_very_narrow_view_is_flagged_unbounded(self):
        cloud = self._cloud(RECT, (2.0, 1.5), 0.0, 30.0, seed=3)
        room = fit_room_polygon(cloud, camera_xy=(2.0, 1.5))
        if room is not None:
            self.assertFalse(room.bounded)

    def test_too_few_points(self):
        self.assertIsNone(fit_room_polygon(np.zeros((50, 3))))


class TestPlaneMerging(unittest.TestCase):
    def test_coplanar_planes_merge(self):
        from pin3d.pipeline.planes import WallSegment
        segs = [
            WallSegment(p0=np.array([0.0, 0.0]), p1=np.array([1.0, 0.0]), inliers=100,
                        normal=np.array([0.0, 1.0]), y_min=0.0, y_max=2.5),
            WallSegment(p0=np.array([1.5, 0.01]), p1=np.array([3.0, 0.01]), inliers=90,
                        normal=np.array([0.0, 1.0]), y_min=0.0, y_max=2.5),
        ]
        planes = merge_coplanar(planes_from_segments(segs, camera_xy=(0.0, -2.0)))
        self.assertEqual(len(planes), 1)
        self.assertEqual(planes[0].support, 190)

    def test_distinct_planes_kept(self):
        from pin3d.pipeline.planes import WallSegment
        segs = [
            WallSegment(p0=np.array([0.0, 0.0]), p1=np.array([1.0, 0.0]), inliers=100,
                        normal=np.array([0.0, 1.0]), y_min=0.0, y_max=2.5),
            WallSegment(p0=np.array([3.0, 0.0]), p1=np.array([3.0, 2.0]), inliers=90,
                        normal=np.array([-1.0, 0.0]), y_min=0.0, y_max=2.5),
        ]
        planes = merge_coplanar(planes_from_segments(segs, camera_xy=(1.5, 1.0)))
        self.assertEqual(len(planes), 2)

    def test_snap_manhattan_makes_orthogonal(self):
        from pin3d.pipeline.planes import WallSegment
        segs = [
            WallSegment(p0=np.array([0.0, 0.0]), p1=np.array([3.0, 0.0]), inliers=100,
                        normal=np.array([0.0, 1.0]), y_min=0.0, y_max=2.5),
            WallSegment(p0=np.array([3.0, 0.0]), p1=np.array([3.0, 2.0]), inliers=90,
                        normal=np.array([-1.0, 0.0]), y_min=0.0, y_max=2.5),
        ]
        planes, dominant = snap_manhattan(planes_from_segments(segs, camera_xy=(1.5, 1.0)))
        angles = sorted(round(p.angle_deg % 180.0, 1) for p in planes)
        self.assertEqual(len(angles), 2)
        self.assertAlmostEqual(abs(angles[0] - angles[1]), 90.0, delta=1.0)


if __name__ == "__main__":
    unittest.main()
