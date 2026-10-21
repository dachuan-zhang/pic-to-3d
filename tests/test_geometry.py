from __future__ import annotations

import math
import unittest

import numpy as np

from pin3d.geometry import (
    chamfer_corner_distance,
    convex_hull,
    ensure_ccw,
    min_area_rect,
    points_in_polygon,
    polygon_area,
    polygon_centroid,
    polygon_iou,
    polygon_overlap_area,
    rectilinear_score,
    rectify_polygon,
    simplify_collinear,
    wall_normal,
)


RECT = np.array([[0.0, 0.0], [4.0, 0.0], [4.0, 3.0], [0.0, 3.0]])


class TestArea(unittest.TestCase):
    def test_area(self):
        self.assertAlmostEqual(polygon_area(RECT), 12.0)

    def test_area_is_orientation_independent(self):
        self.assertAlmostEqual(polygon_area(RECT[::-1]), 12.0)

    def test_ensure_ccw(self):
        cw = ensure_ccw(RECT[::-1])
        self.assertGreater(polygon_area(cw), 0)
        signed = 0.5 * float(
            np.dot(cw[:, 0], np.roll(cw[:, 1], -1)) - np.dot(cw[:, 1], np.roll(cw[:, 0], -1))
        )
        self.assertGreater(signed, 0)

    def test_degenerate_area(self):
        self.assertEqual(polygon_area(np.array([[0.0, 0.0], [1.0, 1.0]])), 0.0)


class TestCentroid(unittest.TestCase):
    def test_rect_centroid(self):
        np.testing.assert_allclose(polygon_centroid(RECT), [2.0, 1.5], atol=1e-9)

    def test_l_shape_centroid_inside(self):
        l = np.array([[0, 0], [4, 0], [4, 2], [2, 2], [2, 4], [0, 4]], dtype=float)
        c = polygon_centroid(l)
        self.assertTrue(points_in_polygon(c[None, :], l)[0])


class TestIou(unittest.TestCase):
    def test_identical(self):
        self.assertAlmostEqual(polygon_iou(RECT, RECT), 1.0, places=6)

    def test_disjoint(self):
        far = RECT + np.array([100.0, 100.0])
        self.assertAlmostEqual(polygon_iou(RECT, far), 0.0, places=6)

    def test_half_overlap(self):
        shifted = RECT + np.array([2.0, 0.0])
        # intersection 2x3=6, union 12+12-6=18
        self.assertAlmostEqual(polygon_iou(RECT, shifted), 6.0 / 18.0, places=6)

    def test_overlap_area(self):
        shifted = RECT + np.array([2.0, 0.0])
        self.assertAlmostEqual(polygon_overlap_area(RECT, shifted), 6.0, places=6)

    def test_chamfer_identical_is_zero(self):
        self.assertAlmostEqual(chamfer_corner_distance(RECT, RECT), 0.0, places=9)

    def test_chamfer_known_shift(self):
        # axis-aligned shift: every corner's nearest neighbour is exactly 1 m away
        self.assertAlmostEqual(chamfer_corner_distance(RECT, RECT + np.array([1.0, 0.0])), 1.0, places=6)
        # diagonal shift: nearest neighbour is sqrt(2)
        self.assertAlmostEqual(
            chamfer_corner_distance(RECT, RECT + np.array([1.0, 1.0])), 2 ** 0.5, places=6
        )


class TestHullAndRect(unittest.TestCase):
    def test_convex_hull_drops_interior(self):
        pts = np.vstack([RECT, [[2.0, 1.5]]])
        hull = convex_hull(pts)
        self.assertEqual(len(hull), 4)
        self.assertAlmostEqual(polygon_area(hull), 12.0)

    def test_min_area_rect_axis_aligned(self):
        corners, angle = min_area_rect(RECT)
        self.assertAlmostEqual(polygon_area(corners), 12.0, places=6)
        self.assertAlmostEqual(angle, 0.0, places=6)

    def test_min_area_rect_rotated(self):
        th = math.radians(30)
        r = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
        rot = RECT @ r.T
        corners, angle = min_area_rect(rot)
        self.assertAlmostEqual(polygon_area(corners), 12.0, places=4)
        # angle is normalised to [0, pi/2)
        self.assertLess(angle, math.pi / 2 + 1e-9)


class TestRectify(unittest.TestCase):
    def test_snaps_near_manhattan(self):
        jitter = np.array([[0.0, 0.0], [4.02, 0.01], [4.01, 3.02], [0.0, 2.99]])
        rect = rectify_polygon(jitter, tol=0.08)
        self.assertAlmostEqual(rectilinear_score(rect), 1.0)

    def test_low_score_for_diagonal(self):
        diag = np.array([[0.0, 0.0], [2.0, 1.0], [1.0, 2.0]])
        self.assertLess(rectilinear_score(diag), 0.5)

    def test_simplify_collinear(self):
        poly = np.array([[0.0, 0.0], [2.0, 0.0], [4.0, 0.0], [4.0, 3.0], [0.0, 3.0]])
        out = simplify_collinear(poly)
        self.assertEqual(len(out), 4)

    def test_wall_normal_points_left(self):
        n = wall_normal(np.array([0.0, 0.0]), np.array([1.0, 0.0]))
        np.testing.assert_allclose(n, [0.0, 1.0])


class TestPointInPolygon(unittest.TestCase):
    def test_inside_outside(self):
        pts = np.array([[2.0, 1.5], [5.0, 1.5], [-0.1, -0.1]])
        mask = points_in_polygon(pts, RECT)
        self.assertEqual(mask.tolist(), [True, False, False])


if __name__ == "__main__":
    unittest.main()
