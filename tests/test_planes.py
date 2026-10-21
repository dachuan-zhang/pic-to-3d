from __future__ import annotations

import unittest

import numpy as np

from pin3d.pipeline.openings import GapOpeningDetector, detect_gap_openings
from pin3d.pipeline.planes import (
    density_filter,
    detect_wall_segments,
    estimate_floor_ceiling,
    fit_room_box,
    wall_plane_bounds,
)

from .helpers import make_room_cloud

LO = (0.0, 0.0, 0.0)
HI = (3.6, 2.6, 4.0)


class TestWallPlane(unittest.TestCase):
    def test_outer_median_beats_minmax_under_noise(self):
        # spike at 0 and 3.6, uniform interior: the exact case that broke percentiles
        rng = np.random.default_rng(1)
        spike = np.concatenate([rng.normal(0, 0.012, 500), rng.normal(3.6, 0.012, 500)])
        interior = rng.uniform(0, 3.6, 400)
        v = np.concatenate([spike, interior])
        lo, hi = wall_plane_bounds(v)
        self.assertAlmostEqual(lo, 0.0, delta=0.01)
        self.assertAlmostEqual(hi, 3.6, delta=0.01)


class TestDensityFilter(unittest.TestCase):
    def test_drops_isolated_points(self):
        rng = np.random.default_rng(2)
        wall = np.tile(np.linspace(0, 4, 200)[:, None], (1, 2)) + rng.normal(0, 0.005, (200, 2))
        flyers = rng.uniform(0, 40, (30, 2))
        pts = np.vstack([wall, flyers])
        kept = density_filter(pts)
        self.assertLess(len(kept), len(pts))
        self.assertLess(kept[:, 0].max(), 5.0)


class TestFitRoomBox(unittest.TestCase):
    def test_recovers_known_room(self):
        cloud = make_room_cloud(LO, HI, seed=3)
        box = fit_room_box(cloud)
        self.assertIsNotNone(box)
        w, h = box.size
        self.assertAlmostEqual(w, 3.6, delta=0.05)
        self.assertAlmostEqual(h, 4.0, delta=0.05)
        self.assertAlmostEqual(box.ceiling, 2.6, delta=0.05)
        self.assertGreater(box.confidence, 0.9)

    def test_survives_outliers(self):
        cloud = make_room_cloud(LO, HI, outliers=0.02, seed=4)
        box = fit_room_box(cloud)
        w, h = box.size
        self.assertAlmostEqual(w, 3.6, delta=0.10)
        self.assertAlmostEqual(h, 4.0, delta=0.10)

    def test_too_few_points(self):
        self.assertIsNone(fit_room_box(np.zeros((5, 3))))

    def test_floor_ceiling(self):
        cloud = make_room_cloud(LO, HI, seed=5)
        floor, ceiling = estimate_floor_ceiling(cloud)
        self.assertAlmostEqual(floor, 0.0, delta=0.02)
        self.assertIsNotNone(ceiling)
        self.assertAlmostEqual(ceiling, 2.6, delta=0.03)

    def test_no_ceiling_detected(self):
        cloud = make_room_cloud((0, 0, 0), (3.6, 1.2, 4.0), seed=6)
        _, ceiling = estimate_floor_ceiling(cloud)
        self.assertIsNone(ceiling)


class TestOpenings(unittest.TestCase):
    def test_detects_single_door(self):
        cloud = make_room_cloud(LO, HI, gap=(0, 3.6, 1.2, 2.1), seed=7)
        box = fit_room_box(cloud)
        ops = detect_gap_openings(cloud, box.polygon, floor_y=box.floor_y, ceiling=box.ceiling)
        self.assertEqual(len(ops), 1)
        op = ops[0]
        self.assertAlmostEqual(op.width, 0.9, delta=0.12)
        self.assertEqual(op.type, "door")

    def test_no_opening_in_solid_room(self):
        cloud = make_room_cloud(LO, HI, seed=8)
        box = fit_room_box(cloud)
        ops = detect_gap_openings(cloud, box.polygon, floor_y=box.floor_y, ceiling=box.ceiling)
        self.assertEqual(ops, [])

    def test_corner_gap_is_not_a_door(self):
        # a missing corner (wall not fully seen) must not become a doorway
        cloud = make_room_cloud(LO, (3.6, 2.6, 4.0), gap=(2, 0.0, 0.0, 0.7), seed=9)
        box = fit_room_box(cloud)
        ops = detect_gap_openings(cloud, box.polygon, floor_y=box.floor_y, ceiling=box.ceiling)
        self.assertEqual(ops, [])

    def test_detector_object_matches_function(self):
        cloud = make_room_cloud(LO, HI, gap=(0, 3.6, 1.2, 2.1), seed=10)
        box = fit_room_box(cloud)
        det = GapOpeningDetector()
        a = det.detect(cloud, box.polygon, box.floor_y, box.ceiling)
        b = detect_gap_openings(cloud, box.polygon, floor_y=box.floor_y, ceiling=box.ceiling)
        self.assertEqual(len(a), len(b))


class TestWallSegments(unittest.TestCase):
    def test_finds_wall_lines(self):
        cloud = make_room_cloud(LO, HI, seed=11)
        segs = detect_wall_segments(cloud, min_inliers=30)
        self.assertGreaterEqual(len(segs), 3)
        self.assertTrue(all(s.length > 0.5 for s in segs))


if __name__ == "__main__":
    unittest.main()
