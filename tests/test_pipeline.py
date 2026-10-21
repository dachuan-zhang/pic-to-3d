from __future__ import annotations

import unittest

import numpy as np

from pin3d.cli import gt_from_scene
from pin3d.eval.metrics import evaluate
from pin3d.pipeline.poses import Poses, StubBackend, StubScene, get_backend
from pin3d.pipeline.reconstruct import group_clouds, reconstruct


class TestStubScene(unittest.TestCase):
    def test_clouds_are_finite_and_sized(self):
        poses = StubBackend(seed=1).estimate([])
        self.assertEqual(len(poses.clouds), 2)
        for cloud in poses.clouds.values():
            self.assertTrue(np.isfinite(cloud).all())
            self.assertGreater(len(cloud), 500)

    def test_deterministic(self):
        a = StubBackend(seed=3).estimate([]).clouds["r_living"]
        b = StubBackend(seed=3).estimate([]).clouds["r_living"]
        np.testing.assert_allclose(a, b)

    def test_group_clouds_prefers_backend_clouds(self):
        poses = StubBackend(seed=1).estimate([])
        self.assertEqual(set(group_clouds(poses)), set(poses.clouds))

    def test_get_backend(self):
        self.assertEqual(get_backend("stub").name, "stub")
        with self.assertRaises(ValueError):
            get_backend("nonsense")


class TestUnprojection(unittest.TestCase):
    def test_roundtrip(self):
        depth = np.full((4, 4), 2.0)
        k = np.array([[100.0, 0.0, 1.5], [0.0, 100.0, 1.5], [0.0, 0.0, 1.0]])
        t = np.eye(4)
        poses = Poses(image_names=["a"], K=k[None], T=t[None], depth=depth[None])
        pts = poses.unproject(0)
        self.assertEqual(len(pts), 16)
        np.testing.assert_allclose(pts[:, 2], 2.0, atol=1e-9)
        # centre pixel maps to the optical axis
        centre = pts[5]  # row 1, col 1 -> u=v=1, not centre; check with exact value instead
        u, v = 1, 1
        self.assertAlmostEqual(pts[5][0], (u - 1.5) / 100.0 * 2.0, places=9)
        self.assertAlmostEqual(pts[5][1], (v - 1.5) / 100.0 * 2.0, places=9)

    def test_translation_applied(self):
        depth = np.full((2, 2), 1.0)
        k = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        t = np.eye(4)
        t[:3, 3] = [10.0, 0.0, 0.0]
        poses = Poses(image_names=["a"], K=k[None], T=t[None], depth=depth[None])
        self.assertGreater(poses.unproject(0)[:, 0].min(), 9.5)

    def test_pointcloud_without_depth_or_clouds_is_empty(self):
        poses = Poses(image_names=["a"])
        self.assertEqual(len(poses.pointcloud()), 0)


class TestEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scene = StubScene()
        cls.poses = StubBackend(scene=cls.scene, seed=7).estimate([])
        cls.result = reconstruct(cls.poses)

    def test_two_rooms_recovered(self):
        self.assertEqual(len(self.result.rooms), 2)

    def test_doors_found_in_both_rooms(self):
        for r in self.result.rooms:
            self.assertGreaterEqual(len(r.openings), 1, msg=f"{r.id} has no openings")

    def test_no_interpenetration(self):
        self.assertEqual(self.result.floorplan.meta["overlap_area"], 0.0)

    def test_floorplan_validates(self):
        self.assertEqual(self.result.floorplan.validate(), [])

    def test_scores_against_ground_truth(self):
        m = evaluate(self.result.floorplan, gt_from_scene(self.scene))
        self.assertAlmostEqual(m["mean_iou"], 1.0, delta=0.03)
        self.assertGreaterEqual(m["door_f1"], 0.99)
        self.assertAlmostEqual(m["adjacency_f1"], 1.0, delta=0.01)
        self.assertLessEqual(m["ceiling_mae_m"], 0.05)

    def test_single_room_cloud_does_not_crash(self):
        poses = StubBackend(seed=7).estimate([])
        poses.clouds = {"r_living": poses.clouds["r_living"]}
        res = reconstruct(poses)
        self.assertEqual(len(res.rooms), 1)

    def test_empty_scene(self):
        poses = Poses(image_names=[])
        res = reconstruct(poses)
        self.assertEqual(res.rooms, [])
        self.assertTrue(res.notes)


if __name__ == "__main__":
    unittest.main()
