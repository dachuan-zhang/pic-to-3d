from __future__ import annotations

import unittest

import numpy as np

from pin3d.pipeline.real import (
    NYU_K,
    affine_fit,
    align_to_metric,
    choose_convention,
    cloud_from_depth,
    depth_metrics,
)

H, W = 48, 64
K = NYU_K.copy()


class TestCloud(unittest.TestCase):
    def test_constant_depth_is_a_plane(self):
        depth = np.full((H, W), 2.0)
        cloud = cloud_from_depth(depth, K, stride=1)
        # every point sits at 2 m in front of the camera in world Z
        np.testing.assert_allclose(cloud[:, 2], 2.0, atol=1e-6)

    def test_y_is_flipped_so_up_is_positive(self):
        # a point in the top half of the image (small v) must land at positive world Y
        depth = np.full((H, W), 1.0)
        cloud = cloud_from_depth(depth, K, stride=1)
        top = cloud[cloud[:, 1] > 0]
        self.assertGreater(len(top), 0)
        # camera-space v<cy (top of image) -> world y positive
        self.assertTrue(np.all(top[:, 1] > 0))

    def test_invalid_depth_dropped(self):
        depth = np.full((H, W), 2.0)
        depth[: H // 2] = 0.0
        cloud = cloud_from_depth(depth, K, stride=1)
        self.assertAlmostEqual(len(cloud), (H // 2) * W)

    def test_stride_reduces_points(self):
        depth = np.full((H, W), 2.0)
        a = len(cloud_from_depth(depth, K, stride=1))
        b = len(cloud_from_depth(depth, K, stride=2))
        self.assertAlmostEqual(b / a, 1 / 4, delta=0.02)


class TestAffine(unittest.TestCase):
    def test_recovers_known_transform(self):
        rng = np.random.default_rng(0)
        gt = rng.uniform(1, 5, 500)
        pred = (gt - 0.7) / 2.5
        valid = np.ones(500, dtype=bool)
        a, b = affine_fit(pred, gt, valid)
        self.assertAlmostEqual(a, 2.5, places=6)
        self.assertAlmostEqual(b, 0.7, places=6)

    def test_metrics_zero_for_affine_relation(self):
        rng = np.random.default_rng(1)
        gt = rng.uniform(1, 5, (H, W))
        pred = (gt - 1.0) / 3.0
        m = depth_metrics(pred, gt, convention="depth")
        self.assertTrue(m["ok"])
        self.assertAlmostEqual(m["abs_rel"], 0.0, places=6)
        self.assertAlmostEqual(m["delta1"], 1.0, places=6)

    def test_metrics_detect_bad_prediction(self):
        rng = np.random.default_rng(2)
        gt = rng.uniform(1, 5, (H, W))
        pred = rng.uniform(0, 1, (H, W))
        m = depth_metrics(pred, gt, convention="depth")
        self.assertGreater(m["abs_rel"], 0.2)
        self.assertLess(m["delta1"], 0.9)

    def test_disparity_convention_is_detected(self):
        rng = np.random.default_rng(3)
        gt = rng.uniform(1, 4, (H, W))
        pred = 1.0 / gt  # a perfect disparity output
        pairs = [(pred, gt)] * 3
        conv, scores = choose_convention(pairs)
        self.assertEqual(conv, "disparity")
        self.assertLess(scores["disparity"], scores["depth"])

    def test_depth_convention_is_detected(self):
        rng = np.random.default_rng(4)
        gt = rng.uniform(1, 4, (H, W))
        pred = 2.0 * gt + 3.0
        conv, _ = choose_convention([(pred, gt)] * 3)
        self.assertEqual(conv, "depth")

    def test_align_to_metric_matches_gt_scale(self):
        rng = np.random.default_rng(5)
        gt = rng.uniform(1, 4, (H, W))
        pred = 3.0 * gt + 1.0
        aligned = align_to_metric(pred, gt, convention="depth")
        np.testing.assert_allclose(aligned, gt, rtol=1e-4, atol=1e-3)

    def test_empty_valid_mask_is_not_ok(self):
        gt = np.zeros((H, W))
        pred = np.ones((H, W))
        self.assertFalse(depth_metrics(pred, gt)["ok"])


if __name__ == "__main__":
    unittest.main()
