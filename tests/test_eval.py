from __future__ import annotations

import unittest

import numpy as np

from pin3d.eval.metrics import aggregate, evaluate, match_rooms
from pin3d.schema import Floorplan, Opening, Room


def make_plan(offset=0.0, with_door=True, ceiling=2.7) -> Floorplan:
    a = Room(id="a", label="living", ceiling=ceiling,
             polygon=np.array([[0.0, 0.0], [4.0, 0.0], [4.0, 3.0], [0.0, 3.0]]) + offset,
             openings=[Opening(wall=1, t=0.5, width=0.9)] if with_door else [])
    b = Room(id="b", label="hall", ceiling=ceiling,
             polygon=np.array([[4.0, 0.0], [6.0, 0.0], [6.0, 3.0], [4.0, 3.0]]) + offset,
             openings=[Opening(wall=3, t=0.5, width=0.9)] if with_door else [])
    return Floorplan(rooms=[a, b], adjacency=[{"from": "a", "to": "b", "via": "a:1"}])


class TestMatching(unittest.TestCase):
    def test_identical_matches(self):
        m = match_rooms(make_plan(), make_plan())
        self.assertEqual(len(m), 2)
        self.assertTrue(all(iou > 0.999 for _, _, iou in m))

    def test_far_apart_does_not_match(self):
        self.assertEqual(match_rooms(make_plan(), make_plan(offset=50.0)), [])


class TestEvaluate(unittest.TestCase):
    def test_perfect(self):
        m = evaluate(make_plan(), make_plan())
        self.assertAlmostEqual(m["mean_iou"], 1.0, places=4)
        self.assertAlmostEqual(m["corner_dist_m"], 0.0, places=6)
        self.assertAlmostEqual(m["door_f1"], 1.0, places=4)
        self.assertAlmostEqual(m["adjacency_f1"], 1.0, places=4)
        self.assertEqual(m["self_overlap_m2"], 0.0)

    def test_missing_room_costs_recall(self):
        pred = make_plan()
        pred.rooms = pred.rooms[:1]
        m = evaluate(pred, make_plan())
        self.assertAlmostEqual(m["matched"], 1)
        self.assertAlmostEqual(m["mean_iou_over_gt"], 0.5, delta=0.02)
        self.assertIn("b", m["unmatched_gt"])

    def test_wrong_ceiling(self):
        m = evaluate(make_plan(ceiling=2.5), make_plan(ceiling=2.7))
        self.assertAlmostEqual(m["ceiling_mae_m"], 0.2, places=4)

    def test_missing_doors(self):
        m = evaluate(make_plan(with_door=False), make_plan())
        self.assertAlmostEqual(m["door_recall"], 0.0)
        self.assertAlmostEqual(m["doors_pred"], 0)

    def test_spurious_doors(self):
        pred = make_plan()
        for r in pred.rooms:
            r.openings.append(Opening(wall=0, t=0.2, width=0.9))
        m = evaluate(pred, make_plan())
        self.assertLess(m["door_precision"], 1.0)
        self.assertAlmostEqual(m["door_recall"], 1.0)

    def test_interpenetration_is_reported(self):
        pred = make_plan()
        pred.rooms[1].polygon = pred.rooms[1].polygon - np.array([2.0, 0.0])
        m = evaluate(pred, make_plan())
        self.assertGreater(m["self_overlap_m2"], 1.0)

    def test_adjacency_mismatch(self):
        pred = make_plan()
        pred.adjacency = []
        m = evaluate(pred, make_plan())
        self.assertAlmostEqual(m["adjacency_f1"], 0.0)


class TestAggregate(unittest.TestCase):
    def test_means(self):
        runs = [evaluate(make_plan(), make_plan()) for _ in range(3)]
        agg = aggregate(runs)
        self.assertEqual(agg["n_runs"], 3)
        self.assertAlmostEqual(agg["mean_iou"], 1.0, places=4)
        self.assertAlmostEqual(agg["room_recall"], 1.0, places=4)

    def test_empty(self):
        self.assertEqual(aggregate([]), {})


if __name__ == "__main__":
    unittest.main()
