from __future__ import annotations

import json
import os
import tempfile
import unittest

import numpy as np

from pin3d.schema import Floorplan, Opening, Room


def sample() -> Floorplan:
    a = Room(
        id="r1", label="kitchen",
        polygon=np.array([[0.0, 0.0], [4.2, 0.0], [4.2, 3.1], [0.0, 3.1]]),
        ceiling=2.7, photos=["p1.jpg"],
        openings=[Opening(wall=0, t=0.42, width=0.9)],
    )
    b = Room(
        id="r2", label="hall",
        polygon=np.array([[4.2, 0.0], [6.0, 0.0], [6.0, 3.1], [4.2, 3.1]]),
        ceiling=2.7,
    )
    return Floorplan(rooms=[a, b], adjacency=[{"from": "r1", "to": "r2", "via": "r1:0"}])


class TestRoundTrip(unittest.TestCase):
    def test_json_roundtrip(self):
        plan = sample()
        back = Floorplan.from_dict(json.loads(plan.to_json()))
        self.assertEqual(back.room_ids, ["r1", "r2"])
        np.testing.assert_allclose(back.room("r1").polygon, plan.room("r1").polygon)
        self.assertEqual(back.room("r1").openings[0].wall, 0)
        self.assertEqual(back.adjacency[0]["via"], "r1:0")

    def test_save_load(self):
        plan = sample()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "fp.json")
            plan.save(path)
            back = Floorplan.load(path)
        self.assertEqual(back.room_ids, plan.room_ids)

    def test_roundtrip_of_empty(self):
        self.assertEqual(Floorplan.from_dict(Floorplan().to_dict()).rooms, [])


class TestGeometryHelpers(unittest.TestCase):
    def test_area_and_centroid(self):
        r = sample().room("r1")
        self.assertAlmostEqual(r.area, 4.2 * 3.1, places=4)
        np.testing.assert_allclose(r.centroid, [2.1, 1.55], atol=1e-6)

    def test_opening_midpoint(self):
        r = sample().room("r1")
        op = r.openings[0]
        mid = r.opening_midpoint(op)
        np.testing.assert_allclose(mid, [4.2 * 0.42, 0.0], atol=1e-6)

    def test_transformed(self):
        r = sample().room("r1")
        moved = r.transformed(0.0, 10.0, 5.0)
        np.testing.assert_allclose(moved.polygon.min(axis=0), [10.0, 5.0], atol=1e-9)
        self.assertAlmostEqual(moved.area, r.area, places=6)
        # original untouched
        np.testing.assert_allclose(r.polygon.min(axis=0), [0.0, 0.0], atol=1e-9)

    def test_bounds(self):
        lo_x, lo_y, hi_x, hi_y = sample().bounds()
        self.assertAlmostEqual(lo_x, 0.0)
        self.assertAlmostEqual(hi_x, 6.0)


class TestValidation(unittest.TestCase):
    def test_clean_plan_is_valid(self):
        self.assertEqual(sample().validate(), [])

    def test_catches_duplicate_ids(self):
        plan = sample()
        plan.rooms.append(plan.rooms[0])
        self.assertTrue(any("duplicate" in p for p in plan.validate()))

    def test_catches_bad_ceiling(self):
        plan = sample()
        plan.room("r1").ceiling = 99.0
        self.assertTrue(any("ceiling" in p for p in plan.validate()))

    def test_catches_bad_opening(self):
        plan = sample()
        plan.room("r1").openings[0].wall = 17
        self.assertTrue(any("out of range" in p for p in plan.validate()))

    def test_catches_unknown_adjacency(self):
        plan = sample()
        plan.adjacency.append({"from": "r1", "to": "ghost", "via": "r1:0"})
        self.assertTrue(any("unknown room" in p for p in plan.validate()))

    def test_catches_degenerate_polygon(self):
        plan = sample()
        plan.room("r2").polygon = np.array([[0.0, 0.0], [1.0, 1.0]])
        self.assertTrue(any("vertices" in p for p in plan.validate()))


if __name__ == "__main__":
    unittest.main()
