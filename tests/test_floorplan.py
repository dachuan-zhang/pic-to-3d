from __future__ import annotations

import math
import unittest

import numpy as np

from pin3d.pipeline.floorplan import (
    assemble,
    assemble_auto,
    assemble_greedy,
    check_overlaps,
    parse_via,
    segment_overlap,
)
from pin3d.schema import Opening, Room


def room(rid, w, h, label=None, doors=()):
    poly = np.array([[0.0, 0.0], [w, 0.0], [w, h], [0.0, h]])
    return Room(id=rid, label=label or rid, polygon=poly, ceiling=2.6,
                openings=[Opening(wall=w_, t=t, width=0.9) for w_, t in doors])


class TestSegmentOverlap(unittest.TestCase):
    def test_full(self):
        self.assertAlmostEqual(
            segment_overlap(np.array([0.0, 0.0]), np.array([4.0, 0.0]),
                            np.array([1.0, 0.0]), np.array([3.0, 0.0])), 2.0)

    def test_partial(self):
        self.assertAlmostEqual(
            segment_overlap(np.array([0.0, 0.0]), np.array([4.0, 0.0]),
                            np.array([3.0, 0.0]), np.array([6.0, 0.0])), 1.0)

    def test_disjoint(self):
        self.assertAlmostEqual(
            segment_overlap(np.array([0.0, 0.0]), np.array([4.0, 0.0]),
                            np.array([5.0, 0.0]), np.array([6.0, 0.0])), 0.0)


class TestParseVia(unittest.TestCase):
    def test_forms(self):
        self.assertEqual(parse_via("r1:2"), ("r1", 2))
        self.assertEqual(parse_via("r1/2"), ("r1", 2))
        self.assertIsNone(parse_via(None))
        self.assertIsNone(parse_via("nope"))


class TestAssembleWithHints(unittest.TestCase):
    def test_two_rooms_share_door(self):
        a = room("a", 4.0, 3.0, doors=[(1, 0.5)])   # door on x=4 wall, mid
        b = room("b", 3.0, 3.0, doors=[(3, 0.5)])   # door on x=0 wall, mid
        plan, rep = assemble([a, b], [{"from": "a", "to": "b", "via": "a:1"}])
        self.assertEqual(len(plan.rooms), 2)
        rb = plan.room("b")
        self.assertAlmostEqual(rb.polygon[:, 0].min(), 4.0, delta=0.06)
        self.assertAlmostEqual(rb.polygon[:, 1].min(), 0.0, delta=0.06)
        self.assertEqual(check_overlaps(plan), [])

    def test_door_pairs_with_door(self):
        # b's only door must not be glued onto a's solid wall
        a = room("a", 4.0, 3.0, doors=[(1, 0.5)])
        b = room("b", 3.0, 3.0, doors=[(0, 0.5)])  # door on x=3 wall
        plan, rep = assemble_greedy([a, b])
        self.assertEqual(len(rep.edges), 1)
        self.assertEqual(rep.edges[0]["door_mismatch"], 0.0)

    def test_disconnected_room_is_reported(self):
        a = room("a", 4.0, 3.0, doors=[(1, 0.5)])
        b = room("b", 3.0, 3.0, doors=[(3, 0.5)])
        c = room("c", 2.0, 2.0, doors=[(0, 0.5)])
        plan, rep = assemble([a, b, c], [{"from": "a", "to": "b", "via": "a:1"}])
        self.assertIn("c", rep.disconnected)


class TestAssembleGreedy(unittest.TestCase):
    def test_places_adjacent_without_hints(self):
        a = room("a", 4.0, 3.0, doors=[(1, 0.5)])
        b = room("b", 3.0, 3.0, doors=[(3, 0.5)])
        plan, rep = assemble_greedy([a, b])
        self.assertEqual(len(plan.rooms), 2)
        self.assertEqual(plan.rooms[0].id, "a")  # anchor keeps identity placement
        self.assertAlmostEqual(rep.placements["a"].tx, 0.0)
        self.assertEqual(len(plan.adjacency), 1)
        self.assertAlmostEqual(plan.meta["overlap_area"], 0.0)

    def test_three_rooms_no_overlap(self):
        rooms = [
            room("a", 4.0, 3.0, doors=[(1, 0.5), (2, 0.5)]),
            room("b", 3.0, 3.0, doors=[(3, 0.5)]),
            room("c", 3.0, 2.5, doors=[(0, 0.5)]),
        ]
        plan, rep = assemble_greedy(rooms)
        self.assertEqual(len(plan.rooms), 3)
        self.assertEqual(plan.meta["overlap_area"], 0.0)


class TestAuto(unittest.TestCase):
    def test_auto_uses_greedy_without_adjacency(self):
        a = room("a", 4.0, 3.0, doors=[(1, 0.5)])
        b = room("b", 3.0, 3.0, doors=[(3, 0.5)])
        plan, rep = assemble_auto([a, b])
        self.assertEqual(rep.solver, "greedy")

    def test_auto_uses_bfs_with_adjacency(self):
        a = room("a", 4.0, 3.0, doors=[(1, 0.5)])
        b = room("b", 3.0, 3.0, doors=[(3, 0.5)])
        plan, rep = assemble_auto([a, b], [{"from": "a", "to": "b", "via": "a:1"}])
        self.assertEqual(rep.solver, "bfs")


if __name__ == "__main__":
    unittest.main()
