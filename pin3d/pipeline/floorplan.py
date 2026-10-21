"""Global 2D assembly: place per-room rectangles into one floorplan.

The floorplan-first decision turns house-scale reconstruction into a **pose graph**: each
room needs only (tx, ty, theta) with theta on the Manhattan lattice, and doorways between
rooms are the relative-pose constraints.

Two solvers:

* ``assemble``       -- adjacency is known (from capture order, a VLM, or ground truth).
                        BFS outward from an anchor room.
* ``assemble_greedy`` -- adjacency is *unknown*. Prim-like: repeatedly place whichever
                        (placed room, unplaced room) pair scores best. This is the honest
                        default for arbitrary listing photos, where you do not know which
                        rooms are adjacent.

Both brute force over the 4 yaw options x candidate door pairings, scored by
(wall normal antiparallelism + door width match + interpenetration). Not yet a global
least-squares solve (see §9 item 4) -- but it is a global place-and-score pass, and it
reports its residuals so the next iteration can be judged against them.

Key geometric fact this relies on: making the two door midpoints coincide *and* requiring the
wall normals antiparallel puts both walls on the same plane. So a shared doorway is a
complete relative-pose constraint up to the yaw choice.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ..geometry import dot2, polygon_overlap_area, rot_vec, wall_normal
from ..schema import Floorplan, Opening, Room

YAW_OPTIONS = (0.0, math.pi / 2, math.pi, 3 * math.pi / 2)

#: score weights -- overlap dominates, because two plausible rooms in the wrong place is the
#: most common silent failure. Contact rewards a shared wall, without which rooms that have no
#: detected doors drift along the wall (a degenerate score).
W_NORMAL, W_WIDTH, W_OVERLAP, W_CONTACT, W_DOOR = 3.0, 2.0, 5.0, 1.0, 4.0


@dataclass
class Placement:
    tx: float = 0.0
    ty: float = 0.0
    theta: float = 0.0
    score: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "tx": round(self.tx, 4),
            "ty": round(self.ty, 4),
            "theta_deg": round(math.degrees(self.theta) % 360, 1),
            "score": round(self.score, 4) if math.isfinite(self.score) else -1.0,
        }


@dataclass
class AssemblyReport:
    placements: Dict[str, Placement] = field(default_factory=dict)
    edges: List[Dict[str, Any]] = field(default_factory=list)
    disconnected: List[str] = field(default_factory=list)
    overlaps: List[Dict[str, Any]] = field(default_factory=list)
    solver: str = "bfs"

    def format(self) -> str:
        lines = [f"assembly report ({self.solver})"]
        for rid, p in self.placements.items():
            lines.append(
                f"  {rid:<10} t=({p.tx:7.2f},{p.ty:7.2f}) yaw={math.degrees(p.theta) % 360:5.1f}"
                f" score={p.score:7.3f}"
            )
        for e in self.edges:
            lines.append(
                f"  edge {e['from']}->{e['to']} via {e.get('via')}: "
                f"normal_err={e['normal_err']:.3f} width_err={e['width_err']:.3f} "
                f"overlap={e['overlap']:.3f} m2 contact={e.get('contact', 0.0):.2f} m"
            )
        if self.disconnected:
            lines.append(f"  DISCONNECTED (fallback placement): {self.disconnected}")
        for o in self.overlaps:
            lines.append(f"  OVERLAP {o['a']}<->{o['b']}: {o['area']:.3f} m2")
        if not self.overlaps:
            lines.append("  no interpenetration")
        return "\n".join(lines)


def parse_via(via: Optional[str]) -> Optional[Tuple[str, int]]:
    """Parse an adjacency descriptor like ``"r1:2"`` into (room_id, wall_index)."""
    if not via:
        return None
    text = str(via).replace("/", ":").replace("#", ":")
    parts = [p for p in text.split(":") if p]
    if len(parts) < 2:
        return None
    try:
        return parts[0], int(parts[1])
    except ValueError:
        return None


def _opening_for_wall(room: Room, wall: int) -> Optional[Opening]:
    n = len(room.polygon)
    if n == 0:
        return None
    candidates = [o for o in room.openings if o.wall == wall % n]
    return max(candidates, key=lambda o: o.confidence) if candidates else None


def _wall_frame(room: Room, wall: int) -> Tuple[np.ndarray, np.ndarray, float]:
    p0, p1 = room.wall(wall)
    d = p1 - p0
    length = float(np.linalg.norm(d))
    return wall_normal(p0, p1), d / length if length > 1e-9 else np.array([1.0, 0.0]), length


def _overlap_with_placed(poly: np.ndarray, placed: Iterable[Room]) -> float:
    return float(sum(polygon_overlap_area(poly, r.polygon) for r in placed))


def segment_overlap(a0: np.ndarray, a1: np.ndarray, b0: np.ndarray, b1: np.ndarray) -> float:
    """Length of the collinear overlap of segment AB projected onto segment A."""
    d = np.asarray(a1, float) - np.asarray(a0, float)
    length = float(np.linalg.norm(d))
    if length < 1e-9:
        return 0.0
    u = d / length
    ta = (0.0, length)
    tb = tuple(sorted((dot2(np.asarray(b0, float) - a0, u),
                       dot2(np.asarray(b1, float) - a0, u))))
    return max(0.0, min(ta[1], tb[1]) - max(ta[0], tb[0]))


def _best_placement(
    anchor: Room,
    local: Room,
    a_walls: Sequence[int],
    placed_rooms: Sequence[Room],
    snap_step: float = 0.05,
) -> Optional[Tuple[Room, Placement, Dict[str, Any]]]:
    """Best (yaw, door pairing, anchor wall) placing ``local`` next to ``anchor``."""
    if len(local.polygon) < 3:
        return None

    if local.openings:
        b_candidates: List[Tuple[int, float]] = [(o.wall, o.t) for o in local.openings]
        b_widths = {o.wall: o.width for o in local.openings}
    else:
        b_candidates = [(i, 0.5) for i in range(len(local.polygon))]
        b_widths = {}

    best: Optional[Tuple[float, Room, Placement, Dict[str, Any]]] = None
    for a_wall in a_walls:
        a_open = _opening_for_wall(anchor, a_wall)
        a_normal, _, _ = _wall_frame(anchor, a_wall)
        a_p0, a_p1 = anchor.wall(a_wall)
        if a_open is not None:
            a_target = anchor.opening_midpoint(a_open)
            a_width: Optional[float] = a_open.width
        else:
            a_target = anchor.wall_midpoint(a_wall)
            a_width = None

        for theta in YAW_OPTIONS:
            c, s = math.cos(theta), math.sin(theta)
            rot = np.array([[c, -s], [s, c]])
            for b_wall, b_t in b_candidates:
                b0, b1 = local.wall(b_wall)
                b_local = b0 + float(np.clip(b_t, 0.0, 1.0)) * (b1 - b0)
                b_rot = rot_vec(rot, b_local)
                tx, ty = float(a_target[0] - b_rot[0]), float(a_target[1] - b_rot[1])
                cand = local.transformed(theta, tx, ty)

                b_normal, _, _ = _wall_frame(local, b_wall)
                b_normal_world = rot_vec(rot, b_normal)
                normal_err = 1.0 + dot2(b_normal_world, a_normal)  # 0 when antiparallel
                width_err = abs(a_width - b_widths[b_wall]) if (a_width is not None and b_wall in b_widths) else 0.0
                # A doorway must pair with a doorway. Without this, a room with a detected
                # door glues that door onto a solid wall whenever contact happens to be
                # marginally longer -- geometrically nonsense that still scores well.
                door_mismatch = 0.0 if (a_open is not None) == (b_wall in b_widths) else 1.0
                overlap = _overlap_with_placed(cand.polygon, placed_rooms)
                b_p0, b_p1 = cand.wall(b_wall)
                contact = segment_overlap(a_p0, a_p1, b_p0, b_p1)
                score = (W_NORMAL * normal_err + W_WIDTH * width_err
                         + W_OVERLAP * overlap + W_DOOR * door_mismatch
                         - W_CONTACT * contact)

                if best is None or score < best[0] - 1e-9:
                    best = (score, cand, Placement(tx, ty, theta, score), {
                        "anchor_wall": a_wall,
                        "local_wall": b_wall,
                        "normal_err": normal_err,
                        "width_err": width_err,
                        "overlap": overlap,
                        "contact": contact,
                        "door_mismatch": door_mismatch,
                    })

    if best is None:
        return None
    score, cand, placement, info = best
    if snap_step > 0:
        cand.polygon = np.round(cand.polygon / snap_step) * snap_step
    return cand, placement, info


def _record(
    report: AssemblyReport, from_id: str, to_id: str, via: Optional[str], info: Dict[str, Any]
) -> None:
    report.edges.append({
        "from": from_id, "to": to_id, "via": via,
        "normal_err": round(info["normal_err"], 4),
        "width_err": round(info["width_err"], 4),
        "overlap": round(info["overlap"], 4),
        "contact": round(info.get("contact", 0.0), 4),
        "door_mismatch": round(info.get("door_mismatch", 0.0), 3),
    })


def _finalise(
    world: Dict[str, Room],
    rooms: Sequence[Room],
    adjacency: Sequence[Dict[str, str]],
    report: AssemblyReport,
    snap_step: float,
    fallback_gap: float,
) -> Floorplan:
    """Park any unplaced room, then compute overlaps and build the Floorplan."""
    for r in rooms:
        if r.id in world:
            continue
        if world:
            right = max(rm.polygon[:, 0].max() for rm in world.values()) + fallback_gap
            floor = min(rm.polygon[:, 1].min() for rm in world.values())
            poly = r.polygon - np.array([r.polygon[:, 0].min(), r.polygon[:, 1].min()])
            placed = Room.from_dict(r.to_dict())
            placed.polygon = poly + np.array([right, floor])
        else:  # pragma: no cover - defensive
            placed = r
        world[r.id] = placed
        report.placements[r.id] = Placement(float(placed.polygon[:, 0].min()),
                                            float(placed.polygon[:, 1].min()), 0.0, float("inf"))
        report.disconnected.append(r.id)

    ids = list(world)
    # Snap before the overlap check: snapping after it can introduce interpenetration.
    if snap_step > 0:
        for room in world.values():
            room.polygon = np.round(room.polygon / snap_step) * snap_step

    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            area = polygon_overlap_area(world[ids[i]].polygon, world[ids[j]].polygon)
            if area > 0.01:
                report.overlaps.append({"a": ids[i], "b": ids[j], "area": round(area, 4)})

    for rid, room in world.items():
        room.meta = {**room.meta, "placement": report.placements[rid].to_dict()}

    return Floorplan(
        rooms=[world[rid] for rid in ids],
        adjacency=list(adjacency),
        meta={
            "method": f"box_fit+door_constrained_{report.solver}",
            "rooms": len(ids),
            "overlap_area": round(sum(o["area"] for o in report.overlaps), 4),
            "edges": len(report.edges),
        },
    )


def assemble(
    rooms: Sequence[Room],
    adjacency: Optional[Sequence[Dict[str, str]]] = None,
    snap_step: float = 0.05,
    anchor: Optional[str] = None,
    fallback_gap: float = 0.30,
) -> Tuple[Floorplan, AssemblyReport]:
    """Place rooms using a known adjacency (BFS outward from an anchor)."""
    if not rooms:
        return Floorplan(), AssemblyReport()
    by_id = {r.id: r for r in rooms}
    adjacency = list(adjacency or [])

    neighbours: Dict[str, List[Dict[str, Any]]] = {rid: [] for rid in by_id}
    for edge in adjacency:
        a, b = edge.get("from"), edge.get("to")
        if a in by_id and b in by_id:
            neighbours[a].append({"to": b, "via": edge.get("via")})
            neighbours[b].append({"to": a, "via": edge.get("via"), "reverse": True})

    start = anchor if anchor in by_id else rooms[0].id
    world: Dict[str, Room] = {start: by_id[start].transformed(0.0, 0.0, 0.0)}
    report = AssemblyReport(placements={start: Placement(0.0, 0.0, 0.0, 0.0)}, solver="bfs")

    queue: List[str] = [start]
    while queue:
        current = queue.pop(0)
        for nb in neighbours.get(current, []):
            other = nb["to"]
            if other in world:
                continue
            via = nb.get("via")
            a_walls = [0]
            parsed = parse_via(via)
            if parsed:
                room_id, wall = parsed
                if nb.get("reverse"):
                    op = _opening_for_wall(by_id[current], wall)
                    a_walls = [op.wall if op is not None else wall]
                elif room_id == current:
                    a_walls = [wall]
            result = _best_placement(world[current], by_id[other], a_walls,
                                     list(world.values()), snap_step)
            if result is None:
                continue
            cand, placement, info = result
            world[other] = cand
            report.placements[other] = placement
            _record(report, current, other, via, info)
            queue.append(other)

    fp = _finalise(world, rooms, adjacency, report, snap_step, fallback_gap)
    return fp, report


def assemble_greedy(
    rooms: Sequence[Room],
    snap_step: float = 0.05,
    anchor: Optional[str] = None,
    fallback_gap: float = 0.30,
) -> Tuple[Floorplan, AssemblyReport]:
    """Place rooms with **unknown** adjacency: repeatedly take the globally best pairing.

    This is the honest default for arbitrary listing photos: you usually do not know which
    rooms are adjacent, but you can score every candidate placement.
    """
    if not rooms:
        return Floorplan(), AssemblyReport()
    by_id = {r.id: r for r in rooms}
    start = anchor if anchor in by_id else rooms[0].id

    world: Dict[str, Room] = {start: by_id[start].transformed(0.0, 0.0, 0.0)}
    report = AssemblyReport(placements={start: Placement(0.0, 0.0, 0.0, 0.0)}, solver="greedy")
    remaining = [r for r in rooms if r.id != start]

    while remaining:
        best: Optional[Tuple[float, str, Room, Room, Placement, Dict[str, Any]]] = None
        for a_id, a_room in world.items():
            a_walls = list(range(len(a_room.polygon)))
            for b_room in remaining:
                result = _best_placement(a_room, b_room, a_walls,
                                         list(world.values()), snap_step)
                if result is None:
                    continue
                cand, placement, info = result
                if best is None or placement.score < best[0] - 1e-9:
                    best = (placement.score, a_id, a_room, cand, placement, info)
        if best is None:
            break
        _, a_id, a_room, cand, placement, info = best
        # `via` is unknown here; record the wall pair that won so a human can audit it.
        via = f"{a_id}:{info['anchor_wall']}"
        world[cand.id] = cand
        report.placements[cand.id] = placement
        _record(report, a_id, cand.id, via, info)
        remaining = [r for r in remaining if r.id != cand.id]

    # The solver's winning pairings *are* the inferred topology; persist them.
    inferred = [{"from": e["from"], "to": e["to"], "via": e["via"]} for e in report.edges]
    fp = _finalise(world, rooms, inferred, report, snap_step, fallback_gap)
    fp.meta["adjacency"] = "inferred"
    return fp, report


def assemble_auto(
    rooms: Sequence[Room],
    adjacency: Optional[Sequence[Dict[str, str]]] = None,
    snap_step: float = 0.05,
    anchor: Optional[str] = None,
) -> Tuple[Floorplan, AssemblyReport]:
    if adjacency:
        return assemble(rooms, adjacency, snap_step=snap_step, anchor=anchor)
    return assemble_greedy(rooms, snap_step=snap_step, anchor=anchor)


def check_overlaps(plan: Floorplan) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for i, a in enumerate(plan.rooms):
        for b in plan.rooms[i + 1:]:
            area = polygon_overlap_area(a.polygon, b.polygon)
            if area > 0.01:
                out.append({"a": a.id, "b": b.id, "area": round(area, 4)})
    return out
