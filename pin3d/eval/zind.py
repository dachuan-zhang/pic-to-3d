"""ZInD annotation adapter.

**Caveat, read before trusting this:** the ZInD ``zind_data.json`` schema is being reproduced
here from memory of the published toolbox, not from a downloaded copy. The adapter is
therefore deliberately defensive -- it never KeyErrors, it reports exactly which keys it found,
and it ignores anything it does not understand. Validate it against your actual download
before drawing conclusions; the fix is usually one key name.

Assumed shape (per home)::

    {"floorplan": {
        "layout":           [[[x, y], ...], ...],   # one vertex list per room, image coords
        "room_labels":      ["living room", ...],
        "ceiling_height":   [255, ...],             # cm (auto-detected if values look like m)
        "doors":            [{"wall_room_pair": [[room, wall], [room, wall]]}, ...],
        "windows":          [...]
    }}

Image coordinates are y-down and in ``units_per_meter`` (ZInD is centimetre-ish); they are
converted to metres with y-up so the result is directly comparable to our predicted plans.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..geometry import ensure_ccw, wall_segments
from ..schema import Floorplan, Opening, Room

KNOWN_KEYS = ("layout", "room_labels", "ceiling_height", "doors", "windows", "id")


def _unwrap(obj: Dict[str, Any]) -> Dict[str, Any]:
    return obj.get("floorplan", obj) if isinstance(obj, dict) else obj


def describe(obj: Dict[str, Any]) -> str:
    fp = _unwrap(obj)
    keys = sorted(fp.keys()) if isinstance(fp, dict) else []
    found = [k for k in KNOWN_KEYS if k in keys]
    return f"floorplan keys={keys} known={found}"


def _ceiling_to_metres(values: List[Any], default: float = 2.7) -> List[float]:
    vals: List[float] = []
    for v in values:
        try:
            vals.append(float(v))
        except (TypeError, ValueError):
            vals.append(default)
    if not vals:
        return [default]
    # auto-detect cm vs m
    if float(np.median(vals)) > 10.0:
        vals = [v / 100.0 for v in vals]
    return [min(max(v, 1.5), 6.0) for v in vals]


def _wall_index_at(poly: np.ndarray, axis: int, at: float, tol: float = 0.05) -> Optional[int]:
    for i, (p0, p1) in enumerate(wall_segments(poly)):
        if abs(p0[axis] - at) < tol and abs(p1[axis] - at) < tol:
            return i
    return None


def floorplan_from_zind(
    obj: Dict[str, Any],
    fp_id: str = "zind",
    units_per_meter: float = 100.0,
    include_doors: bool = False,
) -> Floorplan:
    """Convert one ZInD home/floorplan annotation into our canonical ``Floorplan``.

    ``include_doors=False`` keeps ZInD's door *pairings* as adjacency but does not fabricate
    ``Opening`` objects: ZInD annotates which walls share a door, not where along the wall it
    sits, and inventing a position would silently corrupt door precision/recall.
    """
    fp = _unwrap(obj)
    if not isinstance(fp, dict) or "layout" not in fp:
        raise ValueError(
            f"not a recognisable ZInD floorplan; {describe(obj)}. "
            "Pass the dict for a single floorplan, or adapt the key names here."
        )

    layout = fp.get("layout") or []
    labels = list(fp.get("room_labels") or [])
    ceilings = _ceiling_to_metres(list(fp.get("ceiling_height") or []))

    rooms: List[Room] = []
    for i, verts in enumerate(layout):
        pts = np.asarray(verts, dtype=float).reshape(-1, 2)
        if len(pts) < 3:
            continue
        pts = np.stack([pts[:, 0] / units_per_meter, -pts[:, 1] / units_per_meter], axis=1)
        pts = ensure_ccw(pts)
        rooms.append(Room(
            id=f"r{i}",
            label=str(labels[i]) if i < len(labels) else f"room_{i}",
            polygon=pts,
            ceiling=float(ceilings[i]) if i < len(ceilings) else 2.7,
            meta={"source": "zind", "index": i},
        ))

    if rooms:
        all_pts = np.vstack([r.polygon for r in rooms])
        origin = np.array([all_pts[:, 0].min(), all_pts[:, 1].min()])
        for r in rooms:
            r.polygon = r.polygon - origin

    adjacency: List[Dict[str, str]] = []
    inferred_openings = 0
    for key in ("doors", "windows"):
        for entry in fp.get(key) or []:
            pair = None
            if isinstance(entry, dict):
                pair = entry.get("wall_room_pair") or entry.get("room_pair")
            if not pair or len(pair) != 2:
                continue
            try:
                (ra, wa), (rb, wb) = pair
                ra, wa, rb, wb = int(ra), int(wa), int(rb), int(wb)
            except (TypeError, ValueError):
                continue
            if ra >= len(rooms) or rb >= len(rooms):
                continue
            if key == "doors" and ra != rb:
                adjacency.append({"from": rooms[ra].id, "to": rooms[rb].id,
                                  "via": f"{rooms[ra].id}:{wa}"})
            if include_doors:
                width = 0.9 if key == "doors" else 1.2
                for ridx, widx in ((ra, wa), (rb, wb)):
                    rooms[ridx].openings.append(Opening(
                        wall=widx, t=0.5, width=width,
                        type="door" if key == "doors" else "window",
                        confidence=0.25,  # position is NOT annotated in ZInD
                    ))
                    inferred_openings += 1

    notes = ["door openings have inferred position (t=0.5), not annotated"]
    if not adjacency and len(rooms) > 1:
        notes.append("no door pairings found -- adjacency unknown, use assemble_greedy")
    if inferred_openings == 0 and include_doors:
        notes.append("include_doors was set but no openings could be synthesised")

    return Floorplan(
        rooms=rooms,
        adjacency=adjacency,
        meta={
            "source": "zind",
            "id": str(fp.get("id", fp_id)),
            "units_per_meter": units_per_meter,
            "notes": notes,
        },
    )


def load_zind(path: str, **kwargs: Any) -> Dict[str, Floorplan]:
    """Load a ``zind_data.json``-style file into ``{floorplan_id: Floorplan}``."""
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict) or not data:
        raise ValueError(f"{path}: expected a non-empty JSON object")

    out: Dict[str, Floorplan] = {}
    for key, value in data.items():
        try:
            out[str(key)] = floorplan_from_zind(value, fp_id=str(key), **kwargs)
        except ValueError:
            continue
    if not out:
        sample = next(iter(data.values()))
        raise ValueError(
            f"{path}: could not parse any floorplan. First entry {describe(sample) if isinstance(sample, dict) else type(sample)}"
        )
    return out
