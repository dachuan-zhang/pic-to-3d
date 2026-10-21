"""Canonical floorplan artifact: the database, eval target, editor model and navmesh input.

Deliberately dumb schema (see docs/BRAINSTORM.md 11): explicit polygons, explicit wall
indices, no implicit derivation. Errors must be cheap to repair by hand.

Units are metres, polygons are CCW, `Opening.t` is a fraction 0..1 along the wall from
its start vertex, `Opening.wall` indexes into the room's polygon edges.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

Vec2 = Tuple[float, float]


@dataclass
class Opening:
    wall: int
    t: float
    width: float
    type: str = "door"  # door | window | passage
    head: float = 2.05
    sill: float = 0.0
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "wall": int(self.wall),
            "t": round(float(self.t), 5),
            "width": round(float(self.width), 3),
            "type": self.type,
            "head": round(float(self.head), 3),
            "sill": round(float(self.sill), 3),
            "confidence": round(float(self.confidence), 3),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Opening":
        return cls(
            wall=int(d["wall"]),
            t=float(d["t"]),
            width=float(d["width"]),
            type=str(d.get("type", "door")),
            head=float(d.get("head", 2.05)),
            sill=float(d.get("sill", 0.0)),
            confidence=float(d.get("confidence", 1.0)),
        )


@dataclass
class Room:
    id: str = ""
    label: str = "unknown"
    polygon: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), dtype=float))
    ceiling: float = 2.7
    photos: List[str] = field(default_factory=list)
    openings: List[Opening] = field(default_factory=list)
    confidence: float = 1.0
    # free-form provenance, e.g. {"method": "box_fit", "angle": 0.0}
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.polygon = np.asarray(self.polygon, dtype=float).reshape(-1, 2)

    # -- geometry helpers -------------------------------------------------
    @property
    def area(self) -> float:
        return polygon_area(self.polygon)

    @property
    def centroid(self) -> np.ndarray:
        return polygon_centroid(self.polygon)

    def walls(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        return wall_segments(self.polygon)

    def wall(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        return self.walls()[index % len(self.polygon)]

    def wall_midpoint(self, index: int) -> np.ndarray:
        p0, p1 = self.wall(index)
        return 0.5 * (p0 + p1)

    def opening_midpoint(self, op: Opening) -> np.ndarray:
        p0, p1 = self.wall(op.wall)
        return p0 + float(np.clip(op.t, 0.0, 1.0)) * (p1 - p0)

    def transformed(self, theta: float, tx: float, ty: float) -> "Room":
        """Rigid 2D transform about the room origin, then translate."""
        c, s = math.cos(theta), math.sin(theta)
        r = np.array([[c, -s], [s, c]], dtype=float)
        out = Room(
            id=self.id,
            label=self.label,
            polygon=rot_points(self.polygon, r) + np.array([tx, ty], dtype=float),
            ceiling=self.ceiling,
            photos=list(self.photos),
            openings=[Opening(**{**op.__dict__}) for op in self.openings],
            confidence=self.confidence,
            meta=dict(self.meta),
        )
        return out

    # -- serialisation ----------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "polygon": [[round(float(x), 4), round(float(y), 4)] for x, y in self.polygon],
            "ceiling": round(float(self.ceiling), 3),
            "photos": list(self.photos),
            "openings": [op.to_dict() for op in self.openings],
            "confidence": round(float(self.confidence), 3),
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Room":
        return cls(
            id=str(d["id"]),
            label=str(d.get("label", "unknown")),
            polygon=np.asarray(d.get("polygon", []), dtype=float),
            ceiling=float(d.get("ceiling", 2.7)),
            photos=list(d.get("photos", [])),
            openings=[Opening.from_dict(o) for o in d.get("openings", [])],
            confidence=float(d.get("confidence", 1.0)),
            meta=dict(d.get("meta", {})),
        )


@dataclass
class Floorplan:
    units: str = "meters"
    scale: float = 1.0
    rooms: List[Room] = field(default_factory=list)
    adjacency: List[Dict[str, str]] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    # -- access -----------------------------------------------------------
    def room(self, room_id: str) -> Room:
        for r in self.rooms:
            if r.id == room_id:
                return r
        raise KeyError(room_id)

    @property
    def room_ids(self) -> List[str]:
        return [r.id for r in self.rooms]

    def adjacency_pairs(self) -> set:
        return {frozenset((e["from"], e["to"])) for e in self.adjacency}

    def bounds(self) -> Tuple[float, float, float, float]:
        if not self.rooms:
            return (0.0, 0.0, 0.0, 0.0)
        pts = np.vstack([r.polygon for r in self.rooms if len(r.polygon)])
        return (
            float(pts[:, 0].min()),
            float(pts[:, 1].min()),
            float(pts[:, 0].max()),
            float(pts[:, 1].max()),
        )

    # -- validation -------------------------------------------------------
    def validate(self) -> List[str]:
        problems: List[str] = []
        seen = set()
        for r in self.rooms:
            if not r.id:
                problems.append("room with empty id")
            if r.id in seen:
                problems.append(f"duplicate room id {r.id!r}")
            seen.add(r.id)
            if len(r.polygon) < 3:
                problems.append(f"{r.id}: polygon has {len(r.polygon)} vertices (<3)")
                continue
            if abs(_signed_area(r.polygon)) < 0.1:
                problems.append(f"{r.id}: degenerate polygon (area {r.area:.3f} m2)")
            if not (1.5 <= r.ceiling <= 6.0):
                problems.append(f"{r.id}: implausible ceiling {r.ceiling:.2f} m")
            n_walls = len(r.polygon)
            for op in r.openings:
                if not (0 <= op.wall < n_walls):
                    problems.append(f"{r.id}: opening wall index {op.wall} out of range")
                if not (0.0 <= op.t <= 1.0):
                    problems.append(f"{r.id}: opening t={op.t} outside [0,1]")
                if op.width <= 0:
                    problems.append(f"{r.id}: opening width {op.width} <= 0")
        for e in self.adjacency:
            for key in ("from", "to"):
                if e.get(key) not in seen:
                    problems.append(f"adjacency references unknown room {e.get(key)!r}")
        return problems

    # -- serialisation ----------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "units": self.units,
            "scale": self.scale,
            "rooms": [r.to_dict() for r in self.rooms],
            "adjacency": list(self.adjacency),
            "meta": self.meta,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=False)

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(self.to_json())
            fh.write("\n")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Floorplan":
        return cls(
            units=str(d.get("units", "meters")),
            scale=float(d.get("scale", 1.0)),
            rooms=[Room.from_dict(r) for r in d.get("rooms", [])],
            adjacency=[dict(e) for e in d.get("adjacency", [])],
            meta=dict(d.get("meta", {})),
        )

    @classmethod
    def load(cls, path: str) -> "Floorplan":
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))

    def summary(self) -> str:
        total = sum(r.area for r in self.rooms)
        lines = [f"{len(self.rooms)} room(s), {total:.1f} m2, {len(self.adjacency)} edge(s)"]
        for r in self.rooms:
            w = max((len(p) for p in [r.polygon]), default=0)
            lines.append(
                f"  {r.id:<8} {r.label:<12} {r.area:6.2f} m2  "
                f"ceil {r.ceiling:.2f}  {w}-gon  doors {len(r.openings)}"
            )
        return "\n".join(lines)


# Imported late to keep the dataclass module import-light.
from .geometry import (  # noqa: E402
    _signed_area,
    polygon_area,
    polygon_centroid,
    rot_points,
    wall_segments,
)
