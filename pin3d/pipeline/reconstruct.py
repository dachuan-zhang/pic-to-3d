"""Orchestrator: poses -> per-room clouds -> room boxes -> openings -> global assembly.

This is the seam where the design doc's claim gets tested, in order:
  §5 Level 2 (room box) -> §9 item 1 (doors) -> §9 item 4 (global 2D solve).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..schema import Opening, Room
from .floorplan import AssemblyReport, assemble_auto
from .openings import GapOpeningDetector
from .planes import RoomBox, fit_room_box
from .poses import Poses


@dataclass
class ReconstructResult:
    rooms: List[Room] = field(default_factory=list)
    assembly: Optional[AssemblyReport] = None
    notes: List[str] = field(default_factory=list)
    boxes: Dict[str, RoomBox] = field(default_factory=dict)

    @property
    def floorplan(self):
        return self._floorplan

    _floorplan: Any = None


def group_clouds(poses: Poses) -> Dict[str, np.ndarray]:
    """Split a cloud into per-room clouds.

    v0: uses ``poses.clouds`` if the backend provides them, else ``room_of_image``, else the
    whole scene as one room. Room clustering from arbitrary photos is §9 item 2 and is
    deliberately NOT faked here -- a wrong grouping would silently poison every downstream
    number.
    """
    if poses.clouds:
        return dict(poses.clouds)
    if poses.room_of_image:
        out: Dict[str, List[np.ndarray]] = {}
        for i, rid in enumerate(poses.room_of_image):
            out.setdefault(rid, []).append(poses.unproject(i, stride=2))
        return {k: np.vstack(v) for k, v in out.items()}
    return {"scene": poses.pointcloud()}


def reconstruct(
    poses: Poses,
    labels: Optional[Dict[str, str]] = None,
    adjacency: Optional[Sequence[Dict[str, str]]] = None,
    detector: Optional[Any] = None,
    snap_step: float = 0.05,
    box_kwargs: Optional[Dict[str, Any]] = None,
    min_confidence: float = 0.0,
) -> ReconstructResult:
    """Full pipeline: clouds -> rooms -> doors -> assembled floorplan."""
    from ..schema import Floorplan

    detector = detector or GapOpeningDetector()
    box_kwargs = dict(box_kwargs or {})
    labels = labels or {}

    clouds = group_clouds(poses)
    result = ReconstructResult(notes=[])

    for room_id, cloud in clouds.items():
        if cloud is None or len(cloud) < 10:
            result.notes.append(f"{room_id}: skipped (only {0 if cloud is None else len(cloud)} points)")
            continue
        box = fit_room_box(cloud, **box_kwargs)
        if box is None:
            result.notes.append(f"{room_id}: box fit failed (too few usable points)")
            continue
        if box.confidence < min_confidence:
            result.notes.append(
                f"{room_id}: dropped, wall support {box.confidence:.2f} < {min_confidence:.2f}"
            )
            continue

        openings: List[Opening] = detector.detect(
            cloud, box.polygon, floor_y=box.floor_y, ceiling=box.ceiling
        )
        room = Room(
            id=str(room_id),
            label=labels.get(str(room_id), str(room_id)),
            polygon=box.polygon,
            ceiling=box.ceiling,
            photos=[n for i, n in enumerate(poses.image_names)
                    if poses.room_of_image is None or poses.room_of_image[i] == room_id],
            openings=openings,
            confidence=round(float(box.confidence), 3),
            meta={
                "method": box.method,
                "angle_deg": round(float(np.degrees(box.angle)), 2),
                "floor_y": round(box.floor_y, 3),
                "n_points": box.n_points,
                **box.extra,
            },
        )
        result.rooms.append(room)
        result.boxes[str(room_id)] = box

        if not openings:
            result.notes.append(f"{room_id}: no openings detected (assembly will use walls)")
        for op in openings:
            if op.type == "passage":
                result.notes.append(
                    f"{room_id}: opening on wall {op.wall} has no lintel -- could be a "
                    "detection artefact or an archway"
                )

    if not result.rooms:
        result._floorplan = Floorplan()
        result.notes.append("no rooms reconstructed")
        return result

    floorplan, report = assemble_auto(result.rooms, adjacency, snap_step=snap_step)
    floorplan.meta.update({
        "backend": poses.backend,
        "views": poses.n,
        "notes": result.notes,
    })
    result._floorplan = floorplan
    result.assembly = report
    return result
