"""Reconstruction pipeline stages."""
from __future__ import annotations

from .floorplan import (  # noqa: F401
    AssemblyReport,
    Placement,
    assemble,
    assemble_auto,
    assemble_greedy,
    check_overlaps,
)
from .openings import GapOpeningDetector, VLMDetector, detect_gap_openings  # noqa: F401
from .planes import RoomBox, detect_wall_segments, fit_room_box  # noqa: F401
from .poses import (  # noqa: F401
    HostedBackend,
    PoseBackend,
    Poses,
    StubBackend,
    StubScene,
    VGGTBackend,
    get_backend,
    load_or_estimate,
)
from .reconstruct import ReconstructResult, group_clouds, reconstruct  # noqa: F401

__all__ = [
    "AssemblyReport", "Placement", "assemble", "assemble_auto", "assemble_greedy",
    "check_overlaps", "GapOpeningDetector", "VLMDetector", "detect_gap_openings",
    "RoomBox", "fit_room_box", "detect_wall_segments",
    "Poses", "PoseBackend", "StubBackend", "StubScene", "VGGTBackend", "HostedBackend",
    "get_backend", "load_or_estimate",
    "ReconstructResult", "group_clouds", "reconstruct",
]
