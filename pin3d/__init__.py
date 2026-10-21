"""pic-to-3d: property photos -> editable floorplan -> 3D walkthrough.

v0 entry point. See docs/BRAINSTORM.md for why the pipeline is shaped this way.
"""
from __future__ import annotations

__version__ = "0.1.0"

from .schema import Floorplan, Opening, Room  # noqa: F401

__all__ = ["Floorplan", "Room", "Opening", "__version__"]
