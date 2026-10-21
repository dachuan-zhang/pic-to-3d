"""Door / window detection.

Ranked #1 risk in the design doc (§9). The v0 implementation is a **wall-gap detector**
on the fused point cloud: a doorway is a contiguous run of empty bins along a wall, bounded
by wall support, with the right width. It is honest about what it cannot do and reports a
confidence instead of guessing silently.

The detector interface is a Protocol so a segmentation model (SAM / OneFormer door classes)
or a VLM pass (``VLMDetector``) can replace it without touching the assembly code.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Protocol, Sequence

import numpy as np

from ..geometry import dot_rows, wall_normal, wall_segments
from ..schema import Opening

#: Real doors cluster here; used to sanity-check detections.
TYPICAL_DOOR = (0.70, 1.00)


class OpeningDetector(Protocol):
    def detect(
        self, points: np.ndarray, polygon: np.ndarray, floor_y: float = 0.0, ceiling: float = 2.7
    ) -> List[Opening]:
        ...


def _runs(mask: np.ndarray, merge_gap: int = 0) -> List[tuple]:
    """Contiguous False runs in a boolean mask, as (start, end_exclusive).

    ``merge_gap`` joins runs separated by at most that many True bins. A single stray point
    in a doorway otherwise splits it into two sub-minimum-width runs and the door is lost --
    an observed failure, not a hypothetical one.
    """
    out: List[tuple] = []
    start: Optional[int] = None
    for i, v in enumerate(mask):
        if not v and start is None:
            start = i
        elif v and start is not None:
            out.append((start, i))
            start = None
    if start is not None:
        out.append((start, len(mask)))
    if merge_gap <= 0 or len(out) < 2:
        return out
    merged: List[tuple] = [out[0]]
    for run in out[1:]:
        if run[0] - merged[-1][1] <= merge_gap:
            merged[-1] = (merged[-1][0], run[1])
        else:
            merged.append(run)
    return merged


def detect_gap_openings(
    points: np.ndarray,
    polygon: np.ndarray,
    floor_y: float = 0.0,
    ceiling: float = 2.7,
    wall_eps: float = 0.09,
    bin_width: float = 0.04,
    min_width: float = 0.60,
    max_width: float = 1.90,
    head: float = 2.05,
    min_points_per_bin: int = 1,
    edge_margin: float = 0.15,
    merge_gap_bins: int = 1,
) -> List[Opening]:
    """Detect openings as gaps in wall support along each polygon edge."""
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    if len(pts) < 20:
        return []

    pts2 = pts[:, [0, 2]]
    h = pts[:, 1] - floor_y
    openings: List[Opening] = []

    for wall_idx, (p0, p1) in enumerate(wall_segments(polygon)):
        d = p1 - p0
        length = float(np.linalg.norm(d))
        if length < 0.4:
            continue
        direction = d / length
        normal = wall_normal(p0, p1)

        rel = pts2 - p0[None, :]
        s = dot_rows(rel, direction)
        perp = np.abs(dot_rows(rel, normal))

        near = (perp <= wall_eps) & (s >= -wall_eps) & (s <= length + wall_eps)
        # door band: floor to just below the lintel
        band = near & (h >= 0.10) & (h <= head)
        # header band: the wall above the opening
        header = near & (h > head + 0.10) & (h <= max(ceiling - 0.05, head + 0.2))

        if int(np.count_nonzero(band)) < min_points_per_bin:
            continue

        n_bins = max(3, int(round(length / bin_width)))
        bin_size = length / n_bins
        idx = np.clip((s[band] / length * n_bins).astype(int), 0, n_bins - 1)
        counts = np.bincount(idx, minlength=n_bins)
        occupied = counts >= min_points_per_bin

        margin_bins = int(round(edge_margin / bin_size))
        for run_start, run_end in _runs(occupied, merge_gap=merge_gap_bins):
            if run_start <= margin_bins or run_end >= n_bins - margin_bins:
                continue  # gap at a corner is a missing wall, not a door
            width = (run_end - run_start) * bin_size
            if not (min_width <= width <= max_width):
                continue
            if run_start == 0 or run_end >= n_bins:
                continue
            # need wall support immediately on both flanks
            if not (occupied[run_start - 1] and occupied[run_end]):
                continue

            s_lo = run_start * bin_size
            s_hi = run_end * bin_size
            hdr = header & (s >= s_lo - bin_size) & (s <= s_hi + bin_size)
            header_present = int(np.count_nonzero(hdr)) >= max(2, int(0.25 * (s_hi - s_lo) / bin_size))
            kind = "door" if header_present else "passage"

            confidence = 0.5 + 0.35 * float(header_present)
            if TYPICAL_DOOR[0] <= width <= TYPICAL_DOOR[1]:
                confidence = min(1.0, confidence + 0.15)

            openings.append(Opening(
                wall=wall_idx,
                t=float(np.clip((run_start + run_end) / (2.0 * n_bins), 0.0, 1.0)),
                width=float(width),
                type=kind,
                head=float(head),
                confidence=float(round(confidence, 3)),
            ))

    openings.sort(key=lambda o: (o.wall, o.t))
    return openings


@dataclass
class GapOpeningDetector:
    """Default detector: point-cloud wall-gap analysis."""

    wall_eps: float = 0.09
    bin_width: float = 0.04
    min_width: float = 0.60
    max_width: float = 1.90
    head: float = 2.05
    min_points_per_bin: int = 1
    merge_gap_bins: int = 1

    def detect(self, points, polygon, floor_y: float = 0.0, ceiling: float = 2.7) -> List[Opening]:
        return detect_gap_openings(
            points, polygon,
            floor_y=floor_y, ceiling=ceiling, wall_eps=self.wall_eps,
            bin_width=self.bin_width, min_width=self.min_width,
            max_width=self.max_width, head=self.head,
            min_points_per_bin=self.min_points_per_bin,
            merge_gap_bins=self.merge_gap_bins,
        )


class VLMDetector:
    """Placeholder for a VLM pass ("where are the doors in this photo?").

    Not implemented on purpose: VLM door placement is unverified at v0, and an unverified
    detector must not be wired into assembly. Implement ``detect`` returning ``Opening``
    objects in the same wall/t convention to swap it in.
    """

    def __init__(self, model: str = "unset") -> None:
        self.model = model

    def detect(self, points, polygon, floor_y: float = 0.0, ceiling: float = 2.7) -> List[Opening]:
        raise NotImplementedError(
            "VLMDetector is a v0 interface stub. See docs/BRAINSTORM.md §9 item 1."
        )
