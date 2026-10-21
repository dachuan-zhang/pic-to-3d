"""Objective floorplan metrics.

The whole point of the PoC (see docs/BRAINSTORM.md §6 "the trap"): sparse-view models fail
*plausibly*, so a pretty render proves nothing. These numbers are what decide whether the
pipeline works.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..geometry import chamfer_corner_distance, polygon_iou, polygon_overlap_area
from ..schema import Floorplan, Room


def match_rooms(
    pred: Floorplan, gt: Floorplan, min_iou: float = 0.05
) -> List[Tuple[str, str, float]]:
    """Greedy room matching by descending IoU."""
    pairs: List[Tuple[str, str, float]] = []
    for rp in pred.rooms:
        for rg in gt.rooms:
            iou = polygon_iou(rp.polygon, rg.polygon)
            if iou >= min_iou:
                pairs.append((rp.id, rg.id, iou))
    pairs.sort(key=lambda p: -p[2])
    used_p, used_g, out = set(), set(), []
    for pid, gid, iou in pairs:
        if pid in used_p or gid in used_g:
            continue
        used_p.add(pid)
        used_g.add(gid)
        out.append((pid, gid, iou))
    return out


def _door_midpoints(room: Room, world: bool = True) -> np.ndarray:
    if not room.openings:
        return np.zeros((0, 2))
    pts = []
    for op in room.openings:
        try:
            pts.append(room.opening_midpoint(op))
        except Exception:
            continue
    return np.asarray(pts, dtype=float).reshape(-1, 2)


def _match_points(a: np.ndarray, b: np.ndarray, radius: float) -> int:
    if not len(a) or not len(b):
        return 0
    d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    used = set()
    n = 0
    for i in range(len(a)):
        order = np.argsort(d[i])
        for j in order:
            if d[i, j] > radius:
                break
            if j in used:
                continue
            used.add(j)
            n += 1
            break
    return n


def evaluate(
    pred: Floorplan,
    gt: Floorplan,
    door_radius: float = 0.5,
    label: str = "run",
) -> Dict[str, Any]:
    matches = match_rooms(pred, gt)
    ious = [iou for _, _, iou in matches]
    unmatched_pred = [r.id for r in pred.rooms if r.id not in {p for p, _, _ in matches}]
    unmatched_gt = [r.id for r in gt.rooms if r.id not in {g for _, g, _ in matches}]

    # IoU over *ground truth* rooms, so missed rooms cost you.
    mean_iou_gt = float(np.sum(ious) / max(1, len(gt.rooms)))

    corner_d = [
        chamfer_corner_distance(pred.room(pid).polygon, gt.room(gid).polygon)
        for pid, gid, _ in matches
    ]
    ceiling_err = [
        abs(pred.room(pid).ceiling - gt.room(gid).ceiling) for pid, gid, _ in matches
    ]

    # doors, matched per room pair by world midpoint
    tp = n_pred_doors = n_gt_doors = 0
    for pid, gid, _ in matches:
        pr = pred.room(pid)
        gr = gt.room(gid)
        a, b = _door_midpoints(pr), _door_midpoints(gr)
        n_gt_doors += len(b)
        m = _match_points(a, b, door_radius)
        tp += m
        n_pred_doors += len(a)
    precision = tp / n_pred_doors if n_pred_doors else 0.0
    recall = tp / n_gt_doors if n_gt_doors else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    # adjacency in gt-id space
    pid2gid = {pid: gid for pid, gid, _ in matches}
    pred_pairs = set()
    for e in pred.adjacency:
        a, b = pid2gid.get(e.get("from")), pid2gid.get(e.get("to"))
        if a and b:
            pred_pairs.add(frozenset((a, b)))
    gt_pairs = gt.adjacency_pairs()
    aj_tp = len(pred_pairs & gt_pairs)
    aj_p = aj_tp / len(pred_pairs) if pred_pairs else 0.0
    aj_r = aj_tp / len(gt_pairs) if gt_pairs else 0.0
    aj_f1 = 2 * aj_p * aj_r / (aj_p + aj_r) if (aj_p + aj_r) else 0.0

    overlap = 0.0
    for i, a in enumerate(pred.rooms):
        for b in pred.rooms[i + 1:]:
            overlap += polygon_overlap_area(a.polygon, b.polygon)

    return {
        "label": label,
        "n_pred": len(pred.rooms),
        "n_gt": len(gt.rooms),
        "matched": len(matches),
        "unmatched_pred": unmatched_pred,
        "unmatched_gt": unmatched_gt,
        "mean_iou": round(float(np.mean(ious)) if ious else 0.0, 4),
        "mean_iou_over_gt": round(mean_iou_gt, 4),
        "corner_dist_m": round(float(np.mean(corner_d)) if corner_d else float("nan"), 4),
        "ceiling_mae_m": round(float(np.mean(ceiling_err)) if ceiling_err else float("nan"), 4),
        "door_precision": round(precision, 4),
        "door_recall": round(recall, 4),
        "door_f1": round(f1, 4),
        "doors_pred": n_pred_doors,
        "doors_gt": n_gt_doors,
        "adjacency_f1": round(aj_f1, 4),
        "self_overlap_m2": round(overlap, 4),
        "rooms": [
            {"pred": pid, "gt": gid, "iou": round(iou, 4)} for pid, gid, iou in matches
        ],
    }


def format_report(m: Dict[str, Any]) -> str:
    lines = [
        f"== {m['label']}",
        f"  rooms        pred={m['n_pred']} gt={m['n_gt']} matched={m['matched']}",
        f"  IoU          mean={m['mean_iou']:.3f}  over_gt={m['mean_iou_over_gt']:.3f}",
        f"  corners      {m['corner_dist_m']:.3f} m (symmetric chamfer)",
        f"  ceiling      {m['ceiling_mae_m']:.3f} m MAE",
        f"  doors        P={m['door_precision']:.3f} R={m['door_recall']:.3f} "
        f"F1={m['door_f1']:.3f} ({m['doors_pred']} pred / {m['doors_gt']} gt)",
        f"  adjacency    F1={m['adjacency_f1']:.3f}",
        f"  self-overlap {m['self_overlap_m2']:.3f} m2 (should be ~0)",
    ]
    if m["unmatched_gt"]:
        lines.append(f"  MISSED rooms: {', '.join(m['unmatched_gt'])}")
    if m["unmatched_pred"]:
        lines.append(f"  SPURIOUS rooms: {', '.join(m['unmatched_pred'])}")
    return "\n".join(lines)


def aggregate(runs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Mean of the headline metrics across runs (the PoC's actual deliverable)."""
    if not runs:
        return {}
    keys = ["mean_iou", "mean_iou_over_gt", "corner_dist_m", "ceiling_mae_m",
            "door_f1", "adjacency_f1"]
    out: Dict[str, Any] = {"n_runs": len(runs)}
    for k in keys:
        vals = [r[k] for r in runs if isinstance(r.get(k), (int, float)) and r[k] == r[k]]
        out[k] = round(float(np.mean(vals)), 4) if vals else float("nan")
    out["room_recall"] = round(float(np.mean(
        [r["matched"] / max(1, r["n_gt"]) for r in runs]
    )), 4)
    return out
