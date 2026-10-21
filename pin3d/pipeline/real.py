"""Real-data harness: run the geometry stack on real photos with ground-truth depth.

Two questions this answers, both on real data rather than a synthetic box:

1. **How good is a real monocular depth model on real indoor photos?** Measured against
   metric ground truth (NYU Depth V2) after affine alignment, because Depth Anything V2 is
   affine-invariant — the model does not know scale.
2. **Does the geometry stack survive real depth?** The same code path (plane/box fit, ceiling
   estimate) is run on the ground-truth cloud and on the model cloud. Comparing the two
   isolates *model* error from *geometry* error.

Honest limitation: NYU Depth V2 is 480x640 Kinect RGB from 2011, not listing photography, and
a single view cannot see a whole room — so the room box here is a *partial-extent* estimate,
and its `confidence` is expected to be low. That is a finding, not a bug; multi-view capture
is what fixes it.
"""
from __future__ import annotations

import io
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from ..schema import Floorplan, Opening, Room
from .openings import GapOpeningDetector
from .planes import fit_room_box

#: Standard NYU Depth V2 intrinsics for the 640x480 labeled frames. These are the widely
#: used values; if they are slightly off, both the GT and model clouds are biased the same
#: way, so the A/B comparison between them stays valid (absolute metres may not).
NYU_K = np.array([
    [518.8579, 0.0, 325.5824],
    [0.0, 519.4696, 253.7362],
    [0.0, 0.0, 1.0],
], dtype=float)


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
@dataclass
class Sample:
    name: str
    rgb: np.ndarray        # (H, W, 3) uint8
    depth: Optional[np.ndarray]  # (H, W) float32 metres


def _decode_png(data: bytes) -> np.ndarray:
    from PIL import Image
    return np.array(Image.open(io.BytesIO(data)).convert("RGB"))


def _decode_depth(data: bytes) -> np.ndarray:
    from PIL import Image
    return np.array(Image.open(io.BytesIO(data))).astype(np.float32)


def iter_nyu_parquet(path: str, limit: Optional[int] = None) -> Iterator[Sample]:
    """Yield (rgb, metric depth) samples from a NYU-v2 parquet shard."""
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    n = 0
    for rg in range(pf.num_row_groups):
        table = pf.read_row_group(rg)
        images = table.column("image")
        depths = table.column("depth_map")
        for i in range(table.num_rows):
            if limit is not None and n >= limit:
                return
            rgb = _decode_png(images[i].as_py()["bytes"])
            depth = _decode_depth(depths[i].as_py()["bytes"])
            yield Sample(name=f"{path}:{n}", rgb=rgb, depth=depth)
            n += 1


def iter_nyu_h5(directory: str, limit: Optional[int] = None) -> Iterator[Sample]:
    """Yield samples from NYU-v2 labelled ``.h5`` files (on-disk subset)."""
    import glob
    import os

    import h5py

    files = sorted(glob.glob(os.path.join(directory, "*.h5")))
    for i, path in enumerate(files):
        if limit is not None and i >= limit:
            return
        with h5py.File(path, "r") as f:
            rgb = np.transpose(f["rgb"][()], (1, 2, 0)).astype(np.uint8)
            depth = f["depth"][()].astype(np.float32)
        yield Sample(name=os.path.basename(path), rgb=rgb, depth=depth)


# --------------------------------------------------------------------------
# depth -> cloud
# --------------------------------------------------------------------------
def cloud_from_depth(
    depth: np.ndarray,
    K: np.ndarray = NYU_K,
    stride: int = 2,
    min_depth: float = 0.4,
    max_depth: float = 8.0,
) -> np.ndarray:
    """Unproject a depth map into a world cloud with +Y up and the camera at the origin."""
    d = np.asarray(depth, dtype=np.float32)[::stride, ::stride]
    h, w = d.shape
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    u, v = np.meshgrid(np.arange(0, depth.shape[1], stride, dtype=np.float32),
                       np.arange(0, depth.shape[0], stride, dtype=np.float32))
    valid = np.isfinite(d) & (d > min_depth) & (d < max_depth)
    z = d[valid]
    x = (u[valid] - cx) / fx * z
    y = (v[valid] - cy) / fy * z
    return np.stack([x, -y, z], axis=1)  # flip Y: camera down -> world up


# --------------------------------------------------------------------------
# depth metrics (affine-invariant)
# --------------------------------------------------------------------------
def affine_fit(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> Tuple[float, float]:
    """Least-squares a, b such that a * pred + b ~= gt."""
    p = pred[valid].astype(np.float64)
    g = gt[valid].astype(np.float64)
    A = np.stack([p, np.ones_like(p)], axis=1)
    sol, *_ = np.linalg.lstsq(A, g, rcond=None)
    return float(sol[0]), float(sol[1])


def _warp(pred: np.ndarray, convention: str) -> np.ndarray:
    p = np.asarray(pred, dtype=np.float64)
    return p if convention == "depth" else 1.0 / (np.abs(p) + 1e-3)


def depth_metrics(
    pred: np.ndarray, gt: np.ndarray, convention: Optional[str] = None
) -> Dict[str, Any]:
    """AbsRel / RMSE / delta<1.25 after affine alignment.

    A model has **one** output convention; choosing it per-image is overfitting a flexible
    transform. Pass ``convention`` explicitly (``run_experiment`` picks it once, globally),
    or leave it None to get the best of both with ``convention="auto"`` recorded.
    """
    valid = np.isfinite(gt) & (gt > 0.3) & (gt < 8.0) & np.isfinite(pred)
    if valid.sum() < 100:
        return {"ok": False, "reason": f"only {int(valid.sum())} valid pixels"}

    options = [convention] if convention else ["depth", "disparity"]
    best: Optional[Dict[str, Any]] = None
    for name in options:
        warped = _warp(pred, name)
        a, b = affine_fit(warped, gt, valid)
        aligned = a * warped + b
        diff = np.abs(aligned[valid] - gt[valid])
        ratio = np.maximum(aligned[valid] / gt[valid], gt[valid] / aligned[valid])
        out = {
            "ok": True,
            "convention": name,  # type: ignore[dict-item]
            "a": a,
            "b": b,
            "abs_rel": float(np.mean(diff / gt[valid])),
            "rmse": float(math.sqrt(np.mean(diff ** 2))),
            "delta1": float(np.mean(ratio < 1.25)),
            "delta2": float(np.mean(ratio < 1.25 ** 2)),
            "_aligned": aligned,
        }
        if best is None or out["abs_rel"] < best["abs_rel"]:
            best = out
    assert best is not None
    if convention is None:
        best["convention"] = "auto"
    return best


def choose_convention(
    pairs: Sequence[Tuple[np.ndarray, np.ndarray]]
) -> Tuple[str, Dict[str, float]]:
    """Pick the model's output convention once, from the whole set, by mean AbsRel."""
    scores: Dict[str, List[float]] = {"depth": [], "disparity": []}
    for pred, gt in pairs:
        for name in scores:
            m = depth_metrics(pred, gt, convention=name)
            if m.get("ok"):
                scores[name].append(m["abs_rel"])
    means = {k: _mean(v) for k, v in scores.items()}
    valid = {k: v for k, v in means.items() if v == v}
    if not valid:
        return "depth", means
    return min(valid, key=lambda k: valid[k]), means


def align_to_metric(
    pred: np.ndarray,
    gt: np.ndarray,
    convention: Optional[str] = None,
    min_depth: float = 0.4,
    max_depth: float = 12.0,
) -> np.ndarray:
    """Apply the metric alignment implied by ``gt`` to ``pred`` (evaluation only)."""
    m = depth_metrics(pred, gt, convention=convention)
    if not m.get("ok"):
        return np.asarray(pred, dtype=np.float32)
    warped = _warp(pred, m["convention"] if m["convention"] != "auto" else "depth")
    return np.clip(m["a"] * warped + m["b"], min_depth, max_depth).astype(np.float32)


# --------------------------------------------------------------------------
# geometry on a real cloud
# --------------------------------------------------------------------------
#: A ceiling plane is only trustable if a real fraction of the cloud sits on it.
#: In a single camera view the ceiling is usually out of frame, and then the "top" of the
#: cloud is just the top of the frustum -- measuring that as a ceiling height is fiction.
CEILING_SUPPORT_MIN = 0.05


def geometry_summary(cloud: np.ndarray, detector: Optional[Any] = None) -> Optional[Dict[str, Any]]:
    box = fit_room_box(cloud)
    if box is None:
        return None
    w, d = box.size
    top_y = box.floor_y + box.ceiling
    ceiling_support = float(np.mean(np.abs(cloud[:, 1] - top_y) < 0.15))
    ceiling_valid = ceiling_support >= CEILING_SUPPORT_MIN
    openings: List[Opening] = []
    if detector is not None:
        try:
            openings = detector.detect(cloud, box.polygon, floor_y=box.floor_y,
                                       ceiling=box.ceiling)
        except Exception:
            openings = []
    return {
        "box": box,
        "width": float(w),
        "depth": float(d),
        "area": float(box.area),
        "ceiling": float(box.ceiling),
        "ceiling_support": ceiling_support,
        "ceiling_valid": ceiling_valid,
        "camera_height": float(-box.floor_y),
        "confidence": float(box.confidence),
        "openings": openings,
        "points": int(len(cloud)),
    }


# --------------------------------------------------------------------------
# experiment
# --------------------------------------------------------------------------
@dataclass
class SceneRow:
    name: str
    depth: Dict[str, Any]
    gt_geom: Optional[Dict[str, Any]]
    pred_geom: Optional[Dict[str, Any]]

    def as_dict(self) -> Dict[str, Any]:
        def g(x: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
            if x is None:
                return None
            return {k: (round(v, 4) if isinstance(v, float) else v)
                    for k, v in x.items() if k not in ("box", "openings")} | {
                "openings": len(x["openings"])}

        return {
            "name": self.name,
            "depth": {k: (round(v, 4) if isinstance(v, float) else v)
                      for k, v in self.depth.items() if not k.startswith("_")},
            "gt_geometry": g(self.gt_geom),
            "model_geometry": g(self.pred_geom),
        }


def run_experiment(
    samples: Sequence[Sample],
    model: Any,
    detector: Optional[Any] = None,
    verbose: bool = True,
    progress: Optional[Callable[[int, int, str], None]] = None,
) -> Dict[str, Any]:
    """Run depth + geometry on every sample and return per-scene rows and aggregates."""
    detector = detector or GapOpeningDetector()
    rows: List[SceneRow] = []
    skipped = 0

    usable = [s for s in samples if s.depth is not None]
    skipped = len(samples) - len(usable)
    if not usable:
        return {"rows": [], "skipped": skipped, "aggregate": {}, "convention": "depth"}

    # Pass 1: run the model once, then decide the output convention **globally**.
    predictions = [model.predict(s.rgb) for s in usable]
    convention, conv_scores = choose_convention(
        [(p, s.depth) for p, s in zip(predictions, usable)]
    )
    if verbose:
        print(f"  output convention: {convention} "
              f"(mean AbsRel depth={conv_scores['depth']:.3f}, "
              f"disparity={conv_scores['disparity']:.3f})")

    all_samples = usable
    for i, (s, pred) in enumerate(zip(all_samples, predictions)):
        dm = depth_metrics(pred, s.depth, convention=convention)
        if progress:
            progress(i + 1, len(all_samples), s.name)
        if verbose:
            tag = dm.get("convention", "-")
            print(f"  [{i + 1:3d}/{len(all_samples)}] {s.name[:28]:<28} "
                  f"abs_rel={dm.get('abs_rel', float('nan')):.3f} "
                  f"delta1={dm.get('delta1', float('nan')):.3f} ({tag})")

        gt_cloud = cloud_from_depth(s.depth)
        aligned = align_to_metric(pred, s.depth, convention=convention)
        pred_cloud = cloud_from_depth(aligned)

        rows.append(SceneRow(
            name=s.name,
            depth=dm,
            gt_geom=geometry_summary(gt_cloud, detector),
            pred_geom=geometry_summary(pred_cloud, detector),
        ))

    agg = aggregate(rows)
    agg["convention"] = convention
    agg["abs_rel_depth_conv"] = conv_scores.get("depth", float("nan"))
    agg["abs_rel_disparity_conv"] = conv_scores.get("disparity", float("nan"))
    return {"rows": rows, "skipped": skipped, "aggregate": agg, "convention": convention}


def _mean(vals: Sequence[float]) -> float:
    vals = [v for v in vals if isinstance(v, (int, float)) and v == v]
    return float(np.mean(vals)) if vals else float("nan")


def aggregate(rows: Sequence[SceneRow]) -> Dict[str, Any]:
    ok = [r for r in rows if r.depth.get("ok")]
    gt_ok = [r for r in rows if r.gt_geom is not None]
    both = [r for r in rows if r.gt_geom is not None and r.pred_geom is not None]

    gt_ceil = [r.gt_geom["ceiling"] for r in gt_ok if r.gt_geom["ceiling_valid"]]
    md_ceil = [r.pred_geom["ceiling"] for r in rows
               if r.pred_geom and r.pred_geom["ceiling_valid"]]
    both_ceil = [r for r in rows
                 if r.gt_geom and r.pred_geom
                 and r.gt_geom["ceiling_valid"] and r.pred_geom["ceiling_valid"]]

    out: Dict[str, Any] = {
        "n_scenes": len(rows),
        "n_depth_valid": len(ok),
        "ceiling_valid_rate": len(gt_ceil) / max(1, len(rows)),
        "gt_ceiling_valid": _mean(gt_ceil),
        "model_ceiling_valid": _mean(md_ceil),
        "abs_rel": _mean([r.depth["abs_rel"] for r in ok]),
        "rmse": _mean([r.depth["rmse"] for r in ok]),
        "delta1": _mean([r.depth["delta1"] for r in ok]),
        "delta2": _mean([r.depth["delta2"] for r in ok]),
        "convention_depth": sum(1 for r in ok if r.depth["convention"] == "depth"),
        "convention_disparity": sum(1 for r in ok if r.depth["convention"] == "disparity"),
        "gt_box_rate": len(gt_ok) / max(1, len(rows)),
        "model_box_rate": sum(1 for r in rows if r.pred_geom is not None) / max(1, len(rows)),
        "gt_ceiling": _mean([r.gt_geom["ceiling"] for r in gt_ok]),
        "model_ceiling": _mean([r.pred_geom["ceiling"] for r in rows if r.pred_geom]),
        "gt_confidence": _mean([r.gt_geom["confidence"] for r in gt_ok]),
        "model_confidence": _mean([r.pred_geom["confidence"] for r in rows if r.pred_geom]),
        "gt_area": _mean([r.gt_geom["area"] for r in gt_ok]),
        "model_area": _mean([r.pred_geom["area"] for r in rows if r.pred_geom]),
        "both_boxes": len(both),
    }
    if both_ceil:
        out["ceiling_mae_valid_only"] = _mean(
            [abs(r.gt_geom["ceiling"] - r.pred_geom["ceiling"]) for r in both_ceil]
        )
        out["n_ceiling_compared"] = len(both_ceil)
    if both:
        out["area_rel_err"] = _mean(
            [abs(r.pred_geom["area"] - r.gt_geom["area"]) / max(1e-6, r.gt_geom["area"])
             for r in both]
        )
    return out


def format_aggregate(a: Dict[str, Any]) -> str:
    def f(k: str, nd: int = 3) -> str:
        v = a.get(k)
        return f"{v:.{nd}f}" if isinstance(v, float) and v == v else "—"

    lines = [
        f"scenes                 {a['n_scenes']} ({a['n_depth_valid']} with valid depth)",
        f"depth  AbsRel          {f('abs_rel')}   RMSE {f('rmse')} m   "
        f"d<1.25 {f('delta1')}   d<1.25^2 {f('delta2')}",
        f"       convention      {a.get('convention', '?')} "
        f"(AbsRel depth={f('abs_rel_depth_conv')}, disparity={f('abs_rel_disparity_conv')})",
        f"box    solved          GT {a['gt_box_rate'] * 100:.0f}%  "
        f"model {a['model_box_rate'] * 100:.0f}%   (both {a['both_boxes']})",
        f"       ceiling visible {a['ceiling_valid_rate'] * 100:.0f}% of scenes "
        f"(support >= {CEILING_SUPPORT_MIN:.0%})  ->  height is NOT measurable from one view",
        f"       ceiling (valid) GT {f('gt_ceiling_valid', 2)} m  "
        f"model {f('model_ceiling_valid', 2)} m  MAE {f('ceiling_mae_valid_only', 2)} m "
        f"(n={a.get('n_ceiling_compared', 0)})",
        f"       area            GT {f('gt_area', 1)} m2  model {f('model_area', 1)} m2  "
        f"rel.err {f('area_rel_err')}",
        f"       wall support    GT {f('gt_confidence')}  model {f('model_confidence')}",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# export a real reconstruction as a floorplan the viewer can open
# --------------------------------------------------------------------------
def floorplan_from_geometry(
    geom: Dict[str, Any],
    name: str,
    source: str,
    label: str = "room",
    assumed_ceiling: float = 2.7,
) -> Floorplan:
    """A single view cannot close a room or see its ceiling; the plan says so in `meta`."""
    box = geom["box"]
    ceiling_valid = bool(geom.get("ceiling_valid", False))
    ceiling = float(box.ceiling) if ceiling_valid else assumed_ceiling
    room = Room(
        id="r0",
        label=label,
        polygon=box.polygon,
        ceiling=ceiling,
        openings=list(geom["openings"]),
        confidence=float(box.confidence),
        meta={
            "method": box.method,
            "source": source,
            "camera_height": round(-float(box.floor_y), 3),
            "n_points": int(box.n_points),
            "ceiling_source": "estimated" if ceiling_valid else "assumed 2.7 m",
            "ceiling_support": round(float(geom.get("ceiling_support", 0.0)), 4),
            "wall_support": round(float(geom.get("confidence", 0.0)), 4),
            "note": "single view: extents are a lower bound, not a full room",
        },
    )
    return Floorplan(
        rooms=[room],
        adjacency=[],
        meta={
            "source": source,
            "name": name,
            "single_view": True,
            "ceiling_observable": ceiling_valid,
            "note": ("one camera cannot close a room; the box is a partial extent. "
                     "Ceiling is assumed, not measured." if not ceiling_valid
                     else "one camera cannot close a room; the box is a partial extent."),
        },
    )
