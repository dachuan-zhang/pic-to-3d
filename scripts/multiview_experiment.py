#!/usr/bin/env python3
"""Does multi-view actually fix what single-view cannot?

The claim under test (§15b/§15c of docs/BRAINSTORM.md): a room cannot be recovered from one
frame (29% area error, ceiling observable in 0 of 150 frames, zero doors), and multiple
overlapping views fix it.

NYU Depth V2 turned out to be unusable for this: its labelled frames are a slow handheld pan,
150 of 313 pairs are pure-rotation, and the mapper yields 6 points. So this uses **TUM RGB-D
fr1_room**, which has real translational camera motion plus ground-truth poses and metric depth.

Two arms, so a conclusion cannot be blamed on the depth model:

  gt    -- ground-truth depth          (isolates the geometry)
  model -- Depth Anything V2 (ONNX)    (what a real deployment has)

Within each arm: single view vs. 5 views fused with ground-truth poses.

    python scripts/multiview_experiment.py --tum /tmp/tum/rgbd_dataset_freiburg1_room
"""
from __future__ import annotations

import argparse
import bisect
import math
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pin3d.geometry import polygon_area                       # noqa: E402
from pin3d.pipeline.multiview import fuse, run_colmap          # noqa: E402
from pin3d.pipeline.openings import GapOpeningDetector         # noqa: E402
from pin3d.pipeline.real import cloud_from_depth, geometry_summary  # noqa: E402

# TUM fr1 intrinsics (the dataset's own calibration)
TUM_FR1_K = np.array([[517.3, 0.0, 318.6], [0.0, 516.5, 255.3], [0.0, 0.0, 1.0]])


@dataclass
class Frame:
    stamp: float
    rgb: np.ndarray
    depth: np.ndarray
    R: np.ndarray       # world <- cam
    t: np.ndarray       # world translation, metres


def _read_list(path: str) -> List[Tuple[float, str]]:
    out = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                out.append((float(parts[0]), parts[1]))
    return out


def _quat_to_R(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    n = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw) or 1.0
    x, y, z, w = qx / n, qy / n, qz / n, qw / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def load_tum(root: str, limit: Optional[int] = None) -> List[Frame]:
    from PIL import Image

    rgb_list = _read_list(os.path.join(root, "rgb.txt"))
    dep_list = _read_list(os.path.join(root, "depth.txt"))
    gt: List[Tuple[float, np.ndarray, np.ndarray]] = []
    with open(os.path.join(root, "groundtruth.txt"), "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            p = [float(v) for v in line.split()]
            gt.append((p[0], np.array(p[1:4]), _quat_to_R(*p[4:8])))
    gt_stamps = [g[0] for g in gt]

    dep_stamps = [d[0] for d in dep_list]
    frames: List[Frame] = []
    for stamp, name in rgb_list:
        if limit is not None and len(frames) >= limit:
            break
        j = bisect.bisect_left(dep_stamps, stamp)
        if j >= len(dep_list) or abs(dep_stamps[j] - stamp) > 0.05:
            continue
        k = bisect.bisect_left(gt_stamps, stamp)
        if k >= len(gt):
            continue
        rgb = np.array(Image.open(os.path.join(root, name)).convert("RGB"))
        depth = np.array(Image.open(os.path.join(root, dep_list[j][1]))).astype(np.float32) / 1000.0
        _, gt_t, gt_R = gt[k]
        # TUM groundtruth is camera->world already
        frames.append(Frame(stamp=stamp, rgb=rgb, depth=depth, R=gt_R, t=gt_t))
    return frames


def unproject_raw(depth: np.ndarray, K: np.ndarray, stride: int = 2) -> np.ndarray:
    """Depth -> camera-frame points in TUM's own convention: x right, y DOWN, z forward.

    Deliberately *not* ``cloud_from_depth``: that one flips Y for our internal +Y-up world,
    and feeding flipped points through a y-down camera pose mirrors the scene (observed: a
    3 m room collapsing to 0.87 m wide).
    """
    d = np.asarray(depth, dtype=np.float32)[::stride, ::stride]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    u, v = np.meshgrid(np.arange(0, depth.shape[1], stride, dtype=np.float32),
                       np.arange(0, depth.shape[0], stride, dtype=np.float32))
    m = np.isfinite(d) & (d > 0.3) & (d < 8.0)
    z = d[m]
    return np.stack([(u[m] - cx) / fx * z, (v[m] - cy) / fy * z, z], axis=1)


def world_cloud(frame: Frame, depth: np.ndarray, stride: int = 2) -> np.ndarray:
    cam = unproject_raw(depth, TUM_FR1_K, stride=stride)
    pw = np.einsum("ij,kj->ik", cam, frame.R) + frame.t
    pw[:, 1] *= -1.0          # TUM world is y-down; the pipeline wants +Y up
    return pw


def overlap(a: Frame, b: Frame, stride: int = 6) -> float:
    """Fraction of A's valid depth pixels that land in B with agreeing depth."""
    da = a.depth[::stride, ::stride]
    h, w = da.shape
    fx, fy, cx, cy = TUM_FR1_K[0, 0], TUM_FR1_K[1, 1], TUM_FR1_K[0, 2], TUM_FR1_K[1, 2]
    u, v = np.meshgrid(np.arange(0, a.depth.shape[1], stride), np.arange(0, a.depth.shape[0], stride))
    m = (da > 0.3) & (da < 8)
    if m.sum() < 50:
        return 0.0
    z = da[m]
    cam = np.stack([(u[m] - cx) / fx * z, (v[m] - cy) / fy * z, z], axis=1)
    pw = np.einsum("ij,kj->ik", cam, a.R) + a.t
    rel = np.einsum("ij,jk->ik", pw - b.t, b.R)   # R^T (pw - t)
    zz = rel[:, 2]
    ok = zz > 0.3
    if ok.sum() < 50:
        return 0.0
    uu = fx * rel[ok, 0] / zz[ok] + cx
    vv = fy * rel[ok, 1] / zz[ok] + cy
    inside = (uu >= 0) & (uu < b.depth.shape[1]) & (vv >= 0) & (vv < b.depth.shape[0])
    if inside.sum() < 50:
        return 0.0
    db = b.depth[vv[inside].astype(int), uu[inside].astype(int)]
    agree = (db > 0.3) & (np.abs(db - zz[ok][inside]) < 0.25 * np.maximum(db, 1e-3))
    return float(agree.sum() / len(z))


def pick_group(frames: Sequence[Frame], seed_idx: int, n_views: int,
               min_baseline: float = 0.35, min_overlap: float = 0.35) -> List[int]:
    seed = frames[seed_idx]
    scored = []
    for i, f in enumerate(frames):
        if i == seed_idx:
            continue
        base = float(np.linalg.norm(f.t - seed.t))
        if base < min_baseline:
            continue
        ov = overlap(seed, f)
        if ov >= min_overlap:
            scored.append((ov, base, i))
    scored.sort(reverse=True)
    # spread the picks out so the group is not five near-identical frames
    chosen, used = [seed_idx], [seed.t]
    for ov, base, i in scored:
        if len(chosen) >= n_views:
            break
        if all(np.linalg.norm(frames[i].t - u) > min_baseline for u in used):
            chosen.append(i)
            used.append(frames[i].t)
    return chosen


def heldout_support(polygon: np.ndarray, floor_y: float, cloud: np.ndarray,
                    max_dist: float = 0.20) -> float:
    """Wall support of a box on a cloud it was NOT fitted to.

    The support figure single-view runs report is self-fulfilling: the box is the min-area
    rectangle of exactly those points, so it always "explains" them. Fitting on some views and
    scoring on a held-out view is the only honest version.
    """
    from pin3d.pipeline.planes import band_points, distance_to_polygon_boundary

    shifted = np.asarray(cloud, dtype=float).copy()
    shifted[:, 1] -= floor_y
    band = band_points(shifted, 0.8, 1.6)
    if len(band) < 50:
        return float("nan")
    d = distance_to_polygon_boundary(band[:, [0, 2]], polygon)
    return float(np.mean(d <= max_dist))


def analyse(cloud: np.ndarray, label: str) -> Optional[Dict[str, float]]:
    geom = geometry_summary(cloud, detector=GapOpeningDetector())
    if geom is None:
        print(f"    {label:<22} box fit FAILED")
        return None
    print(f"    {label:<22} {geom['width']:5.2f} x {geom['depth']:5.2f} m "
          f"({geom['area']:6.1f} m2)  wall_support={geom['confidence']:.2f}  "
          f"ceiling={'y' if geom['ceiling_valid'] else 'n'}  doors={len(geom['openings'])}")
    return {"area": geom["area"], "w": geom["width"], "d": geom["depth"],
            "conf": geom["confidence"], "ceiling_valid": geom["ceiling_valid"],
            "doors": len(geom["openings"])}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tum", default="/tmp/tum/rgbd_dataset_freiburg1_room")
    ap.add_argument("--limit", type=int, default=120)
    ap.add_argument("--views", type=int, default=5)
    ap.add_argument("--seed", type=int, default=-1, help="-1 = middle frame")
    ap.add_argument("--model", default="models/depth_anything_v2_small.onnx")
    args = ap.parse_args()

    frames = load_tum(args.tum, limit=args.limit)
    if len(frames) < 10:
        print("not enough frames", file=sys.stderr)
        return 2
    print(f"loaded {len(frames)} synchronised RGB-D frames with ground-truth poses")

    seed_idx = len(frames) // 2 if args.seed < 0 else min(args.seed, len(frames) - 1)
    idx = pick_group(frames, seed_idx, args.views)
    seed = frames[idx[0]]
    base = [float(np.linalg.norm(frames[i].t - seed.t)) for i in idx]
    print(f"\ngroup: {idx}")
    print(f"  baselines from seed: {[round(b, 2) for b in base]} m "
          f"(median scene depth {np.median(seed.depth[seed.depth > 0.3]):.2f} m)")
    print(f"  overlap with seed: {[round(overlap(seed, frames[i]), 2) for i in idx]}")

    # --- ground truth depth -------------------------------------------------
    print("\n== ground-truth depth ==")
    clouds_gt = {i: world_cloud(frames[i], frames[i].depth) for i in idx}
    held = idx[-1]                      # never used for the single-view fit
    fit_views = [i for i in idx if i != held]

    gt_multi = np.vstack([clouds_gt[i] for i in idx])
    g1 = analyse(clouds_gt[idx[0]], "single view (seed)")
    gm = analyse(gt_multi, f"{len(idx)}-view fused")

    # held-out generalisation: does a box fitted on some views explain an unseen one?
    def box_and_floor(cloud: np.ndarray):
        geom = geometry_summary(cloud)
        return None if geom is None else (geom["box"].polygon, geom["box"].floor_y)

    single_box = box_and_floor(clouds_gt[fit_views[0]])
    multi_box = box_and_floor(np.vstack([clouds_gt[i] for i in fit_views]))
    if single_box and multi_box:
        ho_single = heldout_support(single_box[0], single_box[1], clouds_gt[held])
        ho_multi = heldout_support(multi_box[0], multi_box[1], clouds_gt[held])
        print(f"    held-out view {held}: support single-fit={ho_single:.2f}  "
              f"multi-fit={ho_multi:.2f}")

    # --- model depth --------------------------------------------------------
    model_results: Dict[str, Optional[Dict[str, float]]] = {}
    try:
        from pin3d.pipeline.depth import OnnxDepth
        from pin3d.pipeline.real import metric_from_floor_prior
        model = OnnxDepth(args.model)
        print("\n== Depth Anything V2 depth ==")
        clouds, poses_ok = {}, True
        for i in idx:
            pred = model.predict(frames[i].rgb)
            m = metric_from_floor_prior(pred, TUM_FR1_K, camera_height=1.5)
            if not m.get("ok"):
                poses_ok = False
                print(f"    frame {i}: scale recovery failed ({m.get('reason')})")
                continue
            clouds[i] = cloud_from_depth(m["depth"], TUM_FR1_K)
        if clouds:
            names = list(clouds)
            rot = {f"f{i}": frames[i].R for i in names}
            tra = {f"f{i}": frames[i].t for i in names}
            one = fuse({f"f{names[0]}": clouds[names[0]]}, rot, tra, 1.0)
            mv = fuse({f"f{i}": clouds[i] for i in names}, rot, tra, 1.0)
            model_results["single"] = analyse(one, "single view")
            model_results["multi"] = analyse(mv, f"{len(names)}-view fused")
    except Exception as exc:  # the model arm is optional
        print(f"\n  (model arm skipped: {exc})")

    # --- can SfM find those poses by itself? --------------------------------
    print("\n== COLMAP (no ground-truth poses) ==")
    imgs = [(f"f{i:03d}.jpg", frames[i].rgb) for i in idx]
    sfm = run_colmap(imgs, camera_model="SIMPLE_RADIAL")
    if sfm.ok:
        # SfM has an arbitrary global frame, so absolute rotations are not comparable.
        # Estimate the global rotation first (Procrustes over the views), then measure.
        names = [f"f{i:03d}.jpg" for i in idx if f"f{i:03d}.jpg" in sfm.rotations]
        if len(names) >= 3:
            gt = {f"f{i:03d}.jpg": frames[i].R for i in idx}
            # SfM gives world->cam; convert to cam->world for comparison
            est = {n: sfm.rotations[n].T for n in names}
            A = sum(gt[n] @ est[n].T for n in names)
            u, _, vt = np.linalg.svd(A)
            r_align = u @ vt
            errs = []
            for n in names:
                rel = r_align @ est[n]
                cosang = (np.trace(rel.T @ gt[n]) - 1) / 2
                errs.append(math.degrees(math.acos(max(-1.0, min(1.0, cosang)))))
            print(f"    registered {sfm.n}/{len(idx)} images, {sfm.points} points, "
                  f"aligned rotation error: median {np.median(errs):.2f} deg, "
                  f"max {np.max(errs):.2f} deg")
        else:
            print(f"    registered only {sfm.n}/{len(idx)} images, {sfm.points} points "
                  f"(too few to align)")
    else:
        print(f"    FAILED: {sfm.reason}")

    # --- verdict ------------------------------------------------------------
    if g1 and gm:
        print("\n== verdict ==")
        print(f"  GT depth   area single {g1['area']:.1f} -> fused {gm['area']:.1f} m2  "
              f"({(gm['area'] / g1['area'] - 1) * 100:+.0f}%)")
        print(f"  GT depth   wall support {g1['conf']:.2f} -> {gm['conf']:.2f}")
        s, m = model_results.get("single"), model_results.get("multi")
        if s and m:
            print(f"  model      area single {s['area']:.1f} -> fused {m['area']:.1f} m2  "
                  f"({(m['area'] / s['area'] - 1) * 100:+.0f}%)")
            print(f"  model      wall support {s['conf']:.2f} -> {m['conf']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
