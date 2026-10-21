"""CLI: ``python -m pin3d.cli <command>``.

Commands
--------
stub         synthetic two-room house -> floorplan.json + ground truth + scored report
reconstruct  images (or the stub) -> floorplan.json
eval         compare two floorplans
validate     schema/geometry sanity check
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .eval.metrics import evaluate, format_report
from .geometry import ensure_ccw, wall_segments
from .pipeline.openings import GapOpeningDetector
from .pipeline.poses import StubBackend, StubScene, get_backend, load_or_estimate
from .pipeline.depth import OnnxDepth
from .pipeline.real import (
    Sample,
    intrinsics_from_fov,
    metric_from_floor_prior,
    floorplan_from_geometry,
    format_aggregate,
    iter_nyu_h5,
    iter_nyu_parquet,
    run_experiment,
)
from .pipeline.real import cloud_from_depth, geometry_summary
from .viz import save_depth_png, save_rgb
from .pipeline.reconstruct import reconstruct
from .schema import Floorplan, Opening, Room

IMAGE_EXT = ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.JPG", "*.PNG")


def collect_images(path: str) -> List[Path]:
    p = Path(path)
    if p.is_file():
        return [p]
    files: List[Path] = []
    for pat in IMAGE_EXT:
        files.extend(sorted(p.glob(pat)))
    return sorted(set(files))


def gt_from_scene(scene: StubScene) -> Floorplan:
    """Exact ground truth for the synthetic scene (no noise, known doorway)."""
    rooms: List[Room] = []
    d = scene.doorway
    for spec in scene.rooms():
        lo = np.asarray(spec["lo"], dtype=float)
        hi = np.asarray(spec["hi"], dtype=float)
        poly = ensure_ccw(np.array([
            [lo[0], lo[2]], [hi[0], lo[2]], [hi[0], hi[2]], [lo[0], hi[2]],
        ]))
        openings: List[Opening] = []
        for i, (p0, p1) in enumerate(wall_segments(poly)):
            if abs(p0[d["axis"]] - d["at"]) < 1e-6 and abs(p1[d["axis"]] - d["at"]) < 1e-6:
                length = float(np.linalg.norm(p1 - p0))
                direction = (p1 - p0) / length
                z0, z1 = d["z_range"]
                mid = np.array([d["at"], (z0 + z1) / 2.0])
                t = float(np.clip(((mid - p0) @ direction) / length, 0.0, 1.0))
                openings.append(Opening(wall=i, t=t, width=float(z1 - z0),
                                        type="door", head=float(d["height"])))
                break
        rooms.append(Room(
            id=spec["id"], label=spec["label"], polygon=poly,
            ceiling=float(hi[1] - lo[1]), openings=openings,
        ))

    adjacency = []
    if len(rooms) == 2 and rooms[0].openings:
        adjacency.append({
            "from": rooms[0].id, "to": rooms[1].id,
            "via": f"{rooms[0].id}:{rooms[0].openings[0].wall}",
        })
    return Floorplan(rooms=rooms, adjacency=adjacency,
                     meta={"source": "stub_ground_truth"})


def cmd_stub(args: argparse.Namespace) -> int:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    scene = StubScene()
    backend = StubBackend(scene=scene, seed=args.seed)

    print(f"stub scene: {len(scene.rooms())} rooms, doorway x={scene.doorway['at']}")
    poses = backend.estimate([])
    for rid, cloud in (poses.clouds or {}).items():
        print(f"  cloud {rid:<12} {len(cloud):6d} points")

    result = reconstruct(poses, snap_step=args.snap)

    pred_path = out_dir / "floorplan.json"
    gt_path = out_dir / "floorplan_gt.json"
    result._floorplan.save(str(pred_path))
    gt = gt_from_scene(scene)
    gt.save(str(gt_path))

    print()
    print(result.assembly.format() if result.assembly else "no assembly")
    if result.notes:
        print("\nnotes")
        for n in result.notes:
            print(f"  - {n}")
    print()
    print(result._floorplan.summary())
    problems = result._floorplan.validate()
    print()
    if problems:
        print("validation problems:")
        for p in problems:
            print(f"  ! {p}")
    else:
        print("validation: ok")

    metrics = evaluate(result._floorplan, gt, label="stub")
    print()
    print(format_report(metrics))
    print()
    print(f"wrote {pred_path}")
    print(f"wrote {gt_path}")
    rel = pred_path.resolve().relative_to(Path.cwd().resolve())
    print()
    print("view it:")
    print(f"  python3 -m http.server 8000")
    print(f"  open 'http://localhost:8000/viewer/floorplan.html?src=../{rel}'")
    return 0


def cmd_reconstruct(args: argparse.Namespace) -> int:
    if args.backend == "stub":
        poses = StubBackend(seed=args.seed).estimate([])
    else:
        images = collect_images(args.images)
        if not images:
            print(f"no images found under {args.images!r}", file=sys.stderr)
            return 2
        print(f"{len(images)} images, backend={args.backend}")
        backend = get_backend(args.backend)
        poses = load_or_estimate(backend, images, cache_dir=args.cache)
        print(f"poses: {poses.n} views, depth={'yes' if poses.depth is not None else 'no'}")

    labels: Dict[str, str] = {}
    if args.labels:
        with open(args.labels, "r", encoding="utf-8") as fh:
            labels = json.load(fh)

    result = reconstruct(
        poses,
        labels=labels,
        detector=GapOpeningDetector(),
        snap_step=args.snap,
        min_confidence=args.min_confidence,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    result._floorplan.save(str(out))

    if result.assembly:
        print()
        print(result.assembly.format())
    if result.notes:
        print("\nnotes")
        for n in result.notes:
            print(f"  - {n}")
    print()
    print(result._floorplan.summary())
    problems = result._floorplan.validate()
    if problems:
        print("\nvalidation problems:")
        for p in problems:
            print(f"  ! {p}")
    print(f"\nwrote {out}")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    pred = Floorplan.load(args.pred)
    gt = Floorplan.load(args.gt)
    metrics = evaluate(pred, gt, door_radius=args.door_radius, label=Path(args.pred).name)
    print(format_report(metrics))
    if args.json:
        print(json.dumps(metrics, indent=2))
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    plan = Floorplan.load(args.plan)
    problems = plan.validate()
    print(plan.summary())
    print()
    if problems:
        for p in problems:
            print(f"! {p}")
        return 1
    print("ok")
    return 0


def cmd_real(args: argparse.Namespace) -> int:
    """Run the geometry stack on real photos with ground-truth depth (NYU Depth V2)."""
    if args.h5_dir:
        samples = list(iter_nyu_h5(args.h5_dir, limit=args.limit))
    elif args.parquet:
        samples = list(iter_nyu_parquet(args.parquet, limit=args.limit))
    else:
        print("need --parquet <file> or --h5-dir <dir>", file=sys.stderr)
        return 2
    if not samples:
        print("no samples loaded", file=sys.stderr)
        return 2

    first = samples[0]
    print(f"{len(samples)} real scenes from NYU Depth V2 "
          f"({first.rgb.shape[1]}x{first.rgb.shape[0]} RGB + metric depth)")
    model = OnnxDepth(path=args.model)
    print(f"depth model: {model.info.path} on {','.join(model.info.providers)}")

    result = run_experiment(samples, model, verbose=not args.quiet)
    agg = result["aggregate"]

    print()
    print(format_aggregate(agg))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(
        {"aggregate": agg, "rows": [r.as_dict() for r in result["rows"]]}, indent=2))
    print(f"\nwrote {out / 'summary.json'}")

    # export the best-supported single-view reconstructions for the viewer (A/B: GT vs model)
    ranked = sorted(
        result["rows"],
        key=lambda r: ((r.gt_geom or {}).get("confidence", 0.0) + (r.pred_geom or {}).get("confidence", 0.0)),
        reverse=True,
    )
    exported = 0
    for row in ranked:
        if exported >= args.export:
            break
        for tag, geom in (("gt", row.gt_geom), ("model", row.pred_geom)):
            if geom is None:
                continue
            plan = floorplan_from_geometry(geom, row.name, f"nyu_{tag}")
            kind = "gt" if tag == "gt" else "model"
            depth = row.extra_depths.get(kind) if hasattr(row, "extra_depths") else None
            if depth is not None:
                save_depth_png(str(out / f"scene{exported:02d}_{tag}_depth.png"), depth)
                save_rgb(str(out / f"scene{exported:02d}_rgb.jpg"), row.rgb)
                plan.meta["images"] = {"rgb": f"scene{exported:02d}_rgb.jpg",
                                       "depth": f"scene{exported:02d}_{tag}_depth.png"}
            path = out / f"scene{exported:02d}_{tag}.json"
            plan.save(str(path))
            print(f"wrote {path}")
        exported += 1
    if exported:
        print(f"\nview one with:\n  python3 -m http.server 8000")
        print(f"  open 'http://localhost:8000/viewer/floorplan.html?src=../{out}/scene00_gt.json'")
    return 0


def cmd_photos(args: argparse.Namespace) -> int:
    """Arbitrary photos (no ground truth) -> metric depth -> floorplan, one view per room.

    Metric scale comes from the floor plane plus the camera-height prior; ``real`` uses NYU
    ground truth instead. Expect ~0.30 AbsRel for scale recovered this way vs ~0.17 when
    aligned to ground truth.
    """
    from PIL import Image

    images = collect_images(args.images)
    if not images:
        print(f"no images under {args.images!r}", file=sys.stderr)
        return 2
    model = OnnxDepth(path=args.model)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    index = []
    print(f"{len(images)} photos | camera height prior {args.camera_height} m, "
          f"hfov {args.hfov}deg")
    for path in images:
        rgb = np.array(Image.open(path).convert("RGB"))
        h, w = rgb.shape[:2]
        K = intrinsics_from_fov(w, h, args.hfov)
        pred = model.predict(rgb)
        m = metric_from_floor_prior(pred, K, camera_height=args.camera_height)
        if not m.get("ok"):
            print(f"  {path.name:<26} skipped: {m.get('reason')}")
            continue
        cloud = cloud_from_depth(m["depth"], K)
        # scale recovery fixes the floor at exactly -camera_height; do not re-estimate it
        geom = geometry_summary(cloud, floor_y=-args.camera_height)
        if geom is None:
            print(f"  {path.name:<26} skipped: box fit failed")
            continue
        plan = floorplan_from_geometry(geom, path.name, "photo")
        # the reconstruction is meaningless without the picture it came from: ship both
        save_depth_png(str(out / f"{path.stem}_depth.png"), m["depth"])
        try:
            plan.meta["images"] = {
                "rgb": os.path.relpath(path.resolve(), out.resolve()),
                "depth": f"{path.stem}_depth.png",
            }
        except ValueError:
            plan.meta["images"] = {"depth": f"{path.stem}_depth.png"}
        plan.meta["floor_residual_m"] = round(float(m["floor_residual"]), 3)
        plan.meta["scale_source"] = f"floor plane + {args.camera_height} m camera prior"
        dest = out / f"{path.stem}.json"
        plan.save(str(dest))
        room = plan.rooms[0]
        print(f"  {path.name:<26} {geom['width']:.1f}x{geom['depth']:.1f} m "
              f"({geom['area']:.1f} m2)  wall_support={geom['confidence']:.2f} "
              f"floor_res={m['floor_residual']:.2f} m "
              f"(floor inliers {m.get('floor_inlier_ratio', 0)*100:.0f}%)  "
              f"doors={len(geom['openings'])}")
        index.append({"image": path.name, "plan": str(dest),
                      "area": round(geom["area"], 2),
                      "width": round(geom["width"], 2), "depth": round(geom["depth"], 2),
                      "wall_support": round(geom["confidence"], 3),
                      "floor_residual": round(float(m["floor_residual"]), 3),
                      "openings": len(geom["openings"])})

    (out / "index.json").write_text(json.dumps(index, indent=2))
    print(f"\nwrote {len(index)} plans to {out}")
    if index:
        print(f"\nview:\n  open 'http://localhost:8000/viewer/floorplan.html"
              f"?src=../{out}/{Path(index[0]['plan']).name}'")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="pin3d", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("stub", help="run the synthetic house end-to-end")
    s.add_argument("--out", default="examples")
    s.add_argument("--seed", type=int, default=7)
    s.add_argument("--snap", type=float, default=0.05)
    s.add_argument("--emit", type=int, default=0, help="reserved")
    s.set_defaults(func=cmd_stub)

    r = sub.add_parser("reconstruct", help="photos -> floorplan")
    r.add_argument("--images", default="samples", help="directory of photos")
    r.add_argument("--backend", default="stub", choices=["stub", "vggt", "hosted"])
    r.add_argument("--out", default="examples/floorplan.json")
    r.add_argument("--labels", default=None, help="JSON {room_id: label}")
    r.add_argument("--cache", default=".cache/poses")
    r.add_argument("--snap", type=float, default=0.05)
    r.add_argument("--min-confidence", type=float, default=0.0)
    r.add_argument("--seed", type=int, default=7)
    r.set_defaults(func=cmd_reconstruct)

    e = sub.add_parser("eval", help="score a floorplan against ground truth")
    e.add_argument("--pred", required=True)
    e.add_argument("--gt", required=True)
    e.add_argument("--door-radius", type=float, default=0.5)
    e.add_argument("--json", action="store_true")
    e.set_defaults(func=cmd_eval)

    ph = sub.add_parser("photos", help="arbitrary photos (no GT) -> floorplan per view")
    ph.add_argument("--images", default="samples")
    ph.add_argument("--model", default="models/depth_anything_v2_small.onnx")
    ph.add_argument("--out", default="out/photos")
    ph.add_argument("--camera-height", type=float, default=1.5)
    ph.add_argument("--hfov", type=float, default=68.0)
    ph.set_defaults(func=cmd_photos)

    r = sub.add_parser("real", help="real photos + ground-truth depth (NYU Depth V2)")
    r.add_argument("--parquet", default=None, help="NYU depth_v2 parquet shard")
    r.add_argument("--h5-dir", default=None, help="directory of NYU labelled .h5 files")
    r.add_argument("--model", default="models/depth_anything_v2_small.onnx")
    r.add_argument("--limit", type=int, default=40)
    r.add_argument("--out", default="out/real")
    r.add_argument("--export", type=int, default=3, help="single-view plans to export")
    r.add_argument("--quiet", action="store_true")
    r.set_defaults(func=cmd_real)

    v = sub.add_parser("validate", help="schema + geometry sanity check")
    v.add_argument("--plan", required=True)
    v.set_defaults(func=cmd_validate)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
