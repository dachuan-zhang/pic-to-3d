# pic-to-3d

Turn a pile of property photos into an **editable floorplan** and a walkable 3D scene.

v0 is a measurement harness first and a renderer second: the question it answers is *how good
is the reconstruction*, not *how pretty is the demo*. Design rationale, decisions and open
questions live in **[docs/BRAINSTORM.md](docs/BRAINSTORM.md)** — read that first.

## Why floorplan-first

Reconstructing a whole house as one fused mesh from sparse listing photos fails quietly and
plausibly. So the pipeline predicts a 2D floorplan (rooms, polygons, ceiling heights, doors)
and attaches imagery to it. That turns house-scale reconstruction into a small pose graph:

```
point cloud per room → Manhattan room box → wall-gap doors → global 2D assembly → floorplan.json
                                                                                 ↓
                                                              2D editor + extruded 3D walk
```

## Quickstart (no GPU, no weights, no network)

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

.venv/bin/python -m pin3d.cli stub --out examples      # synthetic house, end to end
.venv/bin/python -m unittest discover -s tests -t .    # 92 tests

# viewer needs HTTP (fetch() is blocked on file://)
python3 -m http.server 8000
open http://localhost:8000/viewer/floorplan.html
```

`stub` produces `examples/floorplan.json` + `examples/floorplan_gt.json` and scores one against
the other:

```
== stub
  rooms        pred=2 gt=2 matched=2
  IoU          mean=1.000  over_gt=1.000
  corners      0.000 m (symmetric chamfer)
  ceiling      0.016 m MAE
  doors        P=1.000 R=1.000 F1=1.000 (2 pred / 2 gt)
  adjacency    F1=1.000
  self-overlap 0.000 m2 (should be ~0)
```

That is the pipeline working on a **synthetic** scene. It says nothing about real photos — that
is the next milestone, and it is the actual experiment.

## Real data (the actual experiment)

ZInD is gated, so the real arm runs on **NYU Depth V2** — 480x640 RGB with float32 **metric**
depth ground truth — and **Depth Anything V2 Small via ONNX Runtime** (no torch, no GPU).

```bash
bash scripts/fetch_models.sh          # ~99 MB, once
.venv/bin/python -m pin3d.cli real --parquet path/to/nyu-shard.parquet --limit 150
```

Measured on **150 real rooms** (171 distinct rooms among the first 200 frames):

| metric | value |
|---|---|
| depth AbsRel (affine-aligned) | 0.171 |
| depth RMSE / delta<1.25 | 0.688 m / 0.813 |
| room box solved | 100% |
| room area, GT vs model | 17.1 m² vs 12.5 m² → **29% error** |
| ceiling observable from one view | **0 / 150** |

What that says:

- The depth model is **good enough to use** (17% AbsRel, 0.6 s/room on CPU). Not the bottleneck.
- **Extent geometry amplifies depth error**: 17% depth → 29% area, because a box is defined by
  its extreme points.
- **A single view cannot measure ceiling height** — not one of 150 frames had a real top plane.
  The harness refuses to report it and the exported plans mark the ceiling as assumed.
- The model scores *higher* wall support than ground truth while producing *smaller* boxes:
  smoother depth looks better without being more correct.

Which is the design doc's thesis, now measured: **model quality is not the constraint, capture
geometry is.** Getting real numbers out of one photo per room is not possible; the box needs
multiple views.

## Sample photos

```bash
python scripts/fetch_samples.py --count 10     # CC-licensed interiors from Wikimedia Commons
.venv/bin/python -m pin3d.cli photos --images samples --out out/photos
open 'http://localhost:8000/viewer/floorplan.html?src=../out/photos/sample05.json'
```

Metric scale has no ground truth here, so it comes from a **floor plane + a 1.5 m camera-height
prior**, which solves a linear system in (scale, offset, height) — plain "scale by camera
height" is off by ~2x because the model's offset `b` is the same order as the scene depth.

Result on 7 real photos:

| photo | box | wall support | floor inliers |
|---|---|---|---|
| sample05 | 6.3 x 3.3 m | 0.35 | 88% |
| sample06 | 5.1 x 2.6 m | 0.51 | 79% |
| sample03 | 9.6 x 3.6 m | 0.37 | 75% |
| sample10 | **5.7 x 1.4 m** | 0.88 | **31%** |
| sample08 | 4.9 x 5.5 m | 0.34 | **14%** |

`sample10` is a clean empty apartment photographed from a doorway; the true room is roughly
8 x 5 m. We got 5.7 x 1.4 m and reported 0.88 wall support — **confidently wrong**. The floor
inlier ratio (31%) is the only column that flags it, because a foreground plant occupied the
bottom band and the floor fit locked onto it.

Also: **zero doors detected** across all seven photos. The wall-gap detector expects a closed
room boundary; a single view never provides one.

So the sample-photo path runs end to end and produces viewable plans, but the dimensions are
only trustworthy when the floor is unoccluded and the view is wide. That is the measured
single-view limit, not a missing feature. `--camera-height` and `--hfov` are exposed because
both are priors and both matter.

## Commands

```bash
python -m pin3d.cli stub --out examples            # synthetic end-to-end + scored report
python -m pin3d.cli real --parquet shard.parquet --limit 150   # real photos + GT depth
python -m pin3d.cli photos --images samples                   # arbitrary photos, no GT
python -m pin3d.cli reconstruct --images samples/ --backend stub --out out/plan.json
python -m pin3d.cli reconstruct --images samples/ --backend vggt  # needs torch + vggt
python -m pin3d.cli eval    --pred out/plan.json --gt examples/floorplan_gt.json
python -m pin3d.cli validate --plan out/plan.json
```

## Viewer

`viewer/floorplan.html` — single file, no build step.

- **2D plan**: metric grid, per-room area, door/window marks, drag rooms to fix the plan
  (snap toggle), export the edited JSON.
- **3D walk**: walls extruded with real door gaps and lintels, pointer-lock first-person camera
  clipped to the room polygons, so you cannot walk through walls. `F` toggles an overview camera.
  Uses WebGL when available and falls back to a Canvas2D software renderer otherwise, so the 3D
  view cannot fail to start. three.js is vendored (`viewer/vendor/`), so nothing loads from a CDN.

Editing matters: automatic floorplans are never right, and a room you can fix in ten seconds
beats a 3D scene you cannot fix at all.

## What is real vs. stubbed

| Component | State |
|---|---|
| Schema, validation, JSON round-trip | real |
| Geometry (IoU, hull, min-area rect, rectification) | real, tested |
| Room box fit (density filter + outer-median wall planes) | real, tested |
| Door detection (point-cloud wall gaps) | real, tested |
| Global 2D assembly (BFS with hints, greedy without) | real, tested |
| Metrics vs. ground truth | real, tested |
| Viewer (2D editor + 3D walk) | real |
| `VGGTBackend` | wiring only — written against the published API, **never executed here** |
| `HostedBackend` | interface only |
| `VLMDetector` | interface only |
| ZInD adapter | defensive best-effort; validate against your download |
| Room clustering / identity from arbitrary photos | **not implemented** (deliberately not faked) |
| ONNX depth model (Depth Anything V2 Small) | real, exercised on 150 NYU rooms |
| Real-data evaluation vs. metric GT depth | real (NYU Depth V2) |

## Known limitations (v0)

- **Manhattan world only**: orthogonal walls, flat ceiling, single floor. L-shaped rooms fit a
  bounding box. Curved, 45°, vaulted, split-level and stairs are out of scope.
- **Room identity is unsolved** for arbitrary photos — the stub provides per-room clouds. Until
  clustering lands, real-photo runs collapse everything into one room.
- **Assembly is greedy, not optimal**: brute force over 4 yaw options x door pairings, scored;
  no global least-squares/Bundle adjustment yet. It reports residuals so you can see it struggle.
- **Ceiling heights** are ~1–2 cm off even on synthetic data.
- Scaling: assembly is O(rooms²) placements, fine for a house, not for a neighbourhood.
- **A single image cannot produce a room.** Measured: 29% area error, ceiling unobservable in
  0/150 frames. Multi-view capture is a requirement, not an optimisation.

## Next

1. Wire a real pose+depth backend (VGGT local, or a hosted endpoint) with the npz cache.
2. ZInD ingest → score against published baselines for real error numbers.
3. Room clustering + identity from listing photos.
4. Upgrade door detection (segmentation / VLM) — currently the #1 risk.
5. Depth-parallax 2.5D per room, then 3DGS hero nodes.

## Layout

```
pin3d/
  schema.py          Floorplan / Room / Opening, validation, JSON
  geometry.py        2D geometry (BLAS-free small-vector helpers included)
  pipeline/
    poses.py         PoseBackend protocol; Stub / VGGT / Hosted; npz cache
    planes.py        cloud → room box, wall-plane estimation, wall segments
    openings.py      wall-gap door detection + detector protocol
    floorplan.py     per-room placements → global 2D assembly
    reconstruct.py   orchestrator
    depth.py         ONNX monocular depth (Depth Anything V2 Small)
    real.py          real-photo harness: NYU loader, depth metrics, geometry A/B
  eval/
    metrics.py       IoU, corner distance, ceiling MAE, door P/R, adjacency F1
    zind.py          ZInD annotation adapter
  cli.py
viewer/floorplan.html
examples/            generated demo plan + ground truth
scripts/fetch_models.sh
tests/               92 tests, stdlib unittest
docs/BRAINSTORM.md   design record, decisions, open questions
```

## Viewer gotchas (already handled, worth knowing)

- A canvas accepts **one** context type for its lifetime. The 2D editor and the WebGL view use
  separate elements (`#view2d`, `#view3d`) — sharing one element makes `WebGLRenderer` fail with
  "Error creating WebGL context", which looks like a broken GPU but is a spec behaviour.
- `?renderer=canvas` forces the software renderer; `?renderer=webgl` forces WebGL.

## Note on macOS

numpy 2.0 + Accelerate BLAS raises spurious `divide by zero ... in matmul` warnings for
`(N, 2) @ (2,)` even with finite input. `pin3d.geometry` routes those through BLAS-free helpers;
the test suite runs with `-W error::RuntimeWarning` to keep it that way.
