# pic-to-3d — Design Brainstorm & Decision Record

> Working design doc for turning property photos into a 3D walkthrough.
> This is the pickup artifact: read this first, then §10 (phases), §14 (decision log) and
> §16 (open questions). Decisions are numbered (D1, D2, …) so later changes are cheap to discuss.

---

## 1. TL;DR

**Goal:** a baby-step PoC that takes a bunch of property photos and produces a
navigable 3D experience, to find out **how well the current tech actually works** on
real listing imagery.

**Decision (D1):** build a **floorplan-first** pipeline, not a geometry-first one.
Predict an editable 2D floorplan (rooms, polygons, ceiling height, doors), then attach
photography and extrude. The floorplan is the canonical representation, the eval target,
the asset format, and the navmesh.

**Decision (D2):** the realistic ceiling for sparse listing photos is **2.5D parallax /
room-box assembly**, not fused watertight reconstruction. The PoC's job is to *measure*
that ceiling, not to assume it away.

**Status:** v0 built. Schema, geometry, room-box fit, door detection, global assembly, metrics
and an editable viewer are implemented and tested (92 tests). A deterministic stub backend runs
the whole chain end-to-end with no GPU (§15). Real model backends (VGGT, depth) are typed
wiring; room clustering is deliberately not implemented yet.

---

## 2. Scope

**In scope**
- Indoor residential, **single floor**, **Manhattan world** (orthogonal walls, flat ceiling).
- Input: 360° panoramas *or* flat listing photos (2–4 per room).
- Output: `floorplan.json` (§11) + interactive viewer.
- Objective evaluation against ground truth.

**Out of scope (v0)**
- Multi-floor, stairs, split levels.
- Curved / 45° walls, vaulted ceilings, non-Manhattan geometry.
- Outdoor, large-scale geospatial, streaming LOD.
- Production crawling, auth, multi-tenant anything.

---

## 3. Problem framing: three different products

"3D walkthrough from pictures" decomposes into three products with ~10x cost differences.
The branch is decided by **how the photos were captured**.

| | Panorama nodes | Photogrammetry mesh | Neural (3DGS / NeRF) |
|---|---|---|---|
| What you get | Teleport between viewpoints | True geometry, free movement | Photo-real novel views |
| Capture | Rotation from a fixed point | Overlapping shots along a path | Overlapping shots, dense |
| Compute | Seconds (stitch) | Minutes–hours per scene | Minutes–hours GPU training |
| Delivery | 5–20 MB/node | 50–500 MB scene | 100 MB–1 GB scene |
| Mobile | Easy | Needs LOD/streaming | Heavy |
| Industry example | Street View, Zillow, Matterport | Sketchfab, drone maps | Luma, Polycam, Postshot |

**Diagnostic question:** did the photographer *rotate in place*, or *walk forward*?
- Rotation sets → panoramic stitching is valid.
- Forward motion → stitching into panos is geometrically invalid (parallax). Must go
  photogrammetry / neural.

Real captures usually mix both. **Property listing photos are neither:** they are 2–4
wide-angle, often HDR-bracketed, often lens-corrected shots per room, taken from opposite
corners. That is close to worst-case input for reconstruction (see §5).

### Option details considered

**A. Panorama nodes.** Equirect spheres + graph edges + transitions. Robust, small files.
Stitch: Hugin / PTGui / OpenCV `Stitcher`. View: Marzipano, Pannellum, three.js sphere with
inverted normals. Tile equirect → cube faces → multires. Transition: cut → dissolve →
optical-flow warp (Street View's "dolly" is a cylindrical warp; it's what makes a tour feel
spatial). Depth cheat: monocular depth per pano → 6-layer parallax "3D photo".

**B. Photogrammetry.** COLMAP / Glomap / Meshroom → dense cloud → Poisson → texture atlas →
decimate/bake → glTF + Draco/KTX2 → three.js / Babylon. Scale-free output needs a fiducial,
ARKit/ARCore poses, or a known dimension.

**C. Neural.** 3DGS is the default (`gsplat`, nerfstudio `splatfacto`, Postshot, Luma).
Viewers: `gsplat.js`, `@mkkellogg/gaussian-splats-3d`, PlayCanvas (native GS). Two blockers:
file size (needs SOG/compressed splats + streaming) and no collision (needs a separate
navmesh, or the camera flies through walls).

**Modern shortcut:** DUSt3R / MASt3R / VGGT are feed-forward pose+geometry predictors —
COLMAP is no longer mandatory for pose estimation. VGGT-class models degrade past a few
dozen views; Fast3R / Pi3 push the many-view regime.

---

## 4. Evaluation data: ZInD

**ZInD — Zillow Indoor Dataset** (CVPR 2021). Roughly **67k panoramas across ~5k real, empty
homes**, captured on iPhone with a gyro-stabilized panorama rig. Every home is
human-annotated with a **floor plan, room labels, and per-room 3D layout** (wall corners +
ceiling height). Public toolbox ships download scripts, panorama cropping, floorplan
extraction, and the official layout-eval code.

Why it matters: it **decouples the two hard problems**. ZInD gives the assembly problem for
free (ground-truth adjacency, poses derivable from the floorplan, rooms labeled), so the
renderer and solver can be built and debugged without fighting registration. Flat scraped
listing photos are then evaluated separately to see exactly what degrades.

Caveats:
- It is **panoramas, not flat listing photos** → validates the *tour and floorplan engine*,
  not the sparse-view reconstruction.
- Research / non-commercial license → PoC asset, not a product dependency.
- Siblings: **Structured3D** (~3.5k photoreal houses, 196k panoramas), **Matterport3D**,
  **HM3D**, **2D-3D-S**, **Replica**.

**Decision (D3):** start on ZInD panoramas so that objective error numbers exist from
commit one. Scraped flat photos are the second arm.

### Scraping reality check
The crawler is the least interesting and most expensive part: JS-rendered galleries,
lazy-loaded images, IP bans, and real ToS/litigation exposure (Zillow in particular).
**Decision (D4):** hand-download 5–10 properties (~200 photos) for the first arm. Automate
only after the renderer is worth automating.

---

## 5. What "assembly" actually decomposes into

There is no single model that eats photos and emits a house. Assembly is four problems:

1. **Poses** — every image's camera position + orientation in one common frame.
2. **Scale & gravity** — SfM is scale-free; need an up-vector (Manhattan assumption) and one
   known measurement (camera height ≈1.5 m, ceiling ≈2.4–3.0 m).
3. **Geometry** — depth per view → unproject → fuse.
4. **Topology** — which room connects to which, where the doorways are.

### Three fidelity levels

| Level | What | Cost | Risk |
|---|---|---|---|
| **1. Graph** | Each photo an independently-posed node; edges are transitions | Trivial | None — but no real parallax |
| **2. Room box** | Per-room Manhattan box from layout estimation; photos = wall textures; boxes stitched into a floorplan | Low | Needs correct doors; L-shaped rooms fail |
| **3. Fused** | One mesh / splat in one frame: poses → depth → TSDF/Poisson/3DGS | High | Sparse views hallucinate |

**Decision (D5):** Level 2 is the PoC's main path. Level 1 is the fallback demo. Level 3 is a
**measurement arm only** — run per room, tabulate failures, do not attempt whole-house fusion.

---

## 6. What models do and don't cover

**Covered now**
- **Poses / global alignment** — largely solved. VGGT (all poses in one forward pass), Fast3R,
  Pi3 for many views; MASt3R-SfM, Glomap for sparse/global SfM.
- **Dense geometry per view** — Depth Pro, MoGe, Depth Anything V2, VGGT point maps.
- **Fusion + appearance consistency** — 3DGS *is* the fusion step once poses exist; TSDF /
  Poisson are the deterministic classical alternatives.

**Not covered**
- **Building-level understanding**: room identity across photos, doorway connections, loop
  closure, floorplan, navmesh. This is the part a walkthrough product actually needs.
- **The last mile**: cleanup, UV/texture baking, decimation, collision, streaming.
- **Reliable global drift control** across rooms — pose-graph/BA is still your code.

### The inversion (the important reframe)
Two years ago fusion was the hard part because SfM was fragile. Now fusion is `gsplat` on
VGGT poses — a one-liner. **The bottleneck moved from the pipeline to the input data.**
That is bad news specifically for sparse listing photos, and it is *why* this PoC is worth
running: the answer depends on real data, not on model quality.

### The trap
With 2–4 wide-angle photos per room these models **do not fail loudly — they hallucinate
plausible geometry.** Demos look impressive; numbers lie. Hence objective evaluation on
ground truth (D3), not vibes.

---

## 7. The decision: floorplan-first

**D1 rationale.** Predict the floorplan, then attach imagery to it, rather than reconstruct
geometry and hope it coagulates into a building.

What it buys:
1. **Assembly becomes well-posed.** Rooms sit on a plane → per room, translation + yaw + box.
   That is a pose-graph, not a house-scale bundle adjustment with unknown scale.
2. **Asset size collapses** — KB of JSON, not GB of splats.
3. **Topology for free** — adjacency + doors = navmesh + walk transitions.
4. **Shippable on its own** — Zillow/CubiCasa sell the floorplan; 3D is the upsell.
5. **Constraint loop improves geometry** — walls snap to the polygon, killing drift and
   floaters. Floorplan-first is strictly better 3D than geometry-first + floorplan bolted on.
6. **Objective evaluation** — polygon IoU, corner distance, ceiling error, door count/position,
   adjacency-graph edit distance. Comparable to published ZInD baselines.

What it costs:
- **Manhattan-world assumption is baked in.** Curved walls, 45° bays, split levels, vaulted
  ceilings, stairs all break. Accepted for v0 (§2).
- Rooms must be **rectangular** for the box fit; L-shapes need the wall-segment path.
- **Auto-floorplans are never right** → the floorplan must be **human-editable**. A room you
  can fix in 10 seconds is worth more than a 3D you cannot fix at all.

---

## 8. Where the floorplan comes from

| Source | Accuracy | Needs | Verdict |
|---|---|---|---|
| Panoramas + Manhattan layout models (HorizonNet, LayoutMP3D) | Highest, well-benchmarked | 360 panos | ZInD arm |
| **VGGT point maps → RANSAC planes → Manhattan snap → walls → polygons** | Good | Poses only | **Default for flat photos** |
| Per-image layout models (Total3D, LED2-Net) | Weak — one view can't see the room | Perspective photos | Fallback |
| Read the floorplan off the listing | N/A | Listing has one | Product shortcut; **defeats measurement**, don't build now |

**D6:** hybrid — geometry-derived walls regularized by a topological prior from photo order /
VLM ("kitchen adjacent to living room"), scored against ZInD.

---

## 9. Critical path, ranked by risk

1. **Door / window detection & placement — the #1 risk.** A floorplan without correct
   openings is decorative. Approach: point-cloud wall gap analysis (implemented), plus
   segmentation (SAM / OneFormer door classes) or a VLM pass.
2. **Room identity** — are these 3 photos the *same* room from different corners? Must be
   linked for a consistent box. Approach: DINOv2/CLIP embeddings + clustering, or a VLM
   labeling `{room, viewpoint}` (cheaper and more accurate at this scale).
3. **Metric box per room** — monocular scale ambiguity; ceiling height is the best anchor.
4. **Global 2D solve** — rooms as nodes, doors as relative constraints, global least squares
   with Manhattan regularization. Errors compound → must be global, not pairwise.
5. **Level-3 failure rate** — expected VGGT partial success on maybe 30–50% of rooms; COLMAP
   and 3DGS mostly failing on 2–4 wide-angle views. Measure, don't guess.

---

## 10. Phased plan

| Phase | Deliverable | Status |
|---|---|---|
| 0 | This doc, schema, geometry + assembly + eval libraries, stub end-to-end | **done** |
| 1 | Viewer: editable 2D floorplan + extruded 3D walk | **done (v0)** |
| 2 | Real depth backend (ONNX, no torch) + cache | **done** |
| 3 | Real-data eval vs. ground truth ▶ ZInD is gated; used NYU Depth V2 instead (§15b) | **done** |
| 4 | Room clustering / identity from flat listing photos | next |
| 5 | Door detection upgraded (segmentation / VLM) | next |
| 6 | Depth-parallax 2.5D upgrade per room | later |
| 7 | 3DGS hero nodes (real parallax where users linger) | later |
| 8 | Crawler + authoring UI | later, and last |

Sequencing rule that came out of the discussion: **get signal in a day, not a week.**
Hand-pick photos → build renderer → then automate. The crawler is a distraction from the
actual question ("does this feel 3D / is the floorplan right?").

---

## 11. Artifact schema

`floorplan.json` — simultaneously the database, the eval target, the editor model, and the
navmesh input. Units are metres everywhere; polygons are CCW; `openings[].t` is the position
along the wall as a fraction 0..1, `wall` indexes into the room's polygon edges.

```json
{
  "units": "meters",
  "scale": 1.0,
  "rooms": [{
    "id": "r1", "label": "kitchen",
    "polygon": [[0,0],[4.2,0],[4.2,3.1],[0,3.1]],
    "ceiling": 2.7,
    "photos": ["p12.jpg","p13.jpg"],
    "openings": [{"wall": 0, "t": 0.42, "width": 0.9, "type": "door", "head": 2.05}],
    "confidence": 0.82
  }],
  "adjacency": [{"from": "r1", "to": "r2", "via": "r1:0"}],
  "meta": {"method": "box_fit+door_gap", "backend": "stub", "notes": []}
}
```

Errors are expected to be **cheap to repair by hand**, which is why the schema is
deliberately dumb: explicit polygons, explicit wall indices, no implicit derivation.

---

## 12. Metrics (implemented in `pin3d/eval`)

- Room polygon **IoU** (greedy room matching, then mean).
- **Corner distance** — symmetric Chamfer between vertex sets.
- **Ceiling height** MAE.
- **Door precision/recall** — matched by midpoint distance < 0.5 m.
- **Adjacency edit distance** — symmetric difference over room-pair edges.
- Assembly **overlap area** between rooms (a floorplan that interpenetrates is wrong even if
  each room is right).
- Level-3 arm: per-room success rate, reprojection error, train time, asset size, floater
  severity.

---

## 13. Repo layout

```
pin3d/
  schema.py            Floorplan / Room / Opening dataclasses, JSON round-trip, validation
  geometry.py          polygon area/IoU/Chamfer, convex hull, min-area rect, rectification
  pipeline/
    poses.py           PoseBackend protocol; Stub / VGGT / Depth-only / Hosted; npz cache
    planes.py          point cloud → room box, RANSAC wall segments, floor/ceiling
    openings.py        wall-gap door detection + detector protocol (segmentation/VLM stubs)
    floorplan.py       per-room placements → global 2D assembly (BFS + 4 yaw options)
  eval/
    metrics.py         IoU, corner dist, ceiling MAE, door P/R, adjacency distance
    zind.py            defensive ZInD annotation adapter → canonical Floorplan
  cli.py               reconstruct / stub / eval
viewer/floorplan.html  editable 2D floorplan + extruded 3D walkthrough (single file)
examples/              stub-generated floorplan.json + ground truth
tests/                 geometry, openings, assembly, eval, schema, end-to-end
```

---

## 14. Decision log

| # | Decision | Rejected alternative |
|---|---|---|
| D1 | Floorplan-first | Geometry-first with floorplan bolted on |
| D2 | 2.5D parallax / room boxes is the target ceiling | Whole-house fused mesh |
| D3 | Start on ZInD for objective numbers | Start on scraped photos |
| D4 | Hand-download first arm, crawl later | Build the crawler first |
| D5 | Level 2 (room box) is the main path | Level 3 (fused) as the main path |
| D6 | Hybrid: geometry walls + topological prior | Single-source floorplan |
| D7 | Keep model inference behind a `PoseBackend` protocol; stub runs end-to-end with no GPU | Require GPU to run anything |
| D8 | Plain dataclasses + numpy, shapely optional (raster fallback) | Pydantic + heavy stack |

---

## 15. v0 baselines (measured)

Synthetic two-room house, `python -m pin3d.cli stub`. **This validates the plumbing, not the
tech on real photos** — it is the number the real arm will be compared against.

| metric | value |
|---|---|
| rooms matched | 2 / 2 |
| polygon IoU | 1.000 |
| corner distance | 0.000 m |
| ceiling MAE | 0.016 m |
| door P / R / F1 | 1.000 / 1.000 / 1.000 |
| adjacency F1 | 1.000 |
| interpenetration | 0.000 m² |

Four real bugs were found and fixed getting here, all worth remembering because they will
recur on real data:

1. **Min-area rects are not robust to outliers.** 0.4% flyer points inflated a 14.4 m² room to
   20.5 m². Fix: grid-occupancy density filter (25 cm cell, ≥3 points) before fitting.
2. **Percentile trimming does not work on wall coordinates.** The coordinate distribution is a
   *spike at each wall plus a uniform interior* (the perpendicular walls span the range), so a
   1–2% trim trims the wrong thing. Fix: median of the outer 15% window, where the wall spike
   dominates by count and noise is unbiased.
3. **A doorway split by one stray point is lost.** Fix: merge empty runs across ≤1 occupied bin.
4. **A degenerate assembly score glues a doorway to a solid wall.** With equal wall contact and
   no penalty, "door meets wall" scored marginally better than "door meets door". Fix: explicit
   door-mismatch penalty plus a shared-wall contact reward.

Also: on macOS, numpy 2.0 + Accelerate BLAS emits spurious `divide by zero ... in matmul` for
`(N,2) @ (2,)` on finite input — hence the BLAS-free helpers in `geometry.py`.

Viewer lessons (both cost a debugging round-trip because the first error message blamed the
environment):

5. **A canvas holds exactly one context type.** The 2D editor called `getContext('2d')` on
   page load, so handing that same element to `WebGLRenderer` could never work — it surfaced as
   "Error creating WebGL context", which reads like a broken GPU. Fix: separate `#view2d` and
   `#view3d` elements.
6. **Do not let the demo depend on a CDN or on WebGL.** three.js is vendored under
   `viewer/vendor/`, and there is a Canvas2D painter's-algorithm fallback (near-plane clipping,
   flat shading, distance fog) so the 3D view renders even with WebGL unavailable. A fallback
   plus a verified-by-headless-test renderer beats a better renderer that might not start.
7. **Error overlays must print the real exception.** Both failures were misdiagnosed because
   the catch block substituted a plausible story for `err.message`.
8. **Do not choose an evaluation convention per sample.** Letting each scene pick "depth vs
   disparity" whichever scored better turned an honest 0.171 into an inflated number and hid
   the fact that 34/80 scenes were being fit through a reciprocal. Choose it once, globally,
   and report both scores.

### 15b. Real photos, real depth (NYU Depth V2, 150 rooms)

ZInD turned out to be **gated** (the public S3 bucket 404s; access needs an accepted
agreement), so the real arm uses **NYU Depth V2** via a HuggingFace mirror: 480x640 RGB with
float32 metric depth, 171 distinct rooms among the first 200 frames. Model: **Depth Anything V2
Small, ONNX, on CoreML/CPU** — no torch, 150 rooms in 95 s.

Depth is affine-invariant, so it is aligned to ground truth before measuring (scale+shift), and
the output convention is chosen **once globally** — choosing it per image is overfitting a
flexible transform, and doing so flipped 34/80 scenes to a reciprocal fit and silently
flattered the numbers.

| metric | value |
|---|---|
| AbsRel (affine-aligned) | **0.171** |
| RMSE | 0.688 m |
| delta<1.25 / delta<1.25^2 | 0.813 / 0.937 |
| output convention | depth (0.171) beats disparity (0.204) |
| room box solved | 100% (GT depth and model depth) |
| room area | GT 17.1 m² vs model 12.5 m² → **29% relative error** |
| wall support / confidence | GT 0.400 vs model 0.403 |
| ceiling observable from **one** view | **0 / 150 scenes** |

Four findings, in order of how much they change the plan:

1. **The depth model is good enough to be useful.** 17% AbsRel on real rooms is roughly a
   17 cm error on a 1 m object, and it costs 0.6 s/room on CPU. This is not the bottleneck.
2. **Extent geometry amplifies depth error.** 17% depth error becomes 29% room-area error,
   because a box is defined by the extreme points and inherits the error tail. Median-based
   wall-plane estimation mitigates but cannot remove this.
3. **Ceiling height is not recoverable from one view.** Not one of 150 real single frames had
   ≥5% of its points on a top plane. The NYU captures are often tilted down, and ceilings are
   joists or out of frame. Any single-view "ceiling height" is measuring the top of the
   frustum, so the harness now refuses to report it (`ceiling_valid`) and the exported plans
   say `"ceiling_source": "assumed 2.7 m"` rather than inventing one.
4. **The model looks more confident than the ground truth.** On the exported scenes the model's
   boxes score 0.44–0.48 wall support against GT's 0.33–0.38, *while being 5–6 m² smaller*.
   Smoother depth produces a tidier fit without being more correct — this is §6's "fails
   plausibly" trap, now a measured number rather than a warning.

**Net:** the theorem in §6 holds up on real data. Model quality is no longer the constraint;
**capture geometry is.** Level 2 (room box) is sound, but it requires multiple views per room —
a single frame cannot close a room, see its ceiling, or separate it from the room next door.

### 15c. Sample photos, no ground truth (7 CC-licensed interiors)

Metric scale has no GT here, so it is recovered from a **floor plane plus a 1.5 m camera-height
prior**: for floor pixels ``r_i * (a * pred_i + b) = h`` is homogeneous and linear in
``(a, b, h)``, so an SVD gives the affine depth correction and the scale at once. Validated on
NYU against GT: **AbsRel 0.302**, versus 0.171 when aligned to GT directly. The prior costs ~13
points but is the difference between "relative" and "metres".

It fails in a specific, instructive way. `sample10` — an empty apartment shot from a doorway,
true size roughly 8 x 5 m — came out **5.7 x 1.4 m with 0.88 wall support**. A foreground plant
filled the bottom band, the floor fit locked onto the plant, and every downstream number
inherited the error. The only signal that flagged it was **floor inlier ratio 31%**, against
75–88% for the photos that came out plausible. Confidence from the *geometry* (0.88) was
actively misleading; confidence from the *scale fit* was correct.

Also **0 doors detected across all 7 photos**. The gap detector needs a closed room boundary
that a single view never provides.

Two things this settles:

1. **A confidence number only helps if it is attached to the thing that actually failed.**
   Wall support was high precisely on the worst result.
2. **Do not tune geometry against a handful of images.** A near-field exclusion search was
   tried to rescue the occluded cases; it improved some photos, made others worse, and was
   reverted. Seven images are not an evaluation set — that is what §15b's 150 rooms are for.

### 15d. Multi-view, actually measured

Claim under test: single views cannot recover a room, multiple views fix it.

**NYU cannot answer this.** Its labelled frames are a slow handheld *pan*: of 313 verified
pairs, **150 are classified PANORAMIC** (pure rotation, zero baseline) at every stride from 1
to 12, and the mapper returns **6 points** from 4 images. Sampling a walk-through densely
produced a rotation-only dataset, which is the same failure mode as listing photography.

So the experiment moved to **TUM RGB-D fr1_room** — real Kinect RGB-D with ground-truth poses
*and* metric depth. 200 synchronised frames, trajectory extent 1.13 x 1.41 x 0.19 m, baselines
0.36-1.17 m against a 4.72 m median scene depth, view overlap 0.63-0.78.

| arm | single view | 4-view fused |
|---|---|---|
| GT depth | 1.2 m², support 0.95, 0 doors | 8.5 m² (+620%), support 0.44, **1 door** |
| model depth | 14.8 m², support 0.49, 0 doors | 15.5 m² (+5%), support 0.43, 0 doors |
| held-out view | box explains **0.00** of an unseen view | **0.08** |

Three findings, and the third one matters most:

1. **Multi-view does what theory says** — given real baseline, the footprint grows 7x and the
   first door appears. Coverage, not accuracy, is what one view is missing.
2. **Single-view "wall support" is self-fulfilling and must be retired.** It reads 0.95 on a
   1.2 m² sliver because the box *is* the min-area rectangle of exactly those points. Only
   held-out support is meaningful, and it is ~0 for both.
3. **The room-box abstraction is the actual blocker, not the view count.** With perfect poses
   *and* perfect depth, a box fitted on three views still explains only 8% of a fourth. More
   views will not fix that; a richer room model (wall planes, not a bounding rectangle) will.

SfM at this scale is also weaker than "solved" implies: **5 frames fail outright, 10 frames
register 2 with 46 points, 20 frames register 10 with 569 points.** Multi-view reconstruction
is solved at *dataset* scale (hundreds of views); it is not solved at "five photos of a room",
which is exactly the listing regime.

Three bugs found while building this, all of the kind that silently produce wrong numbers:

- **Y-flip mismatch.** ``cloud_from_depth`` flips Y for our +Y-up convention; feeding the
  result through TUM's y-down camera pose mirrored the scene and collapsed a 3 m room to
  0.87 m wide. The experiment now unprojects in the dataset's own convention and flips once,
  at the end.
- **``cam_from_world`` is a method, not a property** in pycolmap 3.13 — the wrapper raised
  ``AttributeError`` inside a ``try`` and reported "no reconstruction" for four runs.
- **Comparing absolute rotations across SfM and ground truth** is meaningless because SfM has
  an arbitrary global frame; it reported 139 degrees of "error". Needs a Procrustes alignment
  first — or, better, don't report rotation error at all until there are enough views to align.

## 16. Open questions

1. **Compute** — partially answered: depth needs no GPU (ONNX on CoreML/CPU, 0.6 s/room). Still
   open for the *poses* backend (VGGT), which will want a GPU or a hosted endpoint.
   *(user has not answered; current code assumes nothing)*
2. **Deliverable shape** — interactive web viewer vs. pre-rendered MP4 walkthroughs? Videos
   are much easier to A/B and to show people.
3. Is the real product **self-serve on arbitrary user photos** (must be fully automatic) or a
   **tool** where bad scenes get hand-fixed? This decides how much the editor matters.
4. Acceptable **metric scale error** — is a 5%-off floorplan acceptable to users?
5. Does the business need the floorplan as a **primary deliverable** (then accuracy targets
   tighten and multi-floor becomes mandatory) or purely as internal scaffolding for 3D?
