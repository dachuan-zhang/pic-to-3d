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
and an editable viewer are implemented and tested (81 tests). A deterministic stub backend runs
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
| 2 | Wire a real poses backend (VGGT or hosted) + depth, cache | **next** |
| 3 | ZInD ingest + eval vs. published baselines | next |
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

## 15. v0 baseline (measured)

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

## 16. Open questions

1. **Compute** — GPU available locally, or target a hosted endpoint (Replicate/fal) for
   VGGT and depth? *(unanswered; current code assumes nothing)*
2. **Deliverable shape** — interactive web viewer vs. pre-rendered MP4 walkthroughs? Videos
   are much easier to A/B and to show people.
3. Is the real product **self-serve on arbitrary user photos** (must be fully automatic) or a
   **tool** where bad scenes get hand-fixed? This decides how much the editor matters.
4. Acceptable **metric scale error** — is a 5%-off floorplan acceptable to users?
5. Does the business need the floorplan as a **primary deliverable** (then accuracy targets
   tighten and multi-floor becomes mandatory) or purely as internal scaffolding for 3D?
