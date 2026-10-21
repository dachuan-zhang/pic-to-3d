"""Camera pose / depth backends.

Design rule (D7): **nothing in this repo requires a GPU to run.** Model inference sits behind
``PoseBackend``; ``StubBackend`` produces a synthetic, self-consistent scene so the geometry,
assembly and eval layers are testable today, and swapping in a real model is a one-line change.

Status of the real backends: ``VGGTBackend`` is written against the published VGGT API but has
**not been executed here** (no torch in this environment, no weights). Treat it as wiring:
expect to fix an import name or two on first run. ``HostedBackend`` is an interface only.
"""
from __future__ import annotations

import hashlib
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Protocol

import numpy as np


@dataclass
class Poses:
    """Per-image cameras + depth, plus optionally pre-segmented clouds."""

    image_names: List[str] = field(default_factory=list)
    K: np.ndarray = field(default_factory=lambda: np.zeros((0, 3, 3)))
    T: np.ndarray = field(default_factory=lambda: np.zeros((0, 4, 4)))  # camera -> world
    depth: Optional[np.ndarray] = None  # (N, h, w) metres, camera Z
    conf: Optional[np.ndarray] = None
    backend: str = "unknown"
    room_of_image: Optional[List[str]] = None
    #: Clouds that are already attributed to a room (stub backends, or a backend that has a
    #: room-identity prior). Real backends leave this None -- room clustering is a separate
    #: problem (see docs/BRAINSTORM.md §9 item 2).
    clouds: Optional[Dict[str, np.ndarray]] = None

    @property
    def n(self) -> int:
        return len(self.image_names)

    def intrinsics(self, i: int) -> np.ndarray:
        k = np.asarray(self.K)
        return k[i] if k.ndim == 3 else k

    def unproject(self, i: int, stride: int = 1) -> np.ndarray:
        """Depth map -> world-frame points, shape (P, 3)."""
        if self.depth is None:
            raise ValueError("no depth for this backend")
        depth = np.asarray(self.depth[i], dtype=float)[::stride, ::stride]
        h, w = depth.shape
        k = self.intrinsics(i)
        fx, fy, cx, cy = k[0, 0], k[1, 1], k[0, 2], k[1, 2]
        u, v = np.meshgrid(np.arange(w) * stride, np.arange(h) * stride)
        z = depth
        valid = np.isfinite(z) & (z > 0)
        x = (u[valid] - cx) / fx * z[valid]
        y = (v[valid] - cy) / fy * z[valid]
        cam = np.stack([x, y, z[valid]], axis=1)
        t = np.asarray(self.T[i], dtype=float)
        return cam @ t[:3, :3].T + t[:3, 3]

    def pointcloud(self, stride: int = 2) -> np.ndarray:
        if self.clouds is not None:
            return np.vstack(list(self.clouds.values())) if self.clouds else np.zeros((0, 3))
        if self.depth is None:
            return np.zeros((0, 3))
        return np.vstack([self.unproject(i, stride) for i in range(self.n)])

    # -- cache ------------------------------------------------------------
    def save(self, path: str) -> None:
        payload: Dict[str, Any] = {
            "image_names": np.array(self.image_names, dtype=object),
            "K": self.K,
            "T": self.T,
            "backend": np.array(self.backend),
        }
        if self.depth is not None:
            payload["depth"] = self.depth
        if self.conf is not None:
            payload["conf"] = self.conf
        if self.room_of_image is not None:
            payload["room_of_image"] = np.array(self.room_of_image, dtype=object)
        np.savez_compressed(path, **payload)

    @classmethod
    def load(cls, path: str) -> "Poses":
        with np.load(path, allow_pickle=True) as z:
            room = z["room_of_image"].tolist() if "room_of_image" in z else None
            return cls(
                image_names=[str(s) for s in z["image_names"].tolist()],
                K=z["K"],
                T=z["T"],
                depth=z["depth"] if "depth" in z else None,
                conf=z["conf"] if "conf" in z else None,
                backend=str(z["backend"]),
                room_of_image=room,
            )


class PoseBackend(Protocol):
    name: str

    def estimate(self, images: Sequence[Path]) -> Poses:
        ...


# --------------------------------------------------------------------------
# cache
# --------------------------------------------------------------------------
def cache_key(images: Sequence[Path]) -> str:
    h = hashlib.sha1()
    for p in images:
        p = Path(p)
        try:
            st = p.stat()
            h.update(f"{p.resolve()}:{st.st_size}:{st.st_mtime_ns}".encode())
        except OSError:
            h.update(str(p).encode())
    return h.hexdigest()[:16]


def load_or_estimate(
    backend: PoseBackend, images: Sequence[Path], cache_dir: Optional[str] = None
) -> Poses:
    if not cache_dir:
        return backend.estimate(images)
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{getattr(backend, 'name', 'backend')}-{cache_key(images)}.npz")
    if os.path.exists(path):
        return Poses.load(path)
    poses = backend.estimate(images)
    poses.save(path)
    return poses


# --------------------------------------------------------------------------
# stub: synthetic scene, no model, no GPU
# --------------------------------------------------------------------------
@dataclass
class StubScene:
    """A two-room synthetic house with one shared doorway."""

    room_a: Dict[str, Any] = field(default_factory=lambda: {
        "id": "r_living", "label": "living_room",
        "lo": (0.0, 0.0, 0.0), "hi": (3.6, 2.6, 4.0),
    })
    room_b: Dict[str, Any] = field(default_factory=lambda: {
        "id": "r_kitchen", "label": "kitchen",
        "lo": (3.6, 0.0, 0.0), "hi": (7.0, 2.6, 3.0),
    })
    #: shared wall plane x=3.6, opening span on Z, height range
    doorway: Dict[str, Any] = field(default_factory=lambda: {
        "axis": 0, "at": 3.6, "z_range": (1.20, 2.10), "height": 2.05,
    })
    density: float = 70.0  # points per m2 of wall
    noise: float = 0.012

    def rooms(self) -> List[Dict[str, Any]]:
        return [self.room_a, self.room_b]

    def _in_doorway(self, pts: np.ndarray) -> np.ndarray:
        d = self.doorway
        z0, z1 = d["z_range"]
        return (
            (np.abs(pts[:, d["axis"]] - d["at"]) < 0.06)
            & (pts[:, 2] >= z0) & (pts[:, 2] <= z1)
            & (pts[:, 1] <= d["height"])
        )

    def sample_room(self, room: Dict[str, Any], rng: np.random.Generator) -> np.ndarray:
        lo = np.asarray(room["lo"], dtype=float)
        hi = np.asarray(room["hi"], dtype=float)
        out: List[np.ndarray] = []
        # walls: 4 vertical planes
        for axis in (0, 2):
            other = 2 if axis == 0 else 0
            span = hi[other] - lo[other]
            area = span * (hi[1] - lo[1])
            n = max(16, int(area * self.density))
            along = rng.uniform(lo[other], hi[other], n)
            vert = rng.uniform(lo[1], hi[1], n)
            for val in (lo[axis], hi[axis]):
                pts = np.zeros((n, 3))
                pts[:, axis] = val
                pts[:, other] = along
                pts[:, 1] = vert
                out.append(pts)
        # floor + ceiling
        area = (hi[0] - lo[0]) * (hi[2] - lo[2])
        n = max(64, int(area * self.density * 0.5))
        xs = rng.uniform(lo[0], hi[0], n)
        zs = rng.uniform(lo[2], hi[2], n)
        for val in (lo[1], hi[1]):
            pts = np.stack([xs, np.full(n, val), zs], axis=1)
            out.append(pts)

        pts = np.vstack(out)
        pts = pts[~self._in_doorway(pts)]
        pts += rng.normal(0.0, self.noise, pts.shape)
        # a little outlier clutter, because real clouds have it
        n_out = max(1, int(0.004 * len(pts)))
        pts = np.vstack([pts, rng.uniform([0, 0, 0], [7.0, 2.6, 4.0], (n_out, 3))])
        return pts


class StubBackend:
    """Deterministic synthetic scene provider (not a model).

    Exists so the geometry / opening / assembly / eval layers can be exercised end-to-end
    without torch, weights or a GPU. ``clouds`` is pre-segmented per room because the scene is
    synthetic; a real backend must solve room identity separately.
    """

    name = "stub"

    def __init__(self, scene: Optional[StubScene] = None, seed: int = 7) -> None:
        self.scene = scene or StubScene()
        self.seed = seed

    def estimate(self, images: Sequence[Path] = ()) -> Poses:
        rng = np.random.default_rng(self.seed)
        clouds: Dict[str, np.ndarray] = {}
        names: List[str] = []
        room_of: List[str] = []
        ks: List[np.ndarray] = []
        ts: List[np.ndarray] = []
        for room in self.scene.rooms():
            clouds[room["id"]] = self.scene.sample_room(room, rng)
            lo = np.asarray(room["lo"], dtype=float)
            hi = np.asarray(room["hi"], dtype=float)
            centre = 0.5 * (lo + hi)
            for k in range(2):  # two photos per room, opposite corners
                sign = -1.0 if k == 0 else 1.0
                pos = centre + sign * np.array([0.0, 0.0, (hi[2] - lo[2]) / 4.5])
                pos[1] = 1.55
                t = np.eye(4)
                t[:3, 3] = pos
                ts.append(t)
                ks.append(np.array([[320.0, 0.0, 240.0], [0.0, 320.0, 180.0], [0.0, 0.0, 1.0]]))
                names.append(f"{room['id']}_view{k}.jpg")
                room_of.append(room["id"])
        order = np.argsort(names)
        return Poses(
            image_names=[names[i] for i in order],
            K=np.stack([ks[i] for i in order]),
            T=np.stack([ts[i] for i in order]),
            depth=None,
            backend="stub",
            room_of_image=[room_of[i] for i in order],
            clouds=clouds,
        )


# --------------------------------------------------------------------------
# real backends (wiring, not executed in this environment)
# --------------------------------------------------------------------------
def _select_device(preference: str = "auto") -> str:
    if preference != "auto":
        return preference
    try:
        import torch  # noqa: F401

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


class VGGTBackend:
    """Feed-forward poses + depth + point maps (VGGT).

    UNVERIFIED in this environment: written against the published API, never executed here.
    VGGT attention is roughly O(N^2) in views, so long captures are chunked; cross-chunk
    alignment is **not** implemented (it would need a shared view or a room-level merge).
    """

    name = "vggt"

    def __init__(self, device: str = "auto", max_views: int = 24, image_size: int = 518) -> None:
        self.device = device
        self.max_views = max_views
        self.image_size = image_size

    def estimate(self, images: Sequence[Path]) -> Poses:
        try:
            import torch
            from vggt.models.vggt import VGGT
            from vggt.utils.load_fn import load_and_preprocess_images
        except Exception as exc:  # pragma: no cover - dependency path
            raise ImportError(
                "VGGTBackend needs torch + the vggt package:\n"
                "  pip install torch torchvision\n"
                "  pip install git+https://github.com/facebookresearch/vggt.git\n"
                "Or use StubBackend / HostedBackend."
            ) from exc

        images = list(images)
        if len(images) > self.max_views:
            raise NotImplementedError(
                f"{len(images)} views > max_views={self.max_views}. Chunk by room before "
                "calling (see docs/BRAINSTORM.md §6: per-room reconstruction, then stitch)."
            )

        device = _select_device(self.device)
        model = VGGT.from_pretrained("facebook/vggt-1b").to(device).eval()
        batch = load_and_preprocess_images([str(p) for p in images]).to(device)
        with torch.no_grad():
            out = model(batch)

        # absT: (N, 3, 3) rotation + (N, 3) translation; quat: (N, 4) xyzw
        ext = out["pose_enc"][..., :3]
        quat = out["pose_enc"][..., 3:7]
        rot = _quat_to_matrix(quat.detach().cpu().numpy())
        trans = ext.detach().cpu().numpy()
        T = np.tile(np.eye(4), (len(images), 1, 1))
        T[:, :3, :3] = rot
        T[:, :3, 3] = trans
        depth = out["depth"].detach().cpu().numpy().squeeze(-1)  # (N, h, w)

        h, w = depth.shape[1:]
        K = np.tile(np.array([[0.45 * w, 0, w / 2], [0, 0.45 * w, h / 2], [0, 0, 1]]), (len(images), 1, 1))
        return Poses(
            image_names=[Path(p).name for p in images],
            K=K, T=T, depth=depth, backend="vggt",
        )


def _quat_to_matrix(quat: np.ndarray) -> np.ndarray:
    """xyzw -> (N,3,3) rotation matrices."""
    q = np.asarray(quat, dtype=float)
    q = q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-9)
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], axis=-1),
        np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], axis=-1),
        np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], axis=-1),
    ], axis=-2)


class HostedBackend:
    """Interface for a hosted inference endpoint (Replicate / fal / your own service).

    Set ``PIC3D_HOSTED_URL`` and ``PIC3D_HOSTED_TOKEN``. Not implemented: the request schema
    depends on which model you host, and inventing one would be worse than leaving it explicit.
    """

    name = "hosted"

    def __init__(self, url: Optional[str] = None, token: Optional[str] = None) -> None:
        self.url = url or os.environ.get("PIC3D_HOSTED_URL")
        self.token = token or os.environ.get("PIC3D_HOSTED_TOKEN")

    def estimate(self, images: Sequence[Path]) -> Poses:
        if not self.url:
            raise RuntimeError("set PIC3D_HOSTED_URL (and PIC3D_HOSTED_TOKEN) to use HostedBackend")
        raise NotImplementedError(
            "HostedBackend is an interface only. POST the images, expect {K, T, depth} and "
            "return Poses(..., backend='hosted'). See docs/BRAINSTORM.md §15 item 1."
        )


def get_backend(name: str, **kwargs: Any) -> PoseBackend:
    name = name.lower()
    if name == "stub":
        return StubBackend(**kwargs)
    if name == "vggt":
        return VGGTBackend(**kwargs)
    if name == "hosted":
        return HostedBackend(**kwargs)
    raise ValueError(f"unknown backend {name!r} (choose stub|vggt|hosted)")
