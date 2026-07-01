"""3D extrinsic calibration math: camera-frame XYZ → robot base XYZ (T_base_cam).

Ported from farino_app/calibrate_camera_to_base_3d.py. Unlike a planar homography
(pixel → base XY on ONE plane), this fits a full 3D rigid transform, so the
336L's real per-pixel depth drives the robot in base coordinates at ANY height.

How: collect 3D↔3D correspondences (click the marker in the image for the camera
XYZ, touch it with the TCP for the base XYZ) and solve base ≈ s·R·cam + t with
the Umeyama algorithm. The scale absorbs any depth unit error and is folded into
the 3x3 block of the returned 4x4 T_base_cam.

Point-collection rules (enforced by warnings, not hard errors):
* Collect at least MIN_POINTS pairs, spread in X, Y AND HEIGHT — coplanar points
  leave the Z fit unconstrained (this bit us before: a flat-table capture solved
  fine but was degenerate).
* The residual gate flags any point worse than RESID_GATE_MM so a few sloppy
  touches can be dropped and refit instead of dragging the whole fit.

Everything here is hardware-free (numpy only) so it can be unit-tested; frame
grabbing and depth sampling inputs come from the camera worker.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Default output files (working directory, same convention as farino_app).
OUT_FILE = Path("T_base_cam.npy")
META_SUFFIX = ".meta.json"     # sidecar with fit quality, read by load guards

DEPTH_PATCH = 5        # NxN median depth patch around a click (robust to holes)
MIN_POINTS = 6         # recommended; >=3 non-coplanar needed to solve at all
RESID_GATE_MM = 4.0    # flag points whose residual exceeds this before saving
MIN_SOLVE = 4          # never drop below this many points (keep the fit redundant)
Z_SPREAD_WARN_MM = 20.0  # base Z spread under this = nearly coplanar capture


def sample_depth_mm(depth_mm: np.ndarray, uv, patch: int = DEPTH_PATCH) -> float:
    """Median depth (mm) in a patch around pixel (u, v); 0.0 if all holes."""
    h, w = depth_mm.shape
    u, v = int(round(uv[0])), int(round(uv[1]))
    r = patch // 2
    win = depth_mm[max(0, v - r):v + r + 1, max(0, u - r):u + r + 1]
    valid = win[win > 0]
    return float(np.median(valid)) if valid.size else 0.0


def pixel_to_camera(uv, z_mm: float, intr) -> np.ndarray:
    """Deproject pixel (u, v) at depth z (mm) to camera-frame XYZ (mm).

    ``intr`` needs fx/fy/cx/cy attributes (the SDK's rgb_intrinsic works as-is).
    """
    u, v = uv
    x = (u - intr.cx) * z_mm / intr.fx
    y = (v - intr.cy) * z_mm / intr.fy
    return np.array([x, y, z_mm])


def endpoint_3d(p_px, depth_mm: np.ndarray, intr,
                patch: int = DEPTH_PATCH) -> np.ndarray | None:
    """Camera-frame XYZ (mm) for a pixel endpoint, or None if depth is a hole.

    Same as farino_app red_line_viewer: median depth patch at the pixel, then
    deproject through the RGB intrinsics (depth must be ALIGNED to color).
    """
    z = sample_depth_mm(depth_mm, p_px, patch)
    if z <= 0:
        return None
    return pixel_to_camera(p_px, z, intr)


def line_to_camera(endpoints, depth_mm: np.ndarray, intr,
                   patch: int = DEPTH_PATCH) -> tuple:
    """Camera-frame XYZ for both detected line endpoints.

    Returns (cam1, cam2); an endpoint is None when the depth there is a hole
    (common right at a paint edge — retake or nudge the AOI). Feed non-None
    results through ``cam_to_base`` with the calibrated T_base_cam to get the
    robot targets.
    """
    p1, p2 = endpoints
    return endpoint_3d(p1, depth_mm, intr, patch), endpoint_3d(p2, depth_mm, intr, patch)


def umeyama(src, dst, with_scale: bool = True):
    """Least-squares similarity: find R, t, c so dst ≈ c·R@src + t (Umeyama 1991)."""
    src = np.asarray(src, float)
    dst = np.asarray(dst, float)
    n = len(src)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    Xs, Xd = src - mu_s, dst - mu_d
    Sigma = (Xd.T @ Xs) / n                 # 3x3 (dst outer src)
    U, D, Vt = np.linalg.svd(Sigma)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:   # reflection fix
        S[2, 2] = -1.0
    R = U @ S @ Vt
    c = (np.trace(np.diag(D) @ S) / (Xs ** 2).sum() * n) if with_scale else 1.0
    t = mu_d - c * R @ mu_s
    return R, t, c


def fit_transform(cam_pts, base_pts):
    """Fit T_base_cam (4x4) by Umeyama. Returns (T, err, scale) or (None,)*3.

    ``err`` is the per-point residual in mm. The scale is folded into the 3x3
    linear block of T, so applying T is a plain homogeneous multiply.
    """
    cam = np.array(cam_pts, float)
    base = np.array(base_pts, float)
    if len(cam) < 3:
        return None, None, None
    R, t, c = umeyama(cam, base, with_scale=True)
    M = c * R
    err = np.linalg.norm((M @ cam.T).T + t - base, axis=1)
    T = np.eye(4)
    T[:3, :3] = M
    T[:3, 3] = t
    return T, err, c


def cam_to_base(p_cam, T) -> np.ndarray:
    """Apply T_base_cam: camera-frame XYZ (mm) → base XYZ (mm)."""
    return (np.asarray(T) @ np.array([p_cam[0], p_cam[1], p_cam[2], 1.0]))[:3]


def quality_meta(err: np.ndarray, c: float, base_pts) -> dict:
    """The fit-quality dict saved as the .meta.json sidecar."""
    base = np.array(base_pts, float)
    return {
        "n_points": int(len(base)),
        "scale": float(c),
        "mean_resid_mm": float(err.mean()),
        "max_resid_mm": float(err.max()),
        "base_z_spread_mm": float(np.ptp(base[:, 2])),
    }


def quality_warnings(meta: dict) -> list[str]:
    """Human-readable problems with a fit (empty list = looks healthy)."""
    warns = []
    if meta["base_z_spread_mm"] < Z_SPREAD_WARN_MM:
        warns.append(
            f"base Z spread only {meta['base_z_spread_mm']:.0f} mm — points nearly "
            "coplanar, height fit is weak. Re-take at different heights."
        )
    if meta["mean_resid_mm"] > 5:
        warns.append(
            f"mean residual {meta['mean_resid_mm']:.1f} mm > 5 mm — add "
            "better-spread points or re-touch sloppy ones."
        )
    return warns


def save_transform(T: np.ndarray, meta: dict, path: Path = OUT_FILE) -> Path:
    """Save T_base_cam.npy + the quality sidecar load guards read. Returns path."""
    path = Path(path)
    np.save(path, T)
    path.with_suffix(META_SUFFIX).write_text(json.dumps(meta, indent=2))
    return path


def load_transform(path: Path = OUT_FILE) -> tuple[np.ndarray, dict | None] | None:
    """Load (T, meta) if the transform file exists; meta None without a sidecar."""
    path = Path(path)
    if not path.exists():
        return None
    T = np.load(path)
    meta_path = path.with_suffix(META_SUFFIX)
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else None
    return T, meta


@dataclass
class SolveResult:
    """One fit over the banked pairs, with the residual-gate verdict."""

    T: np.ndarray
    resid: np.ndarray
    scale: float
    meta: dict
    over_gate: list[int]      # indices of points whose residual exceeds the gate
    warnings: list[str]


@dataclass
class CalibrationSession:
    """Banked point pairs + the pending pair being assembled — no hardware.

    Mirrors the farino_app workflow: a click sets the pending CAMERA point, a
    pose-record sets the pending BASE point, ``bank()`` commits the pair. The
    interactive residual gate becomes ``solve()`` (reports ``over_gate``) plus
    ``drop_worst()`` / ``drop()`` for the UI to refit with.
    """

    cam_pts: list = field(default_factory=list)
    base_pts: list = field(default_factory=list)
    raw_px: list = field(default_factory=list)
    pending_cam: np.ndarray | None = None
    pending_px: tuple[int, int] | None = None
    pending_base: tuple[float, float, float] | None = None

    @property
    def count(self) -> int:
        return len(self.base_pts)

    @property
    def can_bank(self) -> bool:
        return self.pending_cam is not None and self.pending_base is not None

    @property
    def can_solve(self) -> bool:
        return self.count >= 3

    def set_camera_point(self, cam_xyz, px) -> None:
        """Camera-frame XYZ from a depth click (the 'click' step)."""
        self.pending_cam = np.asarray(cam_xyz, float)
        self.pending_px = (int(px[0]), int(px[1]))

    def set_robot_pose(self, base_xyz) -> None:
        """Base-frame TCP XYZ from touching the marker (the 'r' step)."""
        self.pending_base = (float(base_xyz[0]), float(base_xyz[1]),
                             float(base_xyz[2]))

    def bank(self) -> int:
        """Commit the pending pair; returns the new count. Raises if incomplete."""
        if not self.can_bank:
            raise ValueError("need BOTH a clicked camera XYZ and a recorded pose")
        self.cam_pts.append(self.pending_cam)
        self.base_pts.append(self.pending_base)
        self.raw_px.append(self.pending_px)
        self.pending_cam = self.pending_px = self.pending_base = None
        return self.count

    def undo(self) -> bool:
        """Remove the last banked pair. Returns False if there is none."""
        if not self.base_pts:
            return False
        self.cam_pts.pop()
        self.base_pts.pop()
        self.raw_px.pop()
        return True

    def solve(self, gate_mm: float = RESID_GATE_MM) -> SolveResult | None:
        """Fit the current pairs. None with fewer than 3 points."""
        T, err, c = fit_transform(self.cam_pts, self.base_pts)
        if T is None:
            return None
        meta = quality_meta(err, c, self.base_pts)
        over = [i for i, e in enumerate(err) if e > gate_mm]
        return SolveResult(T=T, resid=err, scale=c, meta=meta, over_gate=over,
                           warnings=quality_warnings(meta))

    def drop(self, idx: int) -> bool:
        """Drop pair ``idx`` for a refit; refuses to go below MIN_SOLVE points."""
        if self.count <= MIN_SOLVE or not (0 <= idx < self.count):
            return False
        self.cam_pts.pop(idx)
        self.base_pts.pop(idx)
        self.raw_px.pop(idx)
        return True

    def drop_worst(self) -> int | None:
        """Drop the highest-residual pair; returns its index, or None if refused."""
        result = self.solve()
        if result is None:
            return None
        idx = int(np.argmax(result.resid))
        return idx if self.drop(idx) else None
