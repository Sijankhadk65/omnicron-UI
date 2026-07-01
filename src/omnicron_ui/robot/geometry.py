"""Pure orientation/geometry helpers for torch motion — no SDK, no Qt.

Ported from the field-tested farino_app logic (test_linear_weld_movements.py /
red_line_viewer.py). Everything here is plain Python on small vectors so it can
be unit-tested headless.

Conventions (Fairino):
* RPY is [rx, ry, rz] in DEGREES, composed as R = Rz @ Ry @ Rx.
* TORCH_AXIS is the TOOL-frame axis the torch/wire points along.
* TORCH_FWD_AXIS is the TOOL-frame axis treated as the torch "forward" heading
  (tune with yaw_offset if the physical torch differs).

Welding technique: passes are welded PULL (drag / backhand — torch leans against
the travel direction), not push. That is the project's end goal, so pull is the
DEFAULT in yaw_to_line; push exists only for experiments.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence

Vec3 = tuple[float, float, float]
Mat3 = tuple[Vec3, Vec3, Vec3]

# Tool-frame axis the torch/wire points along (flip to +Z if the torch points up).
TORCH_AXIS: Vec3 = (0.0, 0.0, -1.0)

# Tool-frame axis taken as the torch 'forward' heading for yaw alignment.
TORCH_FWD_AXIS: Vec3 = (1.0, 0.0, 0.0)


def rpy_to_matrix(rx: float, ry: float, rz: float) -> Mat3:
    """Fairino RPY degrees -> 3x3 rotation matrix (R = Rz @ Ry @ Rx)."""
    a, b, g = math.radians(rx), math.radians(ry), math.radians(rz)
    ca, sa = math.cos(a), math.sin(a)
    cb, sb = math.cos(b), math.sin(b)
    cg, sg = math.cos(g), math.sin(g)
    # Rows of Rz @ Ry @ Rx, expanded.
    return (
        (cg * cb, cg * sb * sa - sg * ca, cg * sb * ca + sg * sa),
        (sg * cb, sg * sb * sa + cg * ca, sg * sb * ca - cg * sa),
        (-sb, cb * sa, cb * ca),
    )


def mat_vec(m: Mat3, v: Sequence[float]) -> Vec3:
    """3x3 matrix @ 3-vector."""
    return (
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    )


def torch_dir_in_base(rx: float, ry: float, rz: float) -> Vec3:
    """Unit vector the torch points along, in the BASE frame, for the given RPY."""
    return mat_vec(rpy_to_matrix(rx, ry, rz), TORCH_AXIS)


def orientation_candidates() -> Iterator[tuple[float, float, float]]:
    """Yield (rx, ry, rz) orientations to try: tool-down first, then a coarse sweep."""
    for rz in (0.0, 90.0, 180.0, -90.0):
        yield (180.0, 0.0, rz)
    for rx in range(-180, 181, 90):
        for ry in range(-90, 91, 45):
            for rz in range(-180, 181, 90):
                yield (float(rx), float(ry), float(rz))


def yaw_to_line(
    rx: float,
    ry: float,
    rz: float,
    p1: Sequence[float],
    p2: Sequence[float],
    pull: bool = True,
    yaw_offset: float = 0.0,
) -> float:
    """rz (deg) rotated about BASE Z so the torch faces along the p1->p2 line.

    Keeps the tool-down tilt (rx, ry) — rotating an RPY orientation about base Z
    is just rz += delta, since R = Rz@Ry@Rx and Rz(d)@Rz(rz) = Rz(rz+d). delta
    aligns the current heading of TORCH_FWD_AXIS with the line heading in XY.

    ``pull=True`` (the DEFAULT — the project welds pull/drag, not push) adds
    180°, so the torch leans AGAINST the travel direction (backhand). pull=False
    is push/forehand, for experiments only. ``yaw_offset`` nudges the alignment
    (e.g. 90 if the physical torch forward is tool +Y).

    Returns rz unchanged when there is no defined heading: p1 == p2 in XY, or
    the forward axis is near-vertical (the pose is not tool-down).
    """
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    if math.hypot(dx, dy) < 1e-6:
        return rz
    fwd = mat_vec(rpy_to_matrix(rx, ry, rz), TORCH_FWD_AXIS)
    if math.hypot(fwd[0], fwd[1]) < 1e-3:
        return rz
    delta = math.degrees(math.atan2(dy, dx) - math.atan2(fwd[1], fwd[0]))
    rz_new = rz + delta + yaw_offset + (180.0 if pull else 0.0)
    return (rz_new + 180.0) % 360.0 - 180.0  # wrap to [-180, 180)
