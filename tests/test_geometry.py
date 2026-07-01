"""Unit tests for the pure orientation/geometry helpers."""

from __future__ import annotations

import math

import pytest

from omnicron_ui.robot.geometry import (
    TORCH_AXIS,
    orientation_candidates,
    rpy_to_matrix,
    torch_dir_in_base,
    yaw_to_line,
)


def _close(a, b, tol=1e-9):
    return all(math.isclose(x, y, abs_tol=tol) for x, y in zip(a, b, strict=True))


# --- rpy_to_matrix -------------------------------------------------------------


def test_identity_rpy_is_identity_matrix():
    m = rpy_to_matrix(0.0, 0.0, 0.0)
    assert _close(m[0], (1, 0, 0))
    assert _close(m[1], (0, 1, 0))
    assert _close(m[2], (0, 0, 1))


def test_rz_rotates_x_toward_y():
    # Pure yaw of +90°: base X → base Y.
    m = rpy_to_matrix(0.0, 0.0, 90.0)
    x_rotated = (m[0][0], m[1][0], m[2][0])
    assert _close(x_rotated, (0, 1, 0))


def test_composition_order_is_rz_ry_rx():
    # rx=90 then rz=90 (R = Rz@Ry@Rx): tool Z (0,0,1) -Rx→ (0,-1,0) -Rz→ (1,0,0).
    m = rpy_to_matrix(90.0, 0.0, 90.0)
    z_rotated = (m[0][2], m[1][2], m[2][2])
    assert _close(z_rotated, (1, 0, 0))


# --- torch_dir_in_base ----------------------------------------------------------


def test_torch_points_down_at_zero_rpy():
    # TORCH_AXIS is tool -Z; with identity orientation it is base -Z (down).
    assert TORCH_AXIS == (0.0, 0.0, -1.0)
    assert _close(torch_dir_in_base(0.0, 0.0, 0.0), (0, 0, -1))


def test_torch_points_up_when_flipped():
    # rx=180 flips the tool: torch axis ends up along base +Z.
    assert _close(torch_dir_in_base(180.0, 0.0, 0.0), (0, 0, 1))


def test_yaw_does_not_change_downwardness():
    for rz in (0.0, 45.0, 90.0, -120.0):
        assert math.isclose(torch_dir_in_base(0.0, 0.0, rz)[2], -1.0, abs_tol=1e-9)


# --- orientation_candidates ------------------------------------------------------


def test_candidates_start_tool_down_then_sweep():
    cands = list(orientation_candidates())
    assert cands[:4] == [(180.0, 0.0, 0.0), (180.0, 0.0, 90.0),
                         (180.0, 0.0, 180.0), (180.0, 0.0, -90.0)]
    # Coarse sweep: 5 rx * 5 ry * 5 rz after the 4 tool-down entries.
    assert len(cands) == 4 + 125
    assert all(len(c) == 3 for c in cands)


# --- yaw_to_line ------------------------------------------------------------------


P1 = (0.0, 0.0, 80.0)
P_PLUS_Y = (0.0, 100.0, 80.0)


def test_push_faces_torch_along_travel():
    # Zero RPY: forward axis heads +X. Line heads +Y (90°) → rz becomes 90.
    rz = yaw_to_line(0.0, 0.0, 0.0, P1, P_PLUS_Y, pull=False)
    assert math.isclose(rz, 90.0, abs_tol=1e-9)


def test_pull_is_default_and_adds_180():
    # Pull/drag (the project's welding technique): torch leans AGAINST travel.
    rz_default = yaw_to_line(0.0, 0.0, 0.0, P1, P_PLUS_Y)
    rz_pull = yaw_to_line(0.0, 0.0, 0.0, P1, P_PLUS_Y, pull=True)
    assert rz_default == rz_pull
    assert math.isclose(rz_pull, -90.0, abs_tol=1e-9)  # 90 + 180 wrapped


def test_yaw_offset_nudges_alignment():
    rz = yaw_to_line(0.0, 0.0, 0.0, P1, P_PLUS_Y, pull=False, yaw_offset=10.0)
    assert math.isclose(rz, 100.0, abs_tol=1e-9)


def test_result_is_wrapped_to_half_open_range():
    rz = yaw_to_line(0.0, 0.0, 170.0, P1, P_PLUS_Y, pull=True)
    assert -180.0 <= rz < 180.0


def test_keeps_rz_when_line_has_no_xy_direction():
    # P1 == P2 in XY (vertical or zero-length line): no heading to align to.
    assert yaw_to_line(0.0, 0.0, 33.0, P1, (0.0, 0.0, 120.0)) == 33.0


def test_keeps_rz_when_forward_axis_is_vertical():
    # ry=90 points the tool forward axis straight down: heading undefined.
    assert yaw_to_line(0.0, 90.0, 33.0, P1, P_PLUS_Y) == 33.0


@pytest.mark.parametrize("rz0", [-170.0, -45.0, 0.0, 45.0, 170.0])
def test_alignment_is_independent_of_starting_rz(rz0):
    """Whatever rz the tool starts at, the aligned heading is the same."""
    rz = yaw_to_line(0.0, 0.0, rz0, P1, P_PLUS_Y, pull=False)
    assert math.isclose(rz, 90.0, abs_tol=1e-9)
