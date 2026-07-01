"""Unit tests for the camera→base 3D calibration math — no camera, no robot."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from omnicron_ui.camera.calibration import (
    MIN_SOLVE,
    RESID_GATE_MM,
    CalibrationSession,
    cam_to_base,
    endpoint_3d,
    fit_transform,
    line_to_camera,
    load_transform,
    pixel_to_camera,
    quality_meta,
    quality_warnings,
    sample_depth_mm,
    save_transform,
    umeyama,
)
from omnicron_ui.camera.red_line import LineDetection

INTR = SimpleNamespace(fx=460.0, fy=460.0, cx=320.0, cy=240.0)

# A non-trivial ground-truth similarity transform for synthetic correspondences.
_ANG = np.radians(30.0)
R_TRUE = np.array([
    [np.cos(_ANG), -np.sin(_ANG), 0.0],
    [np.sin(_ANG), np.cos(_ANG), 0.0],
    [0.0, 0.0, 1.0],
])
T_TRUE = np.array([120.0, -340.0, 55.0])
SCALE_TRUE = 1.02

# Camera points spread in X, Y AND Z (non-coplanar, as the workflow demands).
CAM_PTS = np.array([
    [0.0, 0.0, 500.0],
    [100.0, 0.0, 520.0],
    [0.0, 100.0, 480.0],
    [100.0, 100.0, 560.0],
    [-80.0, 40.0, 610.0],
    [50.0, -60.0, 450.0],
])
BASE_PTS = (SCALE_TRUE * (R_TRUE @ CAM_PTS.T)).T + T_TRUE


# --- depth sampling / deprojection ------------------------------------------------


def test_sample_depth_median_ignores_holes():
    depth = np.zeros((20, 20), np.float32)
    depth[9:12, 9:12] = [[500, 0, 520], [510, 0, 0], [530, 505, 0]]
    # Median of the valid (non-zero) values in the 5x5 patch around (10, 10).
    assert sample_depth_mm(depth, (10, 10)) == 510.0


def test_sample_depth_all_holes_returns_zero():
    depth = np.zeros((20, 20), np.float32)
    assert sample_depth_mm(depth, (10, 10)) == 0.0


def test_sample_depth_clamps_at_image_edge():
    depth = np.full((20, 20), 400.0, np.float32)
    assert sample_depth_mm(depth, (0, 0)) == 400.0  # patch clipped, no IndexError


def test_pixel_to_camera_principal_point_is_on_axis():
    p = pixel_to_camera((INTR.cx, INTR.cy), 500.0, INTR)
    assert np.allclose(p, [0.0, 0.0, 500.0])


def test_pixel_to_camera_scales_with_depth_and_focal():
    p = pixel_to_camera((INTR.cx + 46.0, INTR.cy - 92.0), 500.0, INTR)
    # x = du * z / fx = 46 * 500 / 460 = 50; y = -92 * 500 / 460 = -100.
    assert np.allclose(p, [50.0, -100.0, 500.0])


# --- red-line endpoints → camera frame ---------------------------------------------


def test_endpoint_3d_deprojects_at_sampled_depth():
    depth = np.full((480, 640), 500.0, np.float32)
    p = endpoint_3d((INTR.cx + 46.0, INTR.cy), depth, INTR)
    assert np.allclose(p, [50.0, 0.0, 500.0])


def test_endpoint_3d_hole_returns_none():
    depth = np.zeros((480, 640), np.float32)
    assert endpoint_3d((320, 240), depth, INTR) is None


def test_line_to_camera_maps_both_endpoints():
    depth = np.full((480, 640), 500.0, np.float32)
    endpoints = (np.array([INTR.cx, INTR.cy]), np.array([INTR.cx + 92.0, INTR.cy]))
    cam1, cam2 = line_to_camera(endpoints, depth, INTR)
    assert np.allclose(cam1, [0.0, 0.0, 500.0])
    assert np.allclose(cam2, [100.0, 0.0, 500.0])
    # The camera-frame span matches the physical span implied by depth/focal.
    assert np.linalg.norm(cam2 - cam1) == pytest.approx(100.0)


def test_line_to_camera_flags_only_the_holey_endpoint():
    depth = np.full((480, 640), 500.0, np.float32)
    depth[230:250, 400:420] = 0.0                     # hole under P2 only
    endpoints = (np.array([320.0, 240.0]), np.array([410.0, 240.0]))
    cam1, cam2 = line_to_camera(endpoints, depth, INTR)
    assert cam1 is not None
    assert cam2 is None


def test_line_detection_carries_camera_xyz():
    det = LineDetection(p1=np.array([1.0, 2.0]), p2=np.array([3.0, 4.0]),
                        cam1=np.array([0.0, 0.0, 500.0]),
                        cam2=np.array([10.0, 0.0, 500.0]))
    assert det.has_camera_xyz
    assert det.endpoints == (det.p1, det.p2)


def test_line_detection_without_depth_has_no_camera_xyz():
    det = LineDetection(p1=np.array([1.0, 2.0]), p2=np.array([3.0, 4.0]))
    assert not det.has_camera_xyz
    # one-sided depth (hole at P2) also doesn't count as usable
    det = LineDetection(p1=np.array([1.0, 2.0]), p2=np.array([3.0, 4.0]),
                        cam1=np.array([0.0, 0.0, 500.0]))
    assert not det.has_camera_xyz


# --- umeyama / fit_transform ---------------------------------------------------------


def test_umeyama_recovers_similarity_transform():
    R, t, c = umeyama(CAM_PTS, BASE_PTS)
    assert np.allclose(R, R_TRUE, atol=1e-9)
    assert np.allclose(t, T_TRUE, atol=1e-6)
    assert np.isclose(c, SCALE_TRUE, atol=1e-9)


def test_fit_transform_zero_residuals_on_exact_data():
    T, err, c = fit_transform(CAM_PTS, BASE_PTS)
    assert np.allclose(T[:3, :3], SCALE_TRUE * R_TRUE, atol=1e-9)  # scale folded in
    assert np.allclose(err, 0.0, atol=1e-6)
    assert np.isclose(c, SCALE_TRUE)
    assert np.allclose(T[3], [0, 0, 0, 1])


def test_fit_transform_needs_three_points():
    assert fit_transform(CAM_PTS[:2], BASE_PTS[:2]) == (None, None, None)


def test_cam_to_base_roundtrip():
    T, _, _ = fit_transform(CAM_PTS, BASE_PTS)
    assert np.allclose(cam_to_base(CAM_PTS[3], T), BASE_PTS[3], atol=1e-6)


def test_fit_survives_a_noisy_point_with_residual_showing_it():
    base = BASE_PTS.copy()
    base[2] += [6.0, 0.0, 0.0]              # one sloppy 6 mm touch
    _, err, _ = fit_transform(CAM_PTS, base)
    assert int(np.argmax(err)) == 2         # the bad pair carries the residual


# --- quality meta / warnings -----------------------------------------------------------


def test_quality_meta_reports_z_spread_and_residuals():
    err = np.array([1.0, 2.0, 3.0])
    meta = quality_meta(err, 1.01, BASE_PTS[:3])
    assert meta["n_points"] == 3
    assert meta["mean_resid_mm"] == pytest.approx(2.0)
    assert meta["max_resid_mm"] == pytest.approx(3.0)
    assert meta["base_z_spread_mm"] == pytest.approx(np.ptp(BASE_PTS[:3, 2]))


def test_coplanar_capture_is_flagged():
    flat = BASE_PTS.copy()
    flat[:, 2] = 80.0                        # everything on one table height
    meta = quality_meta(np.array([1.0]), 1.0, flat)
    warns = quality_warnings(meta)
    assert any("coplanar" in w for w in warns)


def test_healthy_fit_has_no_warnings():
    meta = quality_meta(np.array([1.0, 1.5]), 1.0, BASE_PTS)
    assert quality_warnings(meta) == []


# --- save / load -------------------------------------------------------------------


def test_save_and_load_transform_with_sidecar(tmp_path):
    T, err, c = fit_transform(CAM_PTS, BASE_PTS)
    meta = quality_meta(err, c, BASE_PTS)
    path = save_transform(T, meta, tmp_path / "T_base_cam.npy")

    loaded_T, loaded_meta = load_transform(path)
    assert np.allclose(loaded_T, T)
    assert loaded_meta["n_points"] == len(BASE_PTS)
    assert (tmp_path / "T_base_cam.meta.json").exists()


def test_load_transform_missing_file_returns_none(tmp_path):
    assert load_transform(tmp_path / "nope.npy") is None


# --- CalibrationSession ---------------------------------------------------------------


def _banked_session(n: int = len(CAM_PTS)) -> CalibrationSession:
    s = CalibrationSession()
    for i in range(n):
        s.set_camera_point(CAM_PTS[i], (10 * i, 20 * i))
        s.set_robot_pose(BASE_PTS[i])
        s.bank()
    return s


def test_bank_requires_both_halves():
    s = CalibrationSession()
    s.set_camera_point(CAM_PTS[0], (5, 5))
    assert not s.can_bank
    with pytest.raises(ValueError, match="BOTH"):
        s.bank()
    s.set_robot_pose(BASE_PTS[0])
    assert s.bank() == 1
    # Banking consumes the pending pair.
    assert s.pending_cam is None and s.pending_base is None and s.pending_px is None


def test_undo_removes_last_pair():
    s = _banked_session(3)
    assert s.undo() is True
    assert s.count == 2
    s.undo(), s.undo()
    assert s.undo() is False


def test_solve_needs_three_pairs():
    assert _banked_session(2).solve() is None
    assert _banked_session(3).solve() is not None


def test_solve_exact_data_passes_the_gate():
    result = _banked_session().solve()
    assert result.over_gate == []
    assert result.warnings == []
    assert np.allclose(result.resid, 0.0, atol=1e-6)


def test_solve_flags_points_over_the_gate():
    s = _banked_session()
    s.base_pts[2] = tuple(np.array(s.base_pts[2]) + [3 * RESID_GATE_MM, 0, 0])
    result = s.solve()
    assert 2 in result.over_gate


def test_drop_worst_removes_the_flagged_pair_and_refits_clean():
    s = _banked_session()
    s.base_pts[2] = tuple(np.array(s.base_pts[2]) + [3 * RESID_GATE_MM, 0, 0])
    assert s.drop_worst() == 2
    assert s.count == len(CAM_PTS) - 1
    assert s.solve().over_gate == []


def test_drop_refuses_below_min_solve():
    s = _banked_session(MIN_SOLVE)
    assert s.drop(0) is False
    assert s.drop_worst() is None
    assert s.count == MIN_SOLVE
