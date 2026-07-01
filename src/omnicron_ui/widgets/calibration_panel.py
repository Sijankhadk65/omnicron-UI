"""Camera→base 3D calibration panel (T_base_cam).

GUI port of farino_app/calibrate_camera_to_base_3d.py. The panel never MOVES the
robot — the user jogs it from the teach pendant; here we only READ the TCP pose
and the camera depth, so the no-direct-movement-UI rule holds.

Workflow (collect 6+ pairs, spread in X, Y AND HEIGHT — coplanar points leave
the Z fit unconstrained):

1. "Calibrate" — restarts the camera with the aligned depth stream and arms
   click-to-pick on the video.
2. Put a marker where the camera sees it; jog the TCP tip to TOUCH it; press
   "Record robot pose" (reads base XYZ from the live state).
3. Retract the arm, CLICK the marker on the video — depth + intrinsics give the
   camera-frame XYZ.
4. "Bank pair". Move the marker (vary its height for some points!) and repeat.
5. "Solve" fits T_base_cam (Umeyama). Points over the residual gate are listed;
   "Drop worst" removes the worst pair and refits. "Save" writes T_base_cam.npy
   + the .meta.json quality sidecar.
"""

from __future__ import annotations

from PySide6 import QtWidgets

from omnicron_ui.camera import calibration
from omnicron_ui.camera.calibration import CalibrationSession, SolveResult
from omnicron_ui.robot.service import RobotService
from omnicron_ui.widgets.camera_view import CameraView


class CalibrationPanel(QtWidgets.QGroupBox):
    def __init__(self, camera_view: CameraView, robot: RobotService,
                 parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__("Camera → base calibration (3D)", parent)
        self._camera_view = camera_view
        self._robot = robot
        self._session = CalibrationSession()
        self._last_solve: SolveResult | None = None
        self._state: dict | None = None      # latest robot state (tcp/joints)
        self._active = False

        camera_view.service.picked.connect(self._on_picked)
        robot.state.connect(self._on_state)
        robot.connected.connect(self._on_connected)

        self.calib_btn = QtWidgets.QPushButton("Calibrate")
        self.calib_btn.setCheckable(True)
        self.calib_btn.toggled.connect(self._on_toggle)

        self.pose_btn = QtWidgets.QPushButton("Record robot pose")
        self.pose_btn.clicked.connect(self._on_record_pose)
        self.bank_btn = QtWidgets.QPushButton("Bank pair")
        self.bank_btn.clicked.connect(self._on_bank)
        self.undo_btn = QtWidgets.QPushButton("Undo")
        self.undo_btn.clicked.connect(self._on_undo)

        self.solve_btn = QtWidgets.QPushButton("Solve")
        self.solve_btn.clicked.connect(self._on_solve)
        self.drop_btn = QtWidgets.QPushButton("Drop worst")
        self.drop_btn.clicked.connect(self._on_drop_worst)
        self.save_btn = QtWidgets.QPushButton("Save")
        self.save_btn.clicked.connect(self._on_save)

        self.status_lbl = QtWidgets.QLabel("banked 0 — cam: –  pose: –")
        self.status_lbl.setStyleSheet("color:#888;")
        self.fit_lbl = QtWidgets.QLabel("no fit yet")
        self.fit_lbl.setStyleSheet("color:#888;")
        self.fit_lbl.setWordWrap(True)

        row1 = QtWidgets.QHBoxLayout()
        row1.addWidget(self.calib_btn)
        row1.addWidget(self.pose_btn)
        row1.addWidget(self.bank_btn)
        row1.addWidget(self.undo_btn)
        row2 = QtWidgets.QHBoxLayout()
        row2.addWidget(self.solve_btn)
        row2.addWidget(self.drop_btn)
        row2.addWidget(self.save_btn)
        row2.addStretch(1)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(row1)
        layout.addWidget(self.status_lbl)
        layout.addLayout(row2)
        layout.addWidget(self.fit_lbl)

        self._refresh()

    # --- log plumbing (forward through the camera view's shared log signal) --

    def _log(self, msg: str, level: str = "info") -> None:
        self._camera_view.log.emit(msg, level)

    # --- calibration mode -----------------------------------------------------

    def _on_toggle(self, on: bool) -> None:
        self._active = on
        view = self._camera_view
        view.set_pick_mode(on)
        if on:
            # Depth streams by default; only restart if this stream lacks it
            # (the pipeline config is fixed while running).
            view.service.set_depth_enabled(True)
            if not view.running:
                view.start()
            elif not view.service.depth_active:
                self._log("Restarting camera with depth…", "info")
                view.stop()
                view.start()
            self._log("Calibration ON — touch the marker with the TCP, Record "
                      "pose, then click the marker on the video.", "info")
        else:
            self._log("Calibration OFF.", "info")
        self._refresh()

    # --- collecting pairs -------------------------------------------------------

    def _on_state(self, state: dict | None) -> None:
        self._state = state

    def _on_connected(self, tool: int, user: int) -> None:
        if user != 0:
            self._log(f"WARNING: active workpiece frame is {user}, not 0 — TCP "
                      "poses won't be base-frame. Set WObj 0 before calibrating.",
                      "warn")

    def _on_record_pose(self) -> None:
        if self._state is None or not self._state.get("tcp"):
            self._log("Record pose ignored — robot not connected", "warn")
            return
        tcp = self._state["tcp"]
        self._session.set_robot_pose(tcp[:3])
        self._log(f"Pose set: base ({tcp[0]:.1f}, {tcp[1]:.1f}, {tcp[2]:.1f}) mm",
                  "success")
        self._refresh()

    def _on_picked(self, result) -> None:
        # Failures (holes / no depth) are already logged by the camera service.
        if result is None:
            return
        px, cam_xyz = result
        self._session.set_camera_point(cam_xyz, px)
        self._push_markers()
        self._refresh()

    def _on_bank(self) -> None:
        if not self._session.can_bank:
            self._log("Bank ignored — need BOTH a clicked camera point and a "
                      "recorded pose", "warn")
            return
        n = self._session.bank()
        self._last_solve = None
        self._log(f"Banked pair {n} (need {calibration.MIN_POINTS}+, spread in "
                  "X, Y and height)", "success")
        self._push_markers()
        self._refresh()

    def _on_undo(self) -> None:
        if self._session.undo():
            self._last_solve = None
            self._log(f"Undone — {self._session.count} pair(s) left", "info")
        else:
            self._log("Nothing to undo", "warn")
        self._push_markers()
        self._refresh()

    # --- solve / gate / save ------------------------------------------------------

    def _on_solve(self) -> None:
        result = self._session.solve()
        if result is None:
            self._log(f"Solve needs ≥3 pairs (have {self._session.count})", "warn")
            return
        self._last_solve = result
        resid = ", ".join(f"#{i}={e:.1f}" for i, e in enumerate(result.resid))
        self._log(f"Fit over {self._session.count} pairs: mean "
                  f"{result.meta['mean_resid_mm']:.2f} mm, max "
                  f"{result.meta['max_resid_mm']:.2f} mm, scale "
                  f"{result.scale:.4f} | residuals {resid}", "info")
        for w in result.warnings:
            self._log(f"WARNING: {w}", "warn")
        if result.over_gate:
            self._log(f"{len(result.over_gate)} point(s) over the "
                      f"{calibration.RESID_GATE_MM:.0f} mm gate: "
                      + ", ".join(f"#{i}" for i in result.over_gate)
                      + " — Drop worst and re-solve, or Save anyway", "warn")
        else:
            self._log("All points within the residual gate — ready to Save",
                      "success")
        self._refresh()

    def _on_drop_worst(self) -> None:
        if self._session.count <= calibration.MIN_SOLVE:
            self._log(f"Refusing to drop below {calibration.MIN_SOLVE} points — "
                      "bank more pairs instead", "warn")
            return
        idx = self._session.drop_worst()
        if idx is None:
            self._log("Drop worst ignored — no fit yet", "warn")
            return
        self._log(f"Dropped worst pair #{idx}; re-solving…", "info")
        self._push_markers()
        self._on_solve()

    def _on_save(self) -> None:
        if self._last_solve is None:
            self._log("Save ignored — Solve first", "warn")
            return
        path = calibration.save_transform(self._last_solve.T,
                                          self._last_solve.meta)
        self._log(f"Saved {path.resolve()} (+ quality sidecar) — camera XYZ → "
                  "base XYZ", "success")
        # Hot-reload: detections start reporting robot base coords immediately.
        self._camera_view.service.reload_transform()
        self._refresh()

    # --- view state -------------------------------------------------------------

    def _push_markers(self) -> None:
        self._camera_view.service.set_markers(self._session.pending_px,
                                              list(self._session.raw_px))

    def _refresh(self) -> None:
        s = self._session
        cam = ("set" if s.pending_cam is not None else "–")
        pose = ("set" if s.pending_base is not None else "–")
        self.status_lbl.setText(
            f"banked {s.count}/{calibration.MIN_POINTS} — cam: {cam}  pose: {pose}"
        )
        if self._last_solve is not None:
            m = self._last_solve.meta
            gate = (f", {len(self._last_solve.over_gate)} over gate"
                    if self._last_solve.over_gate else "")
            self.fit_lbl.setText(
                f"fit: mean {m['mean_resid_mm']:.2f} mm / max "
                f"{m['max_resid_mm']:.2f} mm, scale {m['scale']:.4f}, Z spread "
                f"{m['base_z_spread_mm']:.0f} mm{gate}"
            )
        else:
            self.fit_lbl.setText("no fit yet")
        for b in (self.pose_btn, self.bank_btn, self.undo_btn):
            b.setEnabled(self._active)
        self.solve_btn.setEnabled(self._active and s.can_solve)
        self.drop_btn.setEnabled(self._active and s.count > calibration.MIN_SOLVE)
        self.save_btn.setEnabled(self._active and self._last_solve is not None)
