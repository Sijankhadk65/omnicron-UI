"""PTP (point-to-point / MoveJ) motion panel.

Two ways to command a PTP move, on tabs:

* **Joints** — type j1..j6 and MoveJ straight to that configuration.
* **XYZ (IK)** — type a base-frame TCP position (and optionally orientation); the
  controller solves IK and MoveJ's to the resulting joints. Coordinates are in the
  BASE frame and the TCP moves there.

The panel never blocks: buttons submit jobs to ``RobotService`` and it reacts to
connected/disconnected/busy/state signals. It's disabled until the robot is
connected, and the move buttons are disabled while a job is running.
"""

from __future__ import annotations

from PySide6 import QtWidgets

from omnicron_ui.robot.service import RobotService


def _spin(minimum, maximum, suffix, decimals=1, step=1.0) -> QtWidgets.QDoubleSpinBox:
    box = QtWidgets.QDoubleSpinBox()
    box.setRange(minimum, maximum)
    box.setDecimals(decimals)
    box.setSingleStep(step)
    box.setSuffix(suffix)
    return box


class PtpPanel(QtWidgets.QGroupBox):
    def __init__(self, service: RobotService, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__("PTP Motion", parent)
        self._service = service
        self._state: dict | None = None

        # Shared velocity (percentage of max speed).
        self.vel = _spin(1.0, 100.0, " %", decimals=0, step=5.0)
        self.vel.setValue(20.0)
        vel_row = QtWidgets.QHBoxLayout()
        vel_row.addWidget(QtWidgets.QLabel("Velocity"))
        vel_row.addWidget(self.vel)
        vel_row.addStretch(1)

        tabs = QtWidgets.QTabWidget()
        tabs.addTab(self._build_joints_tab(), "Joints")
        tabs.addTab(self._build_xyz_tab(), "XYZ (IK)")

        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(vel_row)
        layout.addWidget(tabs)

        # React to the service.
        service.connected.connect(lambda *_: self.setEnabled(True))
        service.disconnected.connect(lambda: self.setEnabled(False))
        service.busy.connect(self._on_busy)
        service.state.connect(self._on_state)
        self.setEnabled(False)   # until connected

    # --- Joints tab ---
    def _build_joints_tab(self) -> QtWidgets.QWidget:
        self.joints = [_spin(-360.0, 360.0, " °") for _ in range(6)]
        grid = QtWidgets.QGridLayout()
        for i, box in enumerate(self.joints):
            grid.addWidget(QtWidgets.QLabel(f"j{i + 1}"), i // 3, (i % 3) * 2)
            grid.addWidget(box, i // 3, (i % 3) * 2 + 1)

        use_cur = QtWidgets.QPushButton("Use current")
        use_cur.clicked.connect(self._fill_current_joints)
        self.move_joints_btn = QtWidgets.QPushButton("Move (PTP joints)")
        self.move_joints_btn.clicked.connect(self._move_joints)

        btns = QtWidgets.QHBoxLayout()
        btns.addWidget(use_cur)
        btns.addStretch(1)
        btns.addWidget(self.move_joints_btn)

        tab = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(tab)
        v.addLayout(grid)
        v.addLayout(btns)
        return tab

    # --- XYZ tab ---
    def _build_xyz_tab(self) -> QtWidgets.QWidget:
        self.x = _spin(-3000.0, 3000.0, " mm")
        self.y = _spin(-3000.0, 3000.0, " mm")
        self.z = _spin(-3000.0, 3000.0, " mm")
        self.rx = _spin(-360.0, 360.0, " °")
        self.ry = _spin(-360.0, 360.0, " °")
        self.rz = _spin(-360.0, 360.0, " °")

        grid = QtWidgets.QGridLayout()
        for col, (label, box) in enumerate(
            (("X", self.x), ("Y", self.y), ("Z", self.z))
        ):
            grid.addWidget(QtWidgets.QLabel(label), 0, col * 2)
            grid.addWidget(box, 0, col * 2 + 1)
        for col, (label, box) in enumerate(
            (("Rx", self.rx), ("Ry", self.ry), ("Rz", self.rz))
        ):
            grid.addWidget(QtWidgets.QLabel(label), 1, col * 2)
            grid.addWidget(box, 1, col * 2 + 1)

        self.keep_orient = QtWidgets.QCheckBox("Keep current orientation")
        self.keep_orient.setChecked(True)
        self.keep_orient.toggled.connect(self._on_keep_orient)
        for box in (self.rx, self.ry, self.rz):
            box.setEnabled(False)

        use_cur = QtWidgets.QPushButton("Use current")
        use_cur.clicked.connect(self._fill_current_xyz)
        self.move_xyz_btn = QtWidgets.QPushButton("Move (PTP → IK)")
        self.move_xyz_btn.clicked.connect(self._move_xyz)

        btns = QtWidgets.QHBoxLayout()
        btns.addWidget(use_cur)
        btns.addStretch(1)
        btns.addWidget(self.move_xyz_btn)

        tab = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(tab)
        v.addLayout(grid)
        v.addWidget(self.keep_orient)
        v.addLayout(btns)
        return tab

    # --- actions ---
    def _move_joints(self) -> None:
        joints = [b.value() for b in self.joints]
        self._service.move_ptp_joints(joints, self.vel.value())

    def _move_xyz(self) -> None:
        if self.keep_orient.isChecked():
            rx = ry = rz = None
        else:
            rx, ry, rz = self.rx.value(), self.ry.value(), self.rz.value()
        self._service.move_ptp_pose(
            self.x.value(), self.y.value(), self.z.value(), rx, ry, rz, self.vel.value()
        )

    def _on_keep_orient(self, keep: bool) -> None:
        for box in (self.rx, self.ry, self.rz):
            box.setEnabled(not keep)

    def _fill_current_joints(self) -> None:
        if self._state and self._state.get("joints"):
            for box, val in zip(self.joints, self._state["joints"], strict=False):
                box.setValue(float(val))

    def _fill_current_xyz(self) -> None:
        if not self._state or not self._state.get("tcp"):
            return
        tcp = self._state["tcp"]
        for box, val in zip((self.x, self.y, self.z, self.rx, self.ry, self.rz),
                            tcp, strict=False):
            box.setValue(float(val))

    # --- service signals ---
    def _on_state(self, state: dict | None) -> None:
        self._state = state

    def _on_busy(self, busy: bool) -> None:
        self.move_joints_btn.setEnabled(not busy)
        self.move_xyz_btn.setEnabled(not busy)
