"""Connection panel: connect/disconnect to the robot and show live state.

This is the first consumer of ``RobotService`` and exists mainly to prove the
async wiring: clicking Connect runs the (blocking, ~2.5 s) SDK connect on the
worker thread, so the button/spinner and the rest of the UI stay responsive the
whole time. Live TCP/joint/fault readouts arrive via the service's ``state``
signal.
"""

from __future__ import annotations

from PySide6 import QtWidgets
from PySide6.QtCore import Qt

from omnicron_ui.robot.service import RobotService

DEFAULT_IP = "192.168.58.2"


def _fmt(vals, fmt="{:.1f}") -> str:
    if not vals:
        return "—"
    return "  ".join(fmt.format(v) for v in vals)


class ConnectionPanel(QtWidgets.QGroupBox):
    def __init__(self, service: RobotService, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__("Robot", parent)
        self._service = service
        self._connected = False

        self.ip_edit = QtWidgets.QLineEdit(DEFAULT_IP)
        self.ip_edit.setMaximumWidth(160)

        self.connect_btn = QtWidgets.QPushButton("Connect")
        self.connect_btn.clicked.connect(self._on_connect_clicked)

        self.status_lbl = QtWidgets.QLabel("Disconnected")
        self.status_lbl.setStyleSheet("color:#ffb300;")

        self.tcp_lbl = QtWidgets.QLabel("—")
        self.joints_lbl = QtWidgets.QLabel("—")
        self.fault_lbl = QtWidgets.QLabel("—")
        for lbl in (self.tcp_lbl, self.joints_lbl, self.fault_lbl):
            lbl.setStyleSheet("font-family:monospace;")
            lbl.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        top = QtWidgets.QHBoxLayout()
        top.addWidget(QtWidgets.QLabel("IP"))
        top.addWidget(self.ip_edit)
        top.addWidget(self.connect_btn)
        top.addWidget(self.status_lbl, 1)

        form = QtWidgets.QFormLayout()
        form.addRow("TCP  [x y z rx ry rz]", self.tcp_lbl)
        form.addRow("Joints [j1..j6]", self.joints_lbl)
        form.addRow("Fault [main sub]", self.fault_lbl)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(top)
        layout.addLayout(form)

        # React to the service (all delivered on the UI thread).
        service.connected.connect(self._on_connected)
        service.disconnected.connect(self._on_disconnected)
        service.state.connect(self._on_state)
        service.busy.connect(self._on_busy)

    # --- button ---
    def _on_connect_clicked(self) -> None:
        if self._connected:
            self._service.disconnect_robot()
        else:
            ip = self.ip_edit.text().strip() or DEFAULT_IP
            self.status_lbl.setText("Connecting…")
            self.status_lbl.setStyleSheet("color:#42a5f5;")
            self._service.connect_robot(ip)

    # --- service signals ---
    def _on_connected(self, tool: int, user: int) -> None:
        self._connected = True
        self.connect_btn.setText("Disconnect")
        self.ip_edit.setEnabled(False)
        self.status_lbl.setText(f"Connected (tool={tool}, wobj={user})")
        self.status_lbl.setStyleSheet("color:#4caf50;")

    def _on_disconnected(self) -> None:
        self._connected = False
        self.connect_btn.setText("Connect")
        self.ip_edit.setEnabled(True)
        self.status_lbl.setText("Disconnected")
        self.status_lbl.setStyleSheet("color:#ffb300;")
        self.tcp_lbl.setText("—")
        self.joints_lbl.setText("—")
        self.fault_lbl.setText("—")

    def _on_state(self, state: dict | None) -> None:
        if not state:
            return
        self.tcp_lbl.setText(_fmt(state.get("tcp")))
        self.joints_lbl.setText(_fmt(state.get("joints")))
        fault = state.get("fault") or []
        self.fault_lbl.setText(_fmt(fault, "{:d}"))
        # Highlight a nonzero main fault code.
        faulted = bool(fault) and fault[0] != 0
        self.fault_lbl.setStyleSheet(
            "font-family:monospace; color:#ef5350;" if faulted else "font-family:monospace;"
        )

    def _on_busy(self, busy: bool) -> None:
        # Disable the button while a job runs so the user can't stack commands.
        self.connect_btn.setEnabled(not busy)
