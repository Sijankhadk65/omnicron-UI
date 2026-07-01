"""The main application window.

Owns the single ``RobotService`` (the async worker thread) and hosts the feature
panels. The window never makes a blocking robot call directly — it only submits
jobs to the service and reacts to the service's signals, so the UI stays
responsive while the robot is talking.
"""

from __future__ import annotations

from PySide6 import QtGui, QtWidgets

from omnicron_ui.robot.service import RobotService
from omnicron_ui.widgets.calibration_panel import CalibrationPanel
from omnicron_ui.widgets.camera_view import CameraView
from omnicron_ui.widgets.connection_panel import ConnectionPanel
from omnicron_ui.widgets.log_panel import LogPanel

APP_TITLE = "Omnicron — Fairino Welding Robot"


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(1024, 720)

        # The async robot service — all blocking SDK calls run on its worker
        # thread; the UI only submits jobs and listens to its signals.
        self.service = RobotService()
        self.service.log.connect(self._on_log)
        self.service.busy.connect(self._on_busy)

        self._build_central()
        self.statusBar().showMessage("Ready")

    def _build_central(self) -> None:
        self.camera_view = CameraView()
        self.camera_view.log.connect(self._on_log)   # camera actions -> Log panel
        self.connection_panel = ConnectionPanel(self.service)
        self.calibration_panel = CalibrationPanel(self.camera_view, self.service)
        self.log_panel = LogPanel()

        # Right column: robot controls on top, log below.
        right = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(self.connection_panel)
        right_layout.addWidget(self.calibration_panel)
        right_layout.addWidget(QtWidgets.QLabel("Log"))
        right_layout.addWidget(self.log_panel, 1)

        # Left: live camera. Split so the user can resize the video vs. controls.
        splitter = QtWidgets.QSplitter()
        splitter.addWidget(self.camera_view)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        central = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(central)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.addWidget(splitter)
        self.setCentralWidget(central)

    def _on_log(self, message: str, level: str) -> None:
        self.log_panel.append(message, level)

    def _on_busy(self, busy: bool) -> None:
        self.statusBar().showMessage("Working…" if busy else "Ready")

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        # Tear down the worker threads cleanly before the window goes away.
        self.camera_view.shutdown()
        self.service.shutdown()
        super().closeEvent(event)
