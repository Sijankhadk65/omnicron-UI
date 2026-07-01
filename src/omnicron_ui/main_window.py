"""The main application window.

Owns the single ``RobotService`` (the async worker thread) and hosts the feature
panels. The window never makes a blocking robot call directly — it only submits
jobs to the service and reacts to the service's signals, so the UI stays
responsive while the robot is talking.
"""

from __future__ import annotations

from PySide6 import QtGui, QtWidgets

from omnicron_ui.robot.service import RobotService
from omnicron_ui.widgets.connection_panel import ConnectionPanel
from omnicron_ui.widgets.log_panel import LogPanel
from omnicron_ui.widgets.ptp_panel import PtpPanel

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
        self.connection_panel = ConnectionPanel(self.service)
        self.ptp_panel = PtpPanel(self.service)
        self.log_panel = LogPanel()

        central = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(central)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.addWidget(self.connection_panel)
        layout.addWidget(self.ptp_panel)
        layout.addWidget(QtWidgets.QLabel("Log"))
        layout.addWidget(self.log_panel, 1)
        self.setCentralWidget(central)

    def _on_log(self, message: str, level: str) -> None:
        self.log_panel.append(message, level)

    def _on_busy(self, busy: bool) -> None:
        self.statusBar().showMessage("Working…" if busy else "Ready")

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        # Tear down the worker thread cleanly before the window goes away.
        self.service.shutdown()
        super().closeEvent(event)
