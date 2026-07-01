"""Live camera view for the Orbbec Gemini 336L.

A self-contained widget: a Start/Stop button and a QLabel that shows the latest
frame from ``CameraService`` (which streams on its own worker thread). The label
never blocks — it just receives ``QImage``s via signal and repaints, scaled to
fit while keeping aspect ratio.
"""

from __future__ import annotations

from PySide6 import QtWidgets
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap

from omnicron_ui.camera.service import CameraService


class CameraView(QtWidgets.QWidget):
    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._service = CameraService()
        self._service.frame.connect(self._on_frame)
        self._service.started.connect(self._on_started)
        self._service.status.connect(self._on_status)
        self._service.stopped.connect(self._on_stopped)

        self.image = QtWidgets.QLabel("Camera stopped")
        self.image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image.setMinimumSize(640, 360)
        self.image.setStyleSheet(
            "background:#111; color:#888; border:1px solid #333;"
        )

        self.toggle_btn = QtWidgets.QPushButton("Start camera")
        self.toggle_btn.clicked.connect(self._on_toggle)

        self.status_lbl = QtWidgets.QLabel("")
        self.status_lbl.setStyleSheet("color:#888;")

        controls = QtWidgets.QHBoxLayout()
        controls.addWidget(self.toggle_btn)
        controls.addWidget(self.status_lbl, 1)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.image, 1)
        layout.addLayout(controls)

        self._running = False
        self._last_frame: QImage | None = None

    # --- button ---
    def _on_toggle(self) -> None:
        if self._running:
            self.stop()
        else:
            self.start()

    def start(self) -> None:
        self._running = True
        self.toggle_btn.setText("Stop camera")
        self.image.setText("Starting camera…")
        self._service.start()

    def stop(self) -> None:
        self._service.stop()

    # --- service signals ---
    def _on_started(self, w: int, h: int, fps: int) -> None:
        self._running = True
        self.toggle_btn.setText("Stop camera")

    def _on_frame(self, image: QImage) -> None:
        self._last_frame = image
        self._render()

    def _on_status(self, text: str) -> None:
        self.status_lbl.setText(text)

    def _on_stopped(self) -> None:
        self._running = False
        self.toggle_btn.setText("Start camera")
        self._last_frame = None
        self.image.clear()
        self.image.setText("Camera stopped")

    # --- rendering ---
    def _render(self) -> None:
        if self._last_frame is None:
            return
        pix = QPixmap.fromImage(self._last_frame).scaled(
            self.image.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.image.setPixmap(pix)

    def resizeEvent(self, event) -> None:  # noqa: N802 (Qt override)
        # Re-scale the current frame to the new label size.
        self._render()
        super().resizeEvent(event)

    def shutdown(self) -> None:
        """Stop the camera worker thread; call on app close."""
        self._service.stop()
