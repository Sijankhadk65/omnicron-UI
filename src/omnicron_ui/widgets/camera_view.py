"""Live camera view for the Orbbec Gemini 336L, with red-line AOI detection.

A self-contained widget: Start/Stop the stream, a "Detect line" toggle, and a
"Clear AOI" button. Drag a box on the video to set the Area Of Interest — red-line
detection is then restricted to that box (mirrors farino_app/red_line_viewer.py).

Capture and detection both run on ``CameraService``'s worker thread; the widget
only displays frames (received as ``QImage`` signals) and translates the user's
mouse drag into a frame-coordinate AOI. The committed AOI box and the fitted line
are drawn into the frame by the worker; the in-progress drag rectangle is drawn
here by ``VideoLabel`` for responsive feedback.
"""

from __future__ import annotations

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPixmap

from omnicron_ui.camera.service import CameraService


class VideoLabel(QtWidgets.QLabel):
    """QLabel that shows the video and lets the user drag an AOI rectangle.

    Emits ``aoiDragged(QRect)`` (in label/display coordinates) on mouse release;
    the parent maps that to frame coordinates. Draws the in-progress rectangle
    itself so the drag feels responsive regardless of frame rate.
    """

    aoiDragged = Signal(QtCore.QRect)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._origin: QtCore.QPoint | None = None
        self._current: QtCore.QPoint | None = None
        self._dragging = False

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = True
            self._origin = self._current = event.position().toPoint()
            self.update()

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._dragging:
            self._current = event.position().toPoint()
            self.update()

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._dragging and event.button() == Qt.MouseButton.LeftButton:
            self._dragging = False
            self._current = event.position().toPoint()
            rect = QtCore.QRect(self._origin, self._current).normalized()
            self._origin = self._current = None
            self.update()
            self.aoiDragged.emit(rect)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        super().paintEvent(event)
        if self._dragging and self._origin and self._current:
            painter = QtGui.QPainter(self)
            painter.setPen(QtGui.QPen(QtGui.QColor(0, 200, 0), 1, Qt.PenStyle.DashLine))
            painter.drawRect(QtCore.QRect(self._origin, self._current).normalized())


class CameraView(QtWidgets.QWidget):
    # Forwarded to the shared Log panel by MainWindow. Every action logs here.
    log = Signal(str, str)  # (message, level)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._service = CameraService()
        self._service.frame.connect(self._on_frame)
        self._service.started.connect(self._on_started)
        self._service.status.connect(self._on_status)
        self._service.stopped.connect(self._on_stopped)
        self._service.line.connect(self._on_line)
        self._service.log.connect(self.log)   # forward worker logs to the panel

        self.image = VideoLabel("Camera stopped")
        self.image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image.setMinimumSize(640, 360)
        self.image.setStyleSheet("background:#111; color:#888; border:1px solid #333;")
        self.image.aoiDragged.connect(self._on_aoi_dragged)

        self.toggle_btn = QtWidgets.QPushButton("Start camera")
        self.toggle_btn.clicked.connect(self._on_toggle)

        self.detect_btn = QtWidgets.QPushButton("Detect line")
        self.detect_btn.clicked.connect(self._on_detect_clicked)

        self.clear_detect_btn = QtWidgets.QPushButton("Clear detection")
        self.clear_detect_btn.clicked.connect(self._on_clear_detection)

        self.clear_aoi_btn = QtWidgets.QPushButton("Clear AOI")
        self.clear_aoi_btn.clicked.connect(self._clear_aoi)

        self.status_lbl = QtWidgets.QLabel("Drag on the video to set an AOI")
        self.status_lbl.setStyleSheet("color:#888;")

        controls = QtWidgets.QHBoxLayout()
        controls.addWidget(self.toggle_btn)
        controls.addWidget(self.detect_btn)
        controls.addWidget(self.clear_detect_btn)
        controls.addWidget(self.clear_aoi_btn)
        controls.addWidget(self.status_lbl, 1)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.image, 1)
        layout.addLayout(controls)

        self._running = False
        self._frame_wh: tuple[int, int] | None = None
        self._last_frame: QImage | None = None
        self._aoi: tuple[int, int, int, int] | None = None
        self._pick_mode = False   # calibration: clicks sample a depth point

    @property
    def service(self) -> CameraService:
        """The camera worker service (for panels that need its signals)."""
        return self._service

    @property
    def running(self) -> bool:
        return self._running

    def set_pick_mode(self, on: bool) -> None:
        """Calibration mode: plain clicks on the video pick a depth point
        (AOI drags still work — only sub-10px 'drags' count as clicks)."""
        self._pick_mode = on

    # --- start/stop ---
    def _on_toggle(self) -> None:
        if self._running:
            self.stop()
        else:
            self.start()

    def start(self) -> None:
        self._running = True
        self.toggle_btn.setText("Stop camera")
        self.image.setText("Starting camera…")
        self.log.emit("Starting camera…", "info")
        self._service.start()

    def stop(self) -> None:
        self.log.emit("Stopping camera…", "info")
        self._service.stop()

    # --- detection ---
    def _on_detect_clicked(self) -> None:
        """One-shot: detect the longest red line inside the AOI on the next frame."""
        if not self._running:
            self.log.emit("Detect ignored — start the camera first", "warn")
            return
        if self._aoi is None:
            self.log.emit("Detect ignored — drag an AOI on the video first", "warn")
            return
        self.log.emit("Detecting red line in AOI…", "info")
        self._service.detect_once()   # result/failure is logged by the service

    def _on_clear_detection(self) -> None:
        self._service.clear_detection()
        self.status_lbl.setText("Detection cleared")
        self.log.emit("Detection cleared", "info")

    # --- AOI ---
    def _on_aoi_dragged(self, rect: QtCore.QRect) -> None:
        """Map a label-coordinate drag rectangle to frame coords and commit it."""
        if self._frame_wh is None:
            self.log.emit("AOI ignored — camera not started yet", "warn")
            return
        p1 = self._label_to_frame(rect.topLeft())
        p2 = self._label_to_frame(rect.bottomRight())
        if p1 is None or p2 is None:
            return
        x1, x2 = sorted((p1[0], p2[0]))
        y1, y2 = sorted((p1[1], p2[1]))
        if x2 - x1 < 10 and y2 - y1 < 10:
            # A tiny drag is a CLICK: in calibration mode it picks a depth point.
            if self._pick_mode:
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                self.log.emit(f"Sampling depth at ({cx},{cy})…", "info")
                self._service.pick_point(cx, cy)
            return
        if x2 - x1 < 10 or y2 - y1 < 10:   # ignore thin/accidental drags
            self.log.emit("AOI too small — ignored", "warn")
            return
        self._aoi = (x1, y1, x2, y2)
        self._service.set_aoi(self._aoi)
        self.status_lbl.setText(f"AOI ({x1},{y1})–({x2},{y2})")
        self.log.emit(f"AOI set to ({x1},{y1})–({x2},{y2})", "success")

    def _clear_aoi(self) -> None:
        self._aoi = None
        self._service.set_aoi(None)
        self.status_lbl.setText("AOI cleared")
        self.log.emit("AOI cleared", "info")

    def _label_to_frame(self, pt: QtCore.QPoint) -> tuple[int, int] | None:
        """Map a point on the label to full-frame pixel coords.

        The pixmap is shown scaled KeepAspectRatio and centered, so undo the
        letterbox offset and scale. Returns None until the frame size is known.
        """
        if self._frame_wh is None:
            return None
        fw, fh = self._frame_wh
        lw, lh = self.image.width(), self.image.height()
        scale = min(lw / fw, lh / fh)
        if scale <= 0:
            return None
        off_x = (lw - fw * scale) / 2
        off_y = (lh - fh * scale) / 2
        fx = (pt.x() - off_x) / scale
        fy = (pt.y() - off_y) / scale
        fx = min(max(fx, 0), fw - 1)
        fy = min(max(fy, 0), fh - 1)
        return int(round(fx)), int(round(fy))

    # --- service signals ---
    def _on_started(self, w: int, h: int, fps: int) -> None:
        self._running = True
        self._frame_wh = (w, h)
        self.toggle_btn.setText("Stop camera")

    def _on_frame(self, image: QImage) -> None:
        self._last_frame = image
        self._render()

    def _on_status(self, text: str) -> None:
        self.status_lbl.setText(text)

    def _on_line(self, det) -> None:
        """Show the detection: pixel endpoints, plus camera-frame mm with depth."""
        if det is None:
            self.status_lbl.setText("no red line")
            return
        p1, p2 = det.endpoints
        length = float(np.hypot(p2[0] - p1[0], p2[1] - p1[1]))
        text = (f"line: P1({p1[0]:.0f},{p1[1]:.0f}) "
                f"P2({p2[0]:.0f},{p2[1]:.0f})  {length:.0f}px")
        if det.has_base_xyz:
            b1, b2 = det.base1, det.base2
            text += (f"  |  base mm P1({b1[0]:.0f},{b1[1]:.0f},{b1[2]:.0f}) "
                     f"P2({b2[0]:.0f},{b2[1]:.0f},{b2[2]:.0f})")
        elif det.has_camera_xyz:
            c1, c2 = det.cam1, det.cam2
            text += (f"  |  cam mm P1({c1[0]:.0f},{c1[1]:.0f},{c1[2]:.0f}) "
                     f"P2({c2[0]:.0f},{c2[1]:.0f},{c2[2]:.0f})")
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

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:  # noqa: N802 (Qt override)
        self._render()
        super().resizeEvent(event)

    def shutdown(self) -> None:
        """Stop the camera worker thread; call on app close."""
        self._service.stop()
