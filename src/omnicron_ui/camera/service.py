"""Asynchronous camera service — streams 336L frames off the Qt UI thread.

``OrbbecCamera.read()`` blocks on ``wait_for_frames``, so capture runs on its own
worker thread and each frame is handed to the UI as a ``QImage`` via a signal.
This mirrors ``robot.service.RobotService``: the UI never makes a blocking camera
call; it just starts/stops the service and reacts to signals.

Pattern: a ``QObject`` moved onto a ``QThread`` whose ``run`` slot loops on the
worker thread until a stop flag is set. ``QImage`` is copied before emit so it
owns its pixels (the source numpy buffer is reused each frame).
"""

from __future__ import annotations

import threading

import numpy as np
from PySide6.QtCore import QObject, QThread, Signal
from PySide6.QtGui import QImage


class CameraService(QObject):
    """Runs the 336L capture loop on a worker thread; emits frames as signals."""

    frame = Signal(QImage)     # a new color frame, ready to display
    started = Signal(int, int, int)  # width, height, fps
    status = Signal(str)       # short human-readable status/errors
    stopped = Signal()
    line = Signal(object)      # red-line endpoints (p1, p2) or None, per frame
    log = Signal(str, str)     # (message, level) for the shared Log panel

    def __init__(self) -> None:
        super().__init__()
        self._stop = threading.Event()
        # Detection state, set from the UI thread and read on the worker thread.
        # Plain attribute assignment is atomic under the GIL, so no lock needed.
        self._aoi: tuple[int, int, int, int] | None = None
        # Detection is ONE-SHOT: a press sets _detect_request; the worker runs it
        # once on the next frame, stores the result in _result, and keeps drawing
        # that result until it's re-detected (override) or cleared.
        self._detect_request = False
        self._result: object = None
        self._thread = QThread()
        self._thread.setObjectName("camera-worker")
        self.moveToThread(self._thread)
        self._thread.started.connect(self._run)

    # --- public API (call from the UI thread) -------------------------------

    def start(self) -> None:
        if self._thread.isRunning():
            return
        self._stop.clear()
        self._thread.start()

    def set_aoi(self, aoi: tuple[int, int, int, int] | None) -> None:
        """Restrict detection to this frame-coordinate rectangle (or None = full).

        Changing the AOI clears any existing detection — the stored result belongs
        to the old region and would otherwise linger.
        """
        self._aoi = aoi
        self._result = None

    def detect_once(self) -> None:
        """Request a single detection on the next frame (overrides any current)."""
        self._detect_request = True

    def clear_detection(self) -> None:
        """Drop the current detection so nothing is drawn until the next detect."""
        self._detect_request = False
        self._result = None

    def stop(self) -> None:
        """Stop the capture loop and wait for the worker thread to finish."""
        self._stop.set()
        self._thread.quit()
        self._thread.wait(3000)

    # --- worker thread ------------------------------------------------------

    def _run(self) -> None:
        # Imported here so the (blocking, hardware-touching) SDK is only loaded on
        # the worker thread, never during UI import.
        from omnicron_ui.camera import red_line
        from omnicron_ui.camera.orbbec import OrbbecCamera

        camera = OrbbecCamera()
        try:
            w, h, fps = camera.open()
            self.started.emit(w, h, fps)
            self.status.emit(f"Streaming {w}x{h} @ {fps} fps")
            self.log.emit(f"Camera started — {w}x{h} @ {fps} fps", "success")
        except Exception as exc:  # noqa: BLE001 — surface open failures to the UI
            self.status.emit(f"Camera open failed: {exc}")
            self.log.emit(f"Camera open failed: {exc}", "error")
            return

        try:
            while not self._stop.is_set():
                try:
                    bgr = camera.read(timeout_ms=1000)
                    if bgr is None:
                        continue
                    aoi = self._aoi
                    # One-shot: only run the detector when a press requested it,
                    # and only ever inside an AOI (no whole-frame detection).
                    if self._detect_request:
                        self._detect_request = False
                        if aoi is None:
                            self.log.emit("No AOI set — detection skipped", "warn")
                        else:
                            self._result = red_line.detect(bgr, aoi)
                            self.line.emit(self._result)
                            self._log_detection(self._result, aoi)
                    # Always draw the committed AOI + the last stored detection
                    # (which persists until re-detected or cleared).
                    red_line.draw_overlay(bgr, aoi, self._result)
                    self.frame.emit(self._to_qimage(bgr))
                except Exception as exc:  # noqa: BLE001 — one bad frame shouldn't kill the stream
                    self.status.emit(f"Frame error: {exc}")
                    continue
        finally:
            camera.close()
            self.stopped.emit()

    def _log_detection(self, endpoints, aoi) -> None:
        """Log the result of a one-shot detection (every press logs)."""
        where = "AOI" if aoi is not None else "frame"
        if endpoints is not None:
            p1, p2 = endpoints
            length = float(np.hypot(p2[0] - p1[0], p2[1] - p1[1]))
            self.log.emit(
                f"Red line detected in {where}: "
                f"P1({p1[0]:.0f},{p1[1]:.0f}) P2({p2[0]:.0f},{p2[1]:.0f}) {length:.0f}px",
                "success",
            )
        else:
            self.log.emit(f"No red line found in {where}", "warn")

    @staticmethod
    def _to_qimage(bgr: np.ndarray) -> QImage:
        """BGR ndarray -> owned RGB QImage (copied off the reused numpy buffer)."""
        rgb = np.ascontiguousarray(bgr[:, :, ::-1])
        h, w = rgb.shape[:2]
        img = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        return img.copy()
