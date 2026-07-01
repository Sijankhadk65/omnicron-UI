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

    def __init__(self) -> None:
        super().__init__()
        self._stop = threading.Event()
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

    def stop(self) -> None:
        """Stop the capture loop and wait for the worker thread to finish."""
        self._stop.set()
        self._thread.quit()
        self._thread.wait(3000)

    # --- worker thread ------------------------------------------------------

    def _run(self) -> None:
        # Imported here so the (blocking, hardware-touching) SDK is only loaded on
        # the worker thread, never during UI import.
        from omnicron_ui.camera.orbbec import OrbbecCamera

        camera = OrbbecCamera()
        try:
            w, h, fps = camera.open()
            self.started.emit(w, h, fps)
            self.status.emit(f"Streaming {w}x{h} @ {fps} fps")
        except Exception as exc:  # noqa: BLE001 — surface open failures to the UI
            self.status.emit(f"Camera open failed: {exc}")
            return

        try:
            while not self._stop.is_set():
                try:
                    bgr = camera.read(timeout_ms=1000)
                    if bgr is None:
                        continue
                    self.frame.emit(self._to_qimage(bgr))
                except Exception as exc:  # noqa: BLE001 — one bad frame shouldn't kill the stream
                    self.status.emit(f"Frame error: {exc}")
                    continue
        finally:
            camera.close()
            self.stopped.emit()

    @staticmethod
    def _to_qimage(bgr: np.ndarray) -> QImage:
        """BGR ndarray -> owned RGB QImage (copied off the reused numpy buffer)."""
        rgb = np.ascontiguousarray(bgr[:, :, ::-1])
        h, w = rgb.shape[:2]
        img = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        return img.copy()
