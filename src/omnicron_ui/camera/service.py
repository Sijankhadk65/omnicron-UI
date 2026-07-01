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
    line = Signal(object)      # red_line.LineDetection (px + camera XYZ) or None
    picked = Signal(object)    # ((u, v), np.array([X, Y, Z]) camera mm) or None
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
        # Aligned depth is ON by default — the 336L is a depth camera and both
        # detection (endpoint mm) and calibration need it. It can only be
        # changed BEFORE start (the pipeline config is fixed while streaming).
        self._depth_enabled = True
        self._depth_active = False          # what the RUNNING stream actually has
        self._pick_request: tuple[int, int] | None = None
        self._markers: tuple = (None, ())   # (pending px, banked px list)
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

    def set_depth_enabled(self, on: bool) -> None:
        """Enable/disable the aligned depth stream (default ON). Takes effect on
        the NEXT start() — the pipeline config is fixed while streaming."""
        self._depth_enabled = on

    @property
    def depth_active(self) -> bool:
        """True while the RUNNING stream actually delivers aligned depth."""
        return self._depth_active

    def pick_point(self, x: int, y: int) -> None:
        """Sample depth at pixel (x, y) on the next frame and deproject it.

        Result arrives on the ``picked`` signal as ((u, v), camera XYZ mm), or
        None when there is no depth there (hole) — click the marker again.
        """
        self._pick_request = (int(x), int(y))

    def set_markers(self, pending: tuple[int, int] | None,
                    banked: list[tuple[int, int]]) -> None:
        """Calibration overlay: the pending click (cross) + banked points (ticks)."""
        self._markers = (pending, tuple(banked))

    def stop(self) -> None:
        """Stop the capture loop and wait for the worker thread to finish."""
        self._stop.set()
        self._thread.quit()
        self._thread.wait(3000)

    # --- worker thread ------------------------------------------------------

    def _run(self) -> None:
        # Imported here so the (blocking, hardware-touching) SDK is only loaded on
        # the worker thread, never during UI import.
        from omnicron_ui.camera import calibration, red_line
        from omnicron_ui.camera.orbbec import OrbbecCamera

        camera = OrbbecCamera()
        with_depth = self._depth_enabled
        try:
            w, h, fps = camera.open(with_depth=with_depth)
            self._depth_active = with_depth
            self.started.emit(w, h, fps)
            mode = " + aligned depth" if with_depth else ""
            self.status.emit(f"Streaming {w}x{h} @ {fps} fps{mode}")
            self.log.emit(f"Camera started — {w}x{h} @ {fps} fps{mode}", "success")
        except Exception as exc:  # noqa: BLE001 — surface open failures to the UI
            self.status.emit(f"Camera open failed: {exc}")
            self.log.emit(f"Camera open failed: {exc}", "error")
            return

        try:
            while not self._stop.is_set():
                try:
                    bgr, depth = camera.read_with_depth(timeout_ms=1000)
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
                            endpoints = red_line.detect(bgr, aoi)
                            self._result = self._build_detection(
                                calibration, red_line, camera, endpoints, depth)
                            self.line.emit(self._result)
                            self._log_detection(self._result, aoi,
                                                depth_on=depth is not None)
                    if self._pick_request is not None:
                        px = self._pick_request
                        self._pick_request = None
                        self._handle_pick(calibration, camera, px, depth)
                    # Always draw the committed AOI + the last stored detection
                    # (which persists until re-detected or cleared).
                    red_line.draw_overlay(
                        bgr, aoi,
                        self._result.endpoints if self._result is not None else None)
                    self._draw_markers(bgr)
                    self.frame.emit(self._to_qimage(bgr))
                except Exception as exc:  # noqa: BLE001 — one bad frame shouldn't kill the stream
                    self.status.emit(f"Frame error: {exc}")
                    continue
        finally:
            camera.close()
            self._depth_active = False
            self.stopped.emit()

    def _handle_pick(self, calibration, camera, px, depth) -> None:
        """Depth-sample one clicked pixel and emit the deprojected camera XYZ."""
        if depth is None:
            self.log.emit("Pick ignored — no depth stream (enable calibration "
                          "mode and restart the camera)", "warn")
            self.picked.emit(None)
            return
        z = calibration.sample_depth_mm(depth, px)
        if z <= 0:
            self.log.emit(f"No depth at ({px[0]},{px[1]}) — hole; click the "
                          "marker again", "warn")
            self.picked.emit(None)
            return
        cam_xyz = calibration.pixel_to_camera(px, z, camera.intrinsics)
        self.log.emit(f"Camera XYZ at ({px[0]},{px[1]}): "
                      f"({cam_xyz[0]:.1f}, {cam_xyz[1]:.1f}, {cam_xyz[2]:.1f}) mm",
                      "success")
        self.picked.emit((px, cam_xyz))

    def _draw_markers(self, bgr) -> None:
        """Draw calibration markers: banked points (ticks) + the pending click."""
        import cv2 as cv

        pending, banked = self._markers
        for p in banked:
            cv.drawMarker(bgr, p, (0, 255, 0), cv.MARKER_TILTED_CROSS, 12, 1)
        if pending is not None:
            cv.drawMarker(bgr, pending, (0, 255, 255), cv.MARKER_CROSS, 16, 2)

    @staticmethod
    def _build_detection(calibration, red_line, camera, endpoints, depth):
        """Package a detection: pixel endpoints + camera XYZ when depth ran."""
        if endpoints is None:
            return None
        cam1 = cam2 = None
        if depth is not None and camera.intrinsics is not None:
            cam1, cam2 = calibration.line_to_camera(endpoints, depth,
                                                    camera.intrinsics)
        return red_line.LineDetection(p1=endpoints[0], p2=endpoints[1],
                                      cam1=cam1, cam2=cam2)

    def _log_detection(self, det, aoi, depth_on: bool) -> None:
        """Log the result of a one-shot detection (every press logs)."""
        where = "AOI" if aoi is not None else "frame"
        if det is None:
            self.log.emit(f"No red line found in {where}", "warn")
            return
        p1, p2 = det.endpoints
        length = float(np.hypot(p2[0] - p1[0], p2[1] - p1[1]))
        self.log.emit(
            f"Red line detected in {where} — image pixels: "
            f"P1 x={p1[0]:.0f} y={p1[1]:.0f}, P2 x={p2[0]:.0f} y={p2[1]:.0f}, "
            f"length {length:.0f} px",
            "success",
        )
        if not depth_on:
            self.log.emit("No camera-frame x/y/z — this stream has no depth "
                          "(started color-only). Restart the camera and "
                          "re-detect to get millimetres.", "warn")
            return
        for label, cam in (("P1", det.cam1), ("P2", det.cam2)):
            if cam is None:
                self.log.emit(f"{label}: no depth at the endpoint (hole) — "
                              "retake or nudge the AOI", "warn")
            else:
                self.log.emit(
                    f"{label} camera frame: x={cam[0]:.1f} mm, y={cam[1]:.1f} mm, "
                    f"z={cam[2]:.1f} mm  (x right, y down, z out from the lens)",
                    "success",
                )

    @staticmethod
    def _to_qimage(bgr: np.ndarray) -> QImage:
        """BGR ndarray -> owned RGB QImage (copied off the reused numpy buffer)."""
        rgb = np.ascontiguousarray(bgr[:, :, ::-1])
        h, w = rgb.shape[:2]
        img = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        return img.copy()
