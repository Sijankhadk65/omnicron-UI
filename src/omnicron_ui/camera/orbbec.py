"""Blocking Orbbec Gemini 336L access — Qt-free, runs on the camera worker thread.

The 336L's color/depth only come through Orbbec's ``pyorbbecsdk`` (the
``pyorbbecsdk2`` wheel), not the UVC ``/dev/videoN`` path the old Astra Pro used.
``pipeline.wait_for_frames()`` BLOCKS (up to the timeout), so — like the robot
SDK — this must never be driven from the Qt main thread. ``CameraService`` owns
the thread; this class owns the pipeline and returns plain BGR ndarrays.

Only the color stream is handled here for now; depth/alignment can be added later
without changing the threading model.
"""

from __future__ import annotations

import cv2 as cv
import numpy as np
from pyorbbecsdk import (  # type: ignore
    Config,
    OBFormat,
    OBSensorType,
    Pipeline,
)


def _color_frame_to_bgr(frame) -> np.ndarray | None:
    """Convert an Orbbec color VideoFrame to a BGR ndarray for display."""
    w, h = frame.get_width(), frame.get_height()
    fmt = frame.get_format()
    data = np.asanyarray(frame.get_data())
    if fmt == OBFormat.RGB:
        return cv.cvtColor(np.resize(data, (h, w, 3)), cv.COLOR_RGB2BGR)
    if fmt == OBFormat.BGR:
        return np.resize(data, (h, w, 3))
    if fmt == OBFormat.MJPG:
        return cv.imdecode(data, cv.IMREAD_COLOR)
    if fmt == OBFormat.YUYV:
        return cv.cvtColor(np.resize(data, (h, w, 2)), cv.COLOR_YUV2BGR_YUYV)
    if fmt == OBFormat.UYVY:
        return cv.cvtColor(np.resize(data, (h, w, 2)), cv.COLOR_YUV2BGR_UYVY)
    if fmt == OBFormat.NV12:
        return cv.cvtColor(np.resize(data, (h * 3 // 2, w)), cv.COLOR_YUV2BGR_NV12)
    if fmt == OBFormat.NV21:
        return cv.cvtColor(np.resize(data, (h * 3 // 2, w)), cv.COLOR_YUV2BGR_NV21)
    return None


class OrbbecCamera:
    """Opens the 336L color stream and yields BGR frames. Blocking; no Qt."""

    def __init__(self) -> None:
        self._pipeline: Pipeline | None = None

    @property
    def opened(self) -> bool:
        return self._pipeline is not None

    def open(self) -> tuple[int, int, int]:
        """Start the color pipeline. Returns (width, height, fps) of the stream.

        Raises the underlying pyorbbecsdk error if no device is present or the
        stream can't start (e.g. camera held by another process, or not on USB3).
        """
        pipeline = Pipeline()
        config = Config()
        profiles = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        profile = profiles.get_default_video_stream_profile()
        config.enable_stream(profile)
        pipeline.start(config)
        self._pipeline = pipeline
        return profile.get_width(), profile.get_height(), profile.get_fps()

    def read(self, timeout_ms: int = 1000) -> np.ndarray | None:
        """Block for the next color frame; return a BGR ndarray, or None.

        The 336L color stream comes in mirrored left-right, so we flip it
        horizontally here (``cv.flip(..., 1)``) so every consumer — display, line
        detection, calibration — sees a correctly-oriented image.
        """
        if self._pipeline is None:
            return None
        frames = self._pipeline.wait_for_frames(timeout_ms)
        if frames is None:
            return None
        color = frames.get_color_frame()
        if color is None:
            return None
        bgr = _color_frame_to_bgr(color)
        if bgr is None:
            return None
        return cv.flip(bgr, 1)

    def close(self) -> None:
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except Exception:  # noqa: BLE001 — best-effort teardown
                pass
        self._pipeline = None
