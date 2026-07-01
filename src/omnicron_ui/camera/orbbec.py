"""Blocking Orbbec Gemini 336L access — Qt-free, runs on the camera worker thread.

The 336L's color/depth only come through Orbbec's ``pyorbbecsdk`` (the
``pyorbbecsdk2`` wheel), not the UVC ``/dev/videoN`` path the old Astra Pro used.
``pipeline.wait_for_frames()`` BLOCKS (up to the timeout), so — like the robot
SDK — this must never be driven from the Qt main thread. ``CameraService`` owns
the thread; this class owns the pipeline and returns plain BGR ndarrays.

Depth is opt-in (``open(with_depth=True)``): the depth stream is aligned to the
color stream so depth[v, u] is the depth AT color pixel (u, v), and the RGB
intrinsics are exposed for deprojection — what the camera→base calibration and
the 3D endpoint mapping need.
"""

from __future__ import annotations

import cv2 as cv
import numpy as np
from pyorbbecsdk import (  # type: ignore
    AlignFilter,
    Config,
    OBFormat,
    OBSensorType,
    OBStreamType,
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


def _depth_frame_to_mm(depth_frame) -> np.ndarray:
    """Convert an Orbbec depth VideoFrame to a float32 mm image (0 = hole)."""
    w, h = depth_frame.get_width(), depth_frame.get_height()
    scale = depth_frame.get_depth_scale()
    raw = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape((h, w))
    return raw.astype(np.float32) * scale


class OrbbecCamera:
    """Opens the 336L color (+ optional aligned depth) stream. Blocking; no Qt."""

    def __init__(self) -> None:
        self._pipeline: Pipeline | None = None
        self._align: AlignFilter | None = None
        self._intrinsics = None

    @property
    def opened(self) -> bool:
        return self._pipeline is not None

    @property
    def intrinsics(self):
        """RGB intrinsics (fx/fy/cx/cy attrs) — set only after open(with_depth=True)."""
        return self._intrinsics

    def open(self, with_depth: bool = False) -> tuple[int, int, int]:
        """Start the pipeline. Returns (width, height, fps) of the color stream.

        With ``with_depth`` the depth sensor is enabled too, depth frames are
        ALIGNED to the color stream (depth[v, u] pairs with color pixel (u, v)),
        and the RGB intrinsics are captured for deprojection.

        Raises the underlying pyorbbecsdk error if no device is present or the
        stream can't start (e.g. camera held by another process, or not on USB3).
        """
        pipeline = Pipeline()
        config = Config()
        profiles = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        profile = profiles.get_default_video_stream_profile()
        config.enable_stream(profile)
        if with_depth:
            depth_profiles = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
            config.enable_stream(depth_profiles.get_default_video_stream_profile())
        pipeline.start(config)
        if with_depth:
            self._align = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)
            self._intrinsics = pipeline.get_camera_param().rgb_intrinsic
        self._pipeline = pipeline
        return profile.get_width(), profile.get_height(), profile.get_fps()

    def read(self, timeout_ms: int = 1000) -> np.ndarray | None:
        """Block for the next color frame; return a BGR ndarray, or None.

        Frames are returned in the camera's native orientation (no flip).
        """
        bgr, _ = self.read_with_depth(timeout_ms)
        return bgr

    def read_with_depth(
        self, timeout_ms: int = 1000
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Block for the next frame set; return (BGR, aligned depth mm).

        Depth is None when the camera was opened color-only, or when the depth
        frame is missing from this frame set.
        """
        if self._pipeline is None:
            return None, None
        frames = self._pipeline.wait_for_frames(timeout_ms)
        if frames is None:
            return None, None
        depth_mm = None
        if self._align is not None:
            aligned = self._align.process(frames)
            if aligned is not None:
                frames = aligned.as_frame_set()
                depth = frames.get_depth_frame()
                if depth is not None:
                    depth_mm = _depth_frame_to_mm(depth)
        color = frames.get_color_frame()
        if color is None:
            return None, None
        return _color_frame_to_bgr(color), depth_mm

    def close(self) -> None:
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except Exception:  # noqa: BLE001 — best-effort teardown
                pass
        self._pipeline = None
        self._align = None
        self._intrinsics = None
