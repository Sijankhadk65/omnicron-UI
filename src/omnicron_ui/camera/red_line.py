"""Red-line detection restricted to an Area Of Interest (AOI). Pure cv/numpy, no Qt.

Ported from farino_app/red_line_viewer.py. The pipeline: build a red mask (BGR
channel dominance, or CIELab a* for faint marker-on-paper), zero everything
outside the AOI, then keep the most LINE-SHAPED region (elongation filter, so fat
red clutter is rejected) and return its two extreme endpoints.

An AOI is a ``(x1, y1, x2, y2)`` rectangle in full-frame pixel coordinates, or
``None`` to search the whole frame.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2 as cv
import numpy as np

# --- Detection thresholds (same defaults as red_line_viewer.py) ---
RED_DIFF = 20          # BGR: how much R must exceed max(G,B) / Lab: a* above neutral
RED_MIN_VALUE = 60     # BGR only: ignore very dark reddish pixels (shadows)
MIN_LINE_AREA = 80     # ignore red specks smaller than this (px)
MIN_ELONGATION = 3.0   # long/short side ratio to qualify as a line, not a blob

AOI = tuple[int, int, int, int]
Endpoints = tuple[np.ndarray, np.ndarray]


@dataclass(frozen=True)
class LineDetection:
    """One detected red line: pixel endpoints + (when depth ran) camera XYZ.

    ``cam1``/``cam2`` are camera-frame mm from aligned depth + intrinsics, or
    None per endpoint when depth had a hole there — and both None when the
    stream is color-only. They are what ``calibration.cam_to_base`` maps to
    robot base coordinates.
    """

    p1: np.ndarray
    p2: np.ndarray
    cam1: np.ndarray | None = None
    cam2: np.ndarray | None = None

    @property
    def endpoints(self) -> Endpoints:
        return self.p1, self.p2

    @property
    def has_camera_xyz(self) -> bool:
        return self.cam1 is not None and self.cam2 is not None


def build_red_mask(image, diff=RED_DIFF, min_val=RED_MIN_VALUE, lab_mode=False):
    """Mask of reddish pixels, morphologically cleaned.

    BGR channel dominance (default): R exceeds both G and B by ``diff`` and
    R > ``min_val``. Robust for saturated red. Lab a* (``lab_mode``): positive for
    red regardless of lightness, so faint red-on-white still clears the threshold.

    CLOSE (5x5) bridges gaps in a broken thin line; OPEN (3x3) clears specks.
    """
    if lab_mode:
        a = cv.cvtColor(image, cv.COLOR_BGR2Lab)[:, :, 1].astype(np.int16)
        mask = ((a - 128) > diff).astype(np.uint8) * 255
    else:
        b, g, r = cv.split(image.astype(np.int16))
        redness = r - np.maximum(g, b)
        mask = ((redness > diff) & (r > min_val)).astype(np.uint8) * 255
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE,
                           cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5)))
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN,
                           cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3)))
    return mask


def crop_to_aoi(mask, aoi: AOI | None):
    """Zero everything outside the AOI so the detector only sees inside it."""
    if aoi is None:
        return mask
    out = np.zeros_like(mask)
    x1, y1, x2, y2 = aoi
    out[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
    return out


def detect_endpoints(mask) -> Endpoints | None:
    """Return (p1, p2) endpoints of the most LINE-SHAPED red region, or None.

    Among red regions keep only elongated ones (min-area-rect aspect ratio >=
    MIN_ELONGATION), pick the longest, then take the extreme points along the
    fitted line direction.
    """
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    best, best_len = None, 0.0
    for c in contours:
        if cv.contourArea(c) < MIN_LINE_AREA:
            continue
        (w, h) = cv.minAreaRect(c)[1]
        long_side, short_side = max(w, h), max(1.0, min(w, h))
        if long_side / short_side < MIN_ELONGATION:
            continue
        if long_side > best_len:
            best, best_len = c, long_side
    if best is None:
        return None

    pts = best.reshape(-1, 2).astype(np.float32)
    vx, vy, x0, y0 = cv.fitLine(pts, cv.DIST_L2, 0, 0.01, 0.01).flatten()
    direction = np.array([vx, vy])
    origin = np.array([x0, y0])
    proj = (pts - origin) @ direction
    return pts[int(np.argmin(proj))], pts[int(np.argmax(proj))]


def detect(bgr, aoi: AOI | None = None, lab_mode: bool = False) -> Endpoints | None:
    """Full pipeline: red mask -> crop to AOI -> line endpoints (or None)."""
    mask = build_red_mask(bgr, lab_mode=lab_mode)
    mask = crop_to_aoi(mask, aoi)
    return detect_endpoints(mask)


def draw_overlay(bgr, aoi: AOI | None = None, endpoints: Endpoints | None = None):
    """Draw the AOI box (yellow) and the detected line + endpoints (in place)."""
    if aoi is not None:
        cv.rectangle(bgr, (aoi[0], aoi[1]), (aoi[2], aoi[3]), (0, 255, 255), 2)
    if endpoints is not None:
        p1, p2 = endpoints
        a = tuple(np.round(p1).astype(int))
        b = tuple(np.round(p2).astype(int))
        cv.line(bgr, a, b, (0, 255, 0), 2)
        for pt, label in ((a, "P1"), (b, "P2")):
            cv.circle(bgr, pt, 6, (255, 0, 0), -1)
            cv.putText(bgr, label, (pt[0] + 8, pt[1] - 8), cv.FONT_HERSHEY_SIMPLEX,
                       0.55, (255, 0, 0), 2, cv.LINE_AA)
    return bgr
