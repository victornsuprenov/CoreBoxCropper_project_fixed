from __future__ import annotations

import math
from typing import Iterable

import cv2
import numpy as np

from .models import Point, Rect


def as_array(points: Iterable[Point]) -> np.ndarray:
    """Convert point collections safely, including NumPy arrays/scalars.

    OpenCV frequently returns numpy scalar values.  The previous implementation
    called ``list(points)`` unconditionally, which raises
    ``TypeError: numpy.int32 object is not iterable`` if a malformed/scalar
    value reaches the geometry layer.  Normalize the input first and produce a
    useful validation error instead of leaking the NumPy TypeError.
    """
    if points is None:
        raise ValueError("points cannot be None")
    try:
        arr = np.asarray(points, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid point collection: {type(points).__name__}") from exc

    # Accept both [(x, y), ...] and a flat [x1, y1, ...] representation.
    if arr.ndim == 1 and arr.size == 8:
        arr = arr.reshape(4, 2)
    return arr


def order_points(points: Iterable[Point]) -> list[Point]:
    """Return four points in TL, TR, BR, BL order.

    The original heuristic relied on raw sum/difference comparisons, which are
    ambiguous for some convex quadrilaterals and can reverse the top/bottom or
    left/right ordering before a perspective transform.  Use the box extrema
    directly so the result is stable for a real target rectangle regardless of
    input ordering or slight rotation.
    """
    pts = as_array(points)
    if pts.shape != (4, 2):
        raise ValueError("exactly four 2D points are required")

    hull = cv2.convexHull(pts.reshape(-1, 1, 2), clockwise=False).reshape(-1, 2)
    if hull.shape[0] != 4:
        raise ValueError("points must form a convex quadrilateral")
    center = hull.mean(axis=0)
    angles = np.arctan2(hull[:, 1] - center[1], hull[:, 0] - center[0])
    cyclic = hull[np.argsort(angles)]
    start = int(np.argmin(cyclic[:, 0] + cyclic[:, 1]))
    cyclic = np.roll(cyclic, -start, axis=0)
    # In image coordinates the desired TL, TR, BR, BL order is clockwise.
    if cv2.contourArea(cyclic.reshape(-1, 1, 2), oriented=True) < 0:
        cyclic = np.array([cyclic[0], cyclic[3], cyclic[2], cyclic[1]], dtype=np.float32)
    ordered = cyclic.astype(np.float32)
    return [(float(x), float(y)) for x, y in ordered]


def is_valid_quadrilateral(points: Iterable[Point], min_area: float = 1.0) -> bool:
    try:
        pts = as_array(points)
        if pts.shape != (4, 2) or not np.isfinite(pts).all():
            return False
        hull = cv2.convexHull(pts.reshape(-1, 1, 2))
        return hull.shape[0] == 4 and abs(float(cv2.contourArea(hull))) >= min_area
    except (TypeError, ValueError, cv2.error):
        return False


def polygon_area(points: Iterable[Point]) -> float:
    pts = as_array(points)
    if len(pts) < 3:
        return 0.0
    return float(abs(cv2.contourArea(pts.reshape(-1, 1, 2))))


def calculate_angle(a: Point, b: Point, c: Point) -> float:
    """Angle ABC in degrees."""
    ba = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    bc = np.asarray(c, dtype=float) - np.asarray(b, dtype=float)
    denominator = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denominator == 0:
        return 0.0
    cosine = float(np.clip(np.dot(ba, bc) / denominator, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def _direction(line: tuple[Point, Point]) -> np.ndarray:
    vector = np.asarray(line[1], dtype=float) - np.asarray(line[0], dtype=float)
    norm = np.linalg.norm(vector)
    return vector / norm if norm else vector


def is_parallel(line_a: tuple[Point, Point], line_b: tuple[Point, Point], tolerance: float = 8.0) -> bool:
    a = _direction(line_a)
    b = _direction(line_b)
    if not np.any(a) or not np.any(b):
        return False
    angle = math.degrees(math.acos(float(np.clip(abs(np.dot(a, b)), -1.0, 1.0))))
    return angle <= tolerance


def is_perpendicular(
    line_a: tuple[Point, Point], line_b: tuple[Point, Point], tolerance: float = 12.0
) -> bool:
    a = _direction(line_a)
    b = _direction(line_b)
    if not np.any(a) or not np.any(b):
        return False
    angle = math.degrees(math.acos(float(np.clip(abs(np.dot(a, b)), -1.0, 1.0))))
    return abs(angle - 90.0) <= tolerance


def line_intersection(
    line_a: tuple[Point, Point], line_b: tuple[Point, Point]
) -> Point | None:
    (x1, y1), (x2, y2) = line_a
    (x3, y3), (x4, y4) = line_b
    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denominator) < 1e-9:
        return None
    determinant_a = x1 * y2 - y1 * x2
    determinant_b = x3 * y4 - y3 * x4
    x = (determinant_a * (x3 - x4) - (x1 - x2) * determinant_b) / denominator
    y = (determinant_a * (y3 - y4) - (y1 - y2) * determinant_b) / denominator
    return float(x), float(y)


def bbox_from_points(points: Iterable[Point], image_shape: tuple[int, ...] | None = None) -> Rect:
    pts = as_array(points)
    x1, y1 = np.floor(pts.min(axis=0)).astype(int)
    x2, y2 = np.ceil(pts.max(axis=0)).astype(int)
    if image_shape:
        height, width = image_shape[:2]
        x1, x2 = max(0, x1), min(width - 1, x2)
        y1, y2 = max(0, y1), min(height - 1, y2)
    return int(x1), int(y1), int(x2), int(y2)


def expand_roi(
    roi: Rect,
    image_shape: tuple[int, ...],
    margin_percent: float,
    extra_bounds: Iterable[Rect] | None = None,
) -> Rect:
    """Expand an axis-aligned ROI proportionally and include related objects."""
    height, width = image_shape[:2]
    x1, y1, x2, y2 = roi
    rectangles = [roi, *(extra_bounds or [])]
    x1 = min(item[0] for item in rectangles)
    y1 = min(item[1] for item in rectangles)
    x2 = max(item[2] for item in rectangles)
    y2 = max(item[3] for item in rectangles)
    dx = max(1, int((x2 - x1) * margin_percent))
    dy = max(1, int((y2 - y1) * margin_percent))
    return max(0, x1 - dx), max(0, y1 - dy), min(width - 1, x2 + dx), min(height - 1, y2 + dy)


def perspective_distortion(corners: Iterable[Point]) -> float:
    tl, tr, br, bl = order_points(corners)
    widths = [math.dist(tl, tr), math.dist(bl, br)]
    heights = [math.dist(tl, bl), math.dist(tr, br)]
    width_delta = abs(widths[0] - widths[1]) / max(max(widths), 1.0)
    height_delta = abs(heights[0] - heights[1]) / max(max(heights), 1.0)
    return float(max(width_delta, height_delta))


def calculate_perspective(
    corners: Iterable[Point], margin_percent: float = 0.0
) -> tuple[np.ndarray, tuple[int, int]]:
    ordered = np.asarray(order_points(corners), dtype=np.float32)
    tl, tr, br, bl = ordered
    width = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    height = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))
    width = max(width, 2)
    height = max(height, 2)
    if margin_percent < 0:
        raise ValueError("margin_percent cannot be negative")
    center = ordered.mean(axis=0)
    expanded = center + (ordered - center) * (1.0 + 2.0 * margin_percent)
    expanded_tl, expanded_tr, expanded_br, expanded_bl = expanded
    output_width = max(
        2, int(round(max(np.linalg.norm(expanded_tr - expanded_tl), np.linalg.norm(expanded_br - expanded_bl))))
    )
    output_height = max(
        2, int(round(max(np.linalg.norm(expanded_bl - expanded_tl), np.linalg.norm(expanded_br - expanded_tr))))
    )
    destination = np.array(
        [
            [0, 0],
            [output_width, 0],
            [output_width, output_height],
            [0, output_height],
        ],
        dtype=np.float32,
    )
    return cv2.getPerspectiveTransform(expanded, destination), (output_width, output_height)
