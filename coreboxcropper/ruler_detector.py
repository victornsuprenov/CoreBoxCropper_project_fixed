from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from .config import DetectorConfig
from .models import Point, Rect, RulerDetection


@dataclass
class _ColorScale:
    bounds: Rect
    confidence: float


class RulerDetector:
    """Detect the distinctive geological ruler before choosing a box.

    The ruler is a much stronger global anchor than a generic rectangle: it
    contains a compact multicolour calibration strip, regular 10 cm marks and
    white information labels. Once the ruler is located, the box is searched
    immediately beside it instead of anywhere in the photograph.
    """

    def __init__(self, config: DetectorConfig):
        self.config = config

    def detect_global(self, image: np.ndarray) -> RulerDetection:
        if image is None or image.size == 0:
            return RulerDetection(reason="invalid_image")
        scale = self._find_color_scale(image)
        if scale is None:
            return RulerDetection(reason="color_scale_not_found")

        bounds = self._estimate_vertical_ruler(image, scale.bounds)
        if bounds is None:
            bounds = self._estimate_horizontal_ruler(image, scale.bounds)
        if bounds is None:
            return RulerDetection(
                reason="ruler_edges_not_found",
                color_scale=scale.bounds,
                color_scale_confidence=scale.confidence,
            )

        crop = image[bounds[1] : bounds[3] + 1, bounds[0] : bounds[2] + 1]
        tick_score = self._tick_score(crop, vertical=True)
        if crop.shape[1] > crop.shape[0]:
            tick_score = max(tick_score, self._tick_score(crop, vertical=False))
        confidence = float(np.clip(0.60 * scale.confidence + 0.40 * tick_score, 0.0, 1.0))
        return RulerDetection(
            found=True,
            bounds=bounds,
            confidence=confidence,
            reason="ruler_detected",
            color_scale=scale.bounds,
            color_scale_confidence=scale.confidence,
            tick_score=tick_score,
            label_region=self._estimate_label_region(bounds, scale.bounds, image.shape),
        )

    def detect(self, image: np.ndarray, box_corners: list[Point]) -> RulerDetection:
        """Backward-compatible entry point; global detection is now preferred."""
        return self.detect_global(image)

    def _find_color_scale(self, image: np.ndarray) -> _ColorScale | None:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, w = image.shape[:2]
        mask = (
            (hsv[:, :, 1] >= max(110, self.config.ruler_color_min_saturation))
            & (hsv[:, :, 2] >= 55)
            & (hsv[:, :, 2] <= 255)
        ).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        candidates: list[tuple[float, _ColorScale]] = []
        for i in range(1, n):
            x, y, cw, ch, area = map(int, stats[i])
            if area < 500 or cw < 15 or ch < 25:
                continue
            if cw > max(130, int(w * 0.05)) or ch > max(500, int(h * 0.15)):
                continue
            hue = hsv[:, :, 0][labels == i]
            if hue.size < 100:
                continue
            hue_std = float(np.std(hue))
            if hue_std < 24:
                continue
            # The calibration strip is normally in the lower part of the ruler.
            lower_score = np.clip((y / max(h, 1) - 0.55) / 0.40, 0.0, 1.0)
            area_score = np.clip(area / 12000.0, 0.0, 1.0)
            diversity_score = np.clip(hue_std / 60.0, 0.0, 1.0)
            confidence = float(0.50 * diversity_score + 0.30 * area_score + 0.20 * lower_score)
            candidates.append((confidence, _ColorScale((x, y, x + cw, y + ch), confidence)))
        if not candidates:
            return None
        return max(candidates, key=lambda item: item[0])[1]

    def _estimate_vertical_ruler(self, image: np.ndarray, scale: Rect) -> Rect | None:
        x, y, x2, y2 = scale
        h, w = image.shape[:2]
        center = (x + x2) / 2.0
        left = max(0, int(center - 0.12 * w))
        right = min(w - 1, int(center + 0.12 * w))
        if right - left < 80:
            return None
        y1 = int(h * 0.04)
        yb = int(h * 0.96)
        gray = cv2.cvtColor(image[y1 : yb + 1, left : right + 1], cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)
        grad = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)).mean(axis=0)
        grad = np.convolve(grad, np.ones(15) / 15, mode="same")
        peaks = self._profile_peaks(grad, min_distance=max(25, int(w * 0.018)))
        global_peaks = [left + p for p in peaks if left + p < w]
        enclosing = [
            (a, b)
            for i, a in enumerate(global_peaks)
            for b in global_peaks[i + 1 :]
            if a < x < b and 0.04 * w <= b - a <= 0.12 * w
        ]
        if not enclosing:
            return None
        a, b = min(enclosing, key=lambda pair: abs(((pair[0] + pair[1]) / 2) - (center - 0.03 * w)))
        return (a, 0, b, h - 1)

    def _estimate_horizontal_ruler(self, image: np.ndarray, scale: Rect) -> Rect | None:
        x, y, x2, y2 = scale
        h, w = image.shape[:2]
        center = (y + y2) / 2.0
        top = max(0, int(center - 0.12 * h))
        bottom = min(h - 1, int(center + 0.12 * h))
        gray = cv2.cvtColor(image[top : bottom + 1], cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)
        grad = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)).mean(axis=1)
        grad = np.convolve(grad, np.ones(15) / 15, mode="same")
        peaks = self._profile_peaks(grad, min_distance=max(25, int(h * 0.018)))
        global_peaks = [top + p for p in peaks]
        enclosing = [
            (a, b)
            for i, a in enumerate(global_peaks)
            for b in global_peaks[i + 1 :]
            if a < y < b and 0.04 * h <= b - a <= 0.12 * h
        ]
        if not enclosing:
            return None
        a, b = min(enclosing, key=lambda pair: abs(((pair[0] + pair[1]) / 2) - (center - 0.03 * h)))
        return (0, a, w - 1, b)

    @staticmethod
    def _profile_peaks(profile: np.ndarray, min_distance: int) -> list[int]:
        if profile.size < 3:
            return []
        threshold = float(np.percentile(profile, 62))
        local = [
            i
            for i in range(1, len(profile) - 1)
            if profile[i] >= threshold and profile[i] >= profile[i - 1] and profile[i] >= profile[i + 1]
        ]
        selected: list[int] = []
        for i in sorted(local, key=lambda idx: float(profile[idx]), reverse=True):
            if all(abs(i - j) >= min_distance for j in selected):
                selected.append(i)
        return sorted(selected)

    def _tick_score(self, crop: np.ndarray, vertical: bool = True) -> float:
        if crop.size == 0:
            return 0.0
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        if vertical:
            signal = np.mean(np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)), axis=1)
        else:
            signal = np.mean(np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)), axis=0)
        signal = np.maximum(signal - np.median(signal), 0)
        if signal.max() <= 1e-6:
            return 0.0
        smooth = np.convolve(signal, np.ones(9) / 9, mode="same")
        peaks = self._profile_peaks(smooth, max(8, len(smooth) // 28))
        if len(peaks) < 4:
            return 0.0
        gaps = np.diff(peaks).astype(float)
        regularity = float(np.mean(np.exp(-np.abs(gaps - np.median(gaps)) / max(np.median(gaps), 1.0))))
        return float(np.clip(regularity * min(1.0, len(peaks) / 8.0), 0.0, 1.0))


    # Compatibility helpers retained for calibration/unit tests and for older
    # integrations that used the previous detector API.
    def _color_scale_score(self, region: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> float:
        strip = region[max(0, y1):min(region.shape[0], y2 + 1), max(0, x1):min(region.shape[1], x2 + 1)]
        if strip.size == 0:
            return 0.0
        hsv = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV)
        colorful = (hsv[:, :, 1] >= self.config.ruler_color_min_saturation) & (hsv[:, :, 2] >= 45)
        ratio = float(np.mean(colorful))
        hue = hsv[:, :, 0][colorful]
        diversity = float(np.clip(np.std(hue) / 60.0, 0.0, 1.0)) if hue.size else 0.0
        return float(np.clip(ratio / 0.16, 0.0, 1.0) * 0.7 + diversity * 0.3)

    def _ten_cm_tick_score(self, region: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> float:
        strip = region[max(0, y1):min(region.shape[0], y2 + 1), max(0, x1):min(region.shape[1], x2 + 1)]
        if strip.size == 0 or strip.shape[0] < 80:
            return 0.0
        return self._tick_score(strip, vertical=strip.shape[0] >= strip.shape[1])

    @staticmethod
    def _estimate_label_region(bounds: Rect, scale: Rect, image_shape: tuple[int, ...]) -> Rect | None:
        x1, y1, x2, y2 = bounds
        sx1, sy1, sx2, sy2 = scale
        if y2 - y1 > x2 - x1:
            # White labels sit above the colour scale on the vertical ruler.
            return (max(x1, sx1 - (sx2 - sx1)), max(y1, sy1 - int((y2 - y1) * 0.25)), min(x2, sx2 + (sx2 - sx1)), sy1)
        return (max(x1, sx1 - int((x2 - x1) * 0.25)), max(y1, sy1 - (sy2 - sy1)), sx1, min(y2, sy2 + (sy2 - sy1)))
