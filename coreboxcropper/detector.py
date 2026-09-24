from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import cv2
import numpy as np

from .config import DetectorConfig
from .geometry import bbox_from_points, expand_roi, order_points, perspective_distortion
from .inference.v3_pipeline import V3Pipeline
from .models import DetectionResult, Point
from .ruler_detector import RulerDetector
from .yolo_detector import YoloFirstDetector


@dataclass
class _RailGroup:
    rails: tuple[int, int, int, int]
    y1: int
    y2: int
    spacing: float
    regularity: float
    strength: float
    score: float = 0.0
    center_score: float = 0.0
    width_score: float = 0.0
    attachment_score: float = 0.0
    header_score: float = 0.0
    header_top: int = 0


class BoxDetector:
    """Detect the target three-cell core box and its geological ruler.

    The important change versus the previous detector is that detection of
    the box is no longer dependent on successful colour-scale detection.
    Photographs from the field can have glare, shadows, blur or an occluded
    colour strip.  The wooden rails are much more stable.

    Strategy:
      1. Find a ruler if possible.  The *full ruler bounds*, not the colour
         strip bounds, are used as the attachment anchor.
      2. Find four strong, approximately vertical rails. Four rails define the
         three core cells.
      3. Score all four-rail combinations by spacing, aspect ratio, rail
         strength, position and ruler attachment.
      4. If no ruler is detected, use a geometry/position fallback and reserve
         a configured ruler-width strip immediately to the right of the box.

    This is deliberately a classical OpenCV solution; no ML model is needed
    for the current photo family.
    """

    def __init__(self, config: DetectorConfig | None = None):
        self.config = config or DetectorConfig()
        self.ruler_detector = RulerDetector(self.config)
        self.yolo_detector = YoloFirstDetector(self.config)
        self.v3_pipeline = V3Pipeline(self.config)

    def detect(self, image: np.ndarray) -> DetectionResult:
        """Run YOLO first and preserve legacy coverage on model misses."""
        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            return DetectionResult(False, 0.0, reason="invalid_image")
        if len(image.shape) != 3 or image.shape[2] != 3:
            return DetectionResult(False, 0.0, reason="unsupported_image_shape")

        if self.config.v3_enabled:
            v3_result = self.v3_pipeline.analyze(image)
            if v3_result.success:
                return v3_result

        yolo_result = self.yolo_detector.detect(image)
        if yolo_result is not None:
            if yolo_result.reason == "yolo_no_detection" and self.config.yolo_legacy_fallback:
                legacy = self._detect_legacy(image)
                legacy.candidate_scores["yolo_miss_legacy_fallback"] = 1.0
                return legacy
            return yolo_result
        return self._detect_legacy(image)

    def _detect_legacy(self, image: np.ndarray) -> DetectionResult:
        """Detect the target box independently of the photo orientation.

        The legacy detector works in a normalized geometry where the box rails
        are vertical and the ruler is on the right. Field photographs can be
        rotated by 90/180/270 degrees, so run the same detector in four
        orientations and transform the best result back to the original
        coordinate system. This is deliberately done at the detection stage;
        the saved crop always comes from the original image and is never
        rotated.
        """
        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            return DetectionResult(False, 0.0, reason="invalid_image")
        if len(image.shape) != 3 or image.shape[2] != 3:
            return DetectionResult(False, 0.0, reason="unsupported_image_shape")

        # Production path: YOLO finds the target cheaply, then all geometry
        # refinement is restricted to that candidate ROI. The legacy detector
        # below remains available when optional ML dependencies/weights are not
        # installed, preserving offline compatibility and old tests.
        variants = [
            (0, image),
            (90, cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)),
            (180, cv2.rotate(image, cv2.ROTATE_180)),
            (270, cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)),
        ]
        results: list[DetectionResult] = []
        for angle, variant in variants:
            result = self._detect_single(variant)
            restored = self._restore_orientation_result(result, image.shape[:2], angle)
            restored.candidate_scores["orientation_degrees"] = float(angle)
            results.append(restored)

        def rank(result: DetectionResult) -> tuple[float, ...]:
            # A confirmed box+ruler is preferable to a geometry-only guess.
            # Confidence remains the main criterion, while the small ruler
            # bonus helps select the orientation in ambiguous scenes.
            return (
                1.0 if result.success else 0.0,
                float(result.confidence),
                1.0 if result.ruler else 0.0,
                float(result.candidate_scores.get("ruler_attachment_score", 0.0)),
                float(result.candidate_scores.get("header_signature_score", 0.0)),
            )

        return max(results, key=rank)

    def _restore_orientation_result(
        self, result: DetectionResult, original_shape: tuple[int, int], angle: int
    ) -> DetectionResult:
        """Map a detection made on a rotated image back to original pixels."""
        if angle not in (0, 90, 180, 270):
            return result
        if angle == 0:
            return result

        height, width = original_shape

        def point_back(point: Point) -> Point:
            x, y = float(point[0]), float(point[1])
            if angle == 90:
                return (float(width - 1) - y, x)
            if angle == 180:
                return (float(width - 1) - x, float(height - 1) - y)
            return (y, float(height - 1) - x)

        def rect_back(rect: tuple[int, int, int, int] | None):
            if not rect:
                return None
            x1, y1, x2, y2 = map(float, rect)
            pts = [
                point_back((x1, y1)), point_back((x2, y1)),
                point_back((x2, y2)), point_back((x1, y2)),
            ]
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            return (
                max(0, int(round(min(xs)))),
                max(0, int(round(min(ys)))),
                min(width - 1, int(round(max(xs)))),
                min(height - 1, int(round(max(ys)))),
            )

        if result.corners:
            result.corners = order_points([point_back(p) for p in result.corners])
        result.bbox = rect_back(result.bbox)
        result.ruler = rect_back(result.ruler)
        result.final_roi = rect_back(result.final_roi)

        restored_debug: dict[str, np.ndarray] = {}
        for name, frame in result.debug_images.items():
            if angle == 90:
                restored_debug[name] = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            elif angle == 180:
                restored_debug[name] = cv2.rotate(frame, cv2.ROTATE_180)
            else:
                restored_debug[name] = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        result.debug_images = restored_debug
        return result

    def _detect_single(self, image: np.ndarray) -> DetectionResult:
        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            return DetectionResult(False, 0.0, reason="invalid_image")
        if len(image.shape) != 3 or image.shape[2] != 3:
            return DetectionResult(False, 0.0, reason="unsupported_image_shape")

        height, width = image.shape[:2]
        if min(height, width) < 240:
            return DetectionResult(False, 0.0, reason="image_too_small")

        scale = min(1.0, self.config.max_analysis_dimension / max(height, width))
        analysis = (
            cv2.resize(
                image,
                (max(1, int(width * scale)), max(1, int(height * scale))),
                interpolation=cv2.INTER_AREA,
            )
            if scale < 1.0
            else image.copy()
        )

        ruler = self.ruler_detector.detect_global(analysis)
        group = None
        mode = "no_ruler"

        if ruler.found and ruler.bounds:
            # IMPORTANT: use the left edge of the whole ruler, not the left
            # edge of the colour scale. The colour scale is near the bottom and
            # is often shifted inside the wooden ruler.
            group = self._find_best_three_cell_group(
                analysis,
                ruler_left=ruler.bounds[0],
                ruler_right=ruler.bounds[2],
            )
            mode = "with_ruler"

        if group is None:
            # The box can still be detected when the ruler signature is weak.
            # Prefer a central three-cell box over a generic rectangle.
            group = self._find_best_three_cell_group(analysis)
            mode = "geometry_fallback"

        if group is None:
            fallback = self._fallback_rectangle(analysis)
            if fallback is None:
                return DetectionResult(False, 0.0, reason="box_not_found")
            corners, score = fallback
            box = bbox_from_points(corners, analysis.shape)
            final = expand_roi(box, analysis.shape, self.config.crop_margin_percent)
            if scale != 1.0:
                corners = [(x / scale, y / scale) for x, y in corners]
                box = tuple(int(round(v / scale)) for v in box)
                final = tuple(int(round(v / scale)) for v in final)
            return DetectionResult(
                success=False,
                confidence=min(score, 0.45),
                corners=order_points(corners),
                bbox=box,
                final_roi=final,
                reason="box_not_confident",
            )

        return self._build_result(image, analysis, scale, group, ruler, mode)

    def _find_best_three_cell_group(
        self,
        image: np.ndarray,
        ruler_left: int | None = None,
        ruler_right: int | None = None,
    ) -> _RailGroup | None:
        """Find four rails representing exactly three cells.

        Two independent signals are combined:
        - a smoothed vertical-edge profile;
        - Hough vertical line evidence.

        The profile catches broad wooden rails even when Hough fragments them;
        Hough evidence helps reject short texture edges from the core.
        """
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)

        # Work on the useful image height. The target box normally occupies
        # most of the height, but we do not hard-code its exact top/bottom.
        y0, y1 = int(h * 0.03), int(h * 0.97)
        roi = blurred[y0:y1]
        grad = np.abs(cv2.Sobel(roi, cv2.CV_32F, 1, 0, ksize=3)).mean(axis=0)
        grad = np.convolve(grad, np.ones(21, dtype=np.float32) / 21.0, mode="same")
        grad_norm = grad / max(float(np.percentile(grad, 90)), 1.0)

        # Restrict search when the ruler is known. The target is immediately
        # to its left in this photo family.
        search_left = int(w * self.config.target_search_left_ratio)
        search_right = int(w * self.config.target_search_right_ratio)
        if ruler_left is not None:
            search_left = max(search_left, int(ruler_left - w * self.config.ruler_box_search_width_ratio))
            # The wooden end rail can overlap the ruler by a few pixels due
            # to perspective, blur and the ruler's own border. Do not exclude
            # that rail from the candidate pool.
            search_right = min(search_right, int(ruler_left + w * self.config.ruler_box_overlap_search_ratio))
            if search_right <= search_left:
                search_left = int(w * 0.08)
                search_right = max(search_left + 50, int(ruler_left - 2))

        search_left = max(0, min(search_left, w - 2))
        search_right = max(search_left + 2, min(search_right, w - 1))

        profile = grad_norm[search_left:search_right]
        peaks = self._profile_peaks(
            profile,
            min_distance=max(25, int(w * self.config.rail_min_distance_ratio)),
            percentile=self.config.rail_peak_percentile,
        )
        peaks = [search_left + p for p in peaks]

        # Add Hough-derived vertical line centres.
        hough_x = self._hough_vertical_xs(blurred, search_left, search_right)
        # Profile peaks are stable global rail hypotheses. Hough produces many
        # short/duplicated texture lines inside the core, so never merge the
        # two lists together (doing so can drag a real rail toward the median
        # of dozens of Hough detections). Add Hough positions only when they
        # fill a gap not already covered by a profile peak.
        candidates = list(peaks)
        hough_tolerance = max(22, int(w * 0.010))
        for hx in sorted(hough_x):
            if all(abs(hx - px) > hough_tolerance for px in candidates):
                candidates.append(hx)
        candidates = sorted(candidates)
        if len(candidates) < 4:
            return None

        groups: list[_RailGroup] = []
        min_spacing = self.config.min_cell_spacing_ratio * w
        max_spacing = self.config.max_cell_spacing_ratio * w

        for combo in itertools.combinations(candidates, 4):
            distances = np.diff(combo).astype(float)
            spacing = float(np.mean(distances))
            if not min_spacing <= spacing <= max_spacing:
                continue

            regularity = float(np.std(distances) / max(spacing, 1.0))
            if regularity > self.config.max_spacing_regularity:
                continue

            width_ratio = (combo[-1] - combo[0]) / max(w, 1)
            if not self.config.min_box_width_ratio <= width_ratio <= self.config.max_box_width_ratio:
                continue

            y_top, y_bottom = self._box_vertical_bounds(gray, combo[0], combo[-1])
            # Field photos often contain identification cards/handwritten
            # signatures immediately above the visible core area.  Treat this
            # as a weak geometric signature of the target box: dark ink on a
            # light label is much more useful than OCR here, and it lets us
            # extend the crop upward without learning any particular text.
            header_score, header_top = self._header_signature(
                gray, combo[0], combo[-1], y_top, y_bottom
            )
            y_top = min(y_top, header_top)
            box_height = max(y_bottom - y_top, 1)
            aspect = box_height / max(combo[-1] - combo[0], 1)
            aspect_error = abs(math.log(max(aspect, 1e-4) / self.config.expected_box_aspect_ratio))
            aspect_score = float(np.exp(-aspect_error / max(self.config.box_aspect_tolerance, 1e-3)))

            # Average edge strength at all four rails.
            strengths = [float(grad_norm[max(0, min(w - 1, x))]) for x in combo]
            rail_strength = float(np.clip(np.mean(strengths) / self.config.rail_strength_reference, 0.0, 1.0))

            regularity_score = float(
                np.clip(1.0 - regularity / max(self.config.max_spacing_regularity, 1e-3), 0.0, 1.0)
            )
            width_score = self._width_score(width_ratio)

            center_ratio = ((combo[0] + combo[-1]) * 0.5) / max(w, 1)
            center_score = self._target_center_score(center_ratio)

            attachment_score = 0.0
            if ruler_left is not None:
                gap_ratio = (ruler_left - combo[-1]) / max(w, 1)
                attachment_score = self._attachment_score(gap_ratio)
                # If the final rail is actually to the right of the ruler, it
                # cannot be the target box.
                if gap_ratio < -self.config.ruler_box_overlap_tolerance:
                    continue

            hough_score = self._hough_support_score(combo, hough_x, w)

            if ruler_left is not None:
                score = (
                    0.29 * attachment_score
                    + 0.21 * regularity_score
                    + 0.17 * aspect_score
                    + 0.11 * rail_strength
                    + 0.09 * width_score
                    + 0.06 * hough_score
                    + 0.02 * center_score
                    + 0.045 * header_score
                )
            else:
                score = (
                    0.23 * center_score
                    + 0.20 * regularity_score
                    + 0.17 * aspect_score
                    + 0.13 * rail_strength
                    + 0.10 * width_score
                    + 0.07 * hough_score
                    + 0.10 * header_score
                )

            groups.append(
                _RailGroup(
                    rails=combo,
                    y1=y_top,
                    y2=y_bottom,
                    spacing=spacing,
                    regularity=regularity,
                    strength=rail_strength,
                    score=float(score),
                    center_score=center_score,
                    width_score=width_score,
                    attachment_score=attachment_score,
                    header_score=header_score,
                    header_top=header_top,
                )
            )

        if not groups:
            return None

        groups.sort(key=lambda g: g.score, reverse=True)
        best = groups[0]

        threshold = (
            self.config.ruler_group_confidence_threshold
            if ruler_left is not None
            else self.config.geometry_group_confidence_threshold
        )
        return best if best.score >= threshold else None

    def _hough_vertical_xs(self, gray: np.ndarray, x1: int, x2: int) -> list[int]:
        h, _ = gray.shape
        crop = gray[:, x1:x2]
        edges = cv2.Canny(crop, self.config.canny_threshold_1, self.config.canny_threshold_2)
        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180,
            threshold=self.config.hough_threshold,
            minLineLength=max(80, int(h * self.config.hough_min_vertical_length_ratio)),
            maxLineGap=max(20, int(h * self.config.hough_max_vertical_gap_ratio)),
        )
        if lines is None:
            return []

        xs: list[int] = []
        max_dx = max(8, int((x2 - x1) * 0.012))
        min_vertical_length = int(h * self.config.hough_min_vertical_length_ratio)

        # OpenCV normally returns HoughLinesP as (N, 1, 4), but some
        # OpenCV builds/configurations can return (N, 4).  Iterating over
        # lines[:, 0] in the latter case yields a single numpy.int32,
        # causing: "numpy.int32 object is not iterable".  Normalize the
        # result to one (x1, y1, x2, y2) row per line.
        line_array = np.asarray(lines).reshape(-1, 4)
        for lx1, ly1, lx2, ly2 in line_array:
            lx1, ly1, lx2, ly2 = (int(lx1), int(ly1), int(lx2), int(ly2))
            if abs(lx2 - lx1) <= max_dx and abs(ly2 - ly1) >= min_vertical_length:
                xs.append(x1 + int(round((lx1 + lx2) / 2)))
        return xs

    @staticmethod
    def _merge_positions(values: list[int], max_distance: int) -> list[int]:
        if not values:
            return []
        values = sorted(values)
        clusters: list[list[int]] = [[values[0]]]
        for value in values[1:]:
            if value - clusters[-1][-1] <= max_distance:
                clusters[-1].append(value)
            else:
                clusters.append([value])
        return [int(round(float(np.median(cluster)))) for cluster in clusters]

    @staticmethod
    def _hough_support_score(combo: tuple[int, int, int, int], hough_x: list[int], width: int) -> float:
        if not hough_x:
            return 0.35
        tolerance = max(20, int(width * 0.012))
        matches = sum(any(abs(x - hx) <= tolerance for hx in hough_x) for x in combo)
        return matches / 4.0

    def _target_center_score(self, center_ratio: float) -> float:
        # In the current photo family the target box is close to the centre.
        # This is a soft prior, not a hard coordinate.
        sigma = max(self.config.target_center_sigma, 0.03)
        return float(np.exp(-((center_ratio - self.config.target_center_ratio) / sigma) ** 2))

    def _width_score(self, width_ratio: float) -> float:
        lo, hi = self.config.min_box_width_ratio, self.config.max_box_width_ratio
        preferred = self.config.preferred_box_width_ratio
        half = max((hi - lo) / 2.0, 1e-4)
        return float(np.clip(1.0 - abs(width_ratio - preferred) / half, 0.0, 1.0))

    @staticmethod
    def _attachment_score(gap_ratio: float) -> float:
        # Ideal: wooden right rail touches or nearly touches the ruler.
        if -0.005 <= gap_ratio <= 0.015:
            return 1.0
        if gap_ratio < -0.005:
            return max(0.0, 1.0 - abs(gap_ratio + 0.005) / 0.03)
        return max(0.0, 1.0 - (gap_ratio - 0.015) / 0.06)

    def _header_signature(
        self, gray: np.ndarray, x1: int, x2: int, y_top: int, y_bottom: int
    ) -> tuple[float, int]:
        """Find a dark-on-light identification label near the box top.

        OCR is intentionally avoided.  A label is treated as a small light
        patch containing dark ink.  The signal is only a secondary feature: a
        ruler/rail match remains more important than the label.
        """
        h, w = gray.shape
        height = max(y_bottom - y_top, 1)
        # Search only a narrow header band.  A hard image-relative cap is
        # important on photos where the box itself starts close to the frame.
        upward_window = min(int(height * 0.09), max(36, int(h * 0.035)))
        search_top = max(0, y_top - upward_window)
        search_bottom = min(h - 1, y_top + int(height * 0.10))
        if search_bottom - search_top < 10 or x2 - x1 < 30:
            return 0.0, y_top

        cell_width = (x2 - x1) / 3.0
        detections: list[tuple[float, int]] = []
        for cell in range(3):
            xa = int(round(x1 + cell * cell_width + cell_width * 0.10))
            xb = int(round(x1 + (cell + 1) * cell_width - cell_width * 0.10))
            if xb - xa < 18:
                continue
            band = gray[search_top : search_bottom + 1, xa:xb]
            if band.size == 0:
                continue

            # Bright mask approximates paper/card labels. Close small holes,
            # but do not merge the large bright core pieces into one label.
            bright = (band >= 160).astype(np.uint8) * 255
            bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
            n, labels, stats, _ = cv2.connectedComponentsWithStats(bright)
            candidates: list[tuple[float, int]] = []
            for i in range(1, n):
                bx, by, bw, bh, area = map(int, stats[i])
                cell_area = max(1, band.shape[1] * band.shape[0])
                if area < max(80, int(cell_area * 0.004)):
                    continue
                if area > int(cell_area * 0.30):
                    continue
                if bw < max(12, int((xb - xa) * 0.16)):
                    continue
                if bh < 8 or bh > int(height * 0.25):
                    continue
                aspect = bw / max(bh, 1)
                if not 0.18 <= aspect <= 4.5:
                    continue
                component = band[by : by + bh, bx : bx + bw]
                dark_ratio = float(np.mean(component <= 105))
                if dark_ratio < 0.025:
                    continue
                # Ink must occupy a modest fraction of a light patch.
                light_ratio = float(np.mean(component >= 160))
                if light_ratio < 0.45:
                    continue
                ink_score = float(np.clip(dark_ratio / 0.16, 0.0, 1.0))
                light_score = float(np.clip((light_ratio - 0.45) / 0.40, 0.0, 1.0))
                size_score = float(np.clip(area / max(cell_area * 0.06, 1.0), 0.0, 1.0))
                score = 0.55 * ink_score + 0.30 * light_score + 0.15 * size_score
                candidates.append((score, search_top + by))
            if candidates:
                detections.append(max(candidates, key=lambda item: item[0]))

        strong = [item for item in detections if item[0] >= 0.38]
        if not strong:
            return 0.0, y_top

        best = max(item[0] for item in strong)
        # Multiple cells carrying similar labels is a useful tie-breaker.
        score = min(1.0, best + (0.12 if len(strong) >= 2 else 0.0) + (0.08 if len(strong) >= 3 else 0.0))
        label_top = min(item[1] for item in strong)
        pad = max(5, int(height * 0.025))
        header_top = max(0, min(y_top, label_top - pad))
        # Never move the box boundary upward by more than the inspected window.
        header_top = max(y_top - upward_window, header_top)
        return float(np.clip(score, 0.0, 1.0)), int(header_top)

    def _box_vertical_bounds(self, gray: np.ndarray, x1: int, x2: int) -> tuple[int, int]:
        h, w = gray.shape
        xa = max(0, min(x1, w - 1))
        xb = max(xa + 1, min(x2, w))
        region = cv2.GaussianBlur(gray[:, xa:xb], (9, 9), 0)
        profile = np.abs(cv2.Sobel(region, cv2.CV_32F, 0, 1, ksize=3)).mean(axis=1)
        profile = np.convolve(profile, np.ones(25, dtype=np.float32) / 25.0, mode="same")

        # The outer frame is normally near the top/bottom. Search generous
        # bands, but prefer strong horizontal frame transitions.
        top_lo, top_hi = int(h * 0.015), int(h * 0.28)
        bot_lo, bot_hi = int(h * 0.72), int(h * 0.985)
        top = top_lo + int(np.argmax(profile[top_lo:top_hi]))
        bottom = bot_lo + int(np.argmax(profile[bot_lo:bot_hi]))

        if bottom - top < int(h * 0.60):
            top = int(h * 0.04)
            bottom = int(h * 0.96)
        return max(0, top), min(h - 1, bottom)

    def _build_result(
        self,
        original: np.ndarray,
        analysis: np.ndarray,
        scale: float,
        group: _RailGroup,
        ruler,
        mode: str,
    ) -> DetectionResult:
        a = group.rails
        corners_analysis: list[Point] = [
            (float(a[0]), float(group.y1)),
            (float(a[3]), float(group.y1)),
            (float(a[3]), float(group.y2)),
            (float(a[0]), float(group.y2)),
        ]
        box_analysis = bbox_from_points(corners_analysis, analysis.shape)

        ruler_found = bool(ruler.found and ruler.bounds)
        ruler_bounds_analysis: tuple[int, int, int, int]
        ruler_quality = 0.0
        color_score = 0.0
        tick_score = 0.0

        if ruler_found:
            detected_ruler = ruler.bounds
            assert detected_ruler is not None
            ruler_bounds_analysis = (
                detected_ruler[0],
                box_analysis[1],
                detected_ruler[2],
                box_analysis[3],
            )
            ruler_quality = float(ruler.confidence)
            color_score = float(ruler.color_scale_confidence)
            tick_score = float(ruler.tick_score)
        else:
            # If the signature failed, still reserve the physical ruler strip
            # immediately to the right of the target box. This is intentionally
            # bounded so that a neighbouring box is not swallowed.
            x1, y1, x2, y2 = box_analysis
            expected_width = max(40, int(analysis.shape[1] * self.config.expected_ruler_width_ratio))
            ruler_bounds_analysis = (
                x2,
                y1,
                min(analysis.shape[1] - 1, x2 + expected_width),
                y2,
            )

        final_analysis = expand_roi(
            (
                min(box_analysis[0], ruler_bounds_analysis[0]),
                min(box_analysis[1], ruler_bounds_analysis[1]),
                max(box_analysis[2], ruler_bounds_analysis[2]),
                max(box_analysis[3], ruler_bounds_analysis[3]),
            ),
            analysis.shape,
            self.config.crop_margin_percent,
        )

        if ruler_found:
            confidence = float(
                np.clip(
                    0.55 * group.score + 0.25 * ruler_quality + 0.20 * color_score,
                    0.0,
                    1.0,
                )
            )
            success = (
                group.score >= self.config.ruler_group_confidence_threshold
                and confidence >= self.config.confidence_threshold
            )
            reason = "three_cell_box_with_ruler" if success else "box_confidence_below_threshold"
        else:
            confidence = float(np.clip(0.90 * group.score, 0.0, 1.0))
            success = (
                group.score >= self.config.geometry_group_confidence_threshold
                and confidence >= self.config.geometry_fallback_success_threshold
            )
            reason = "three_cell_box_ruler_signature_weak" if success else "box_confidence_below_threshold"

        if scale != 1.0:
            corners = [(x / scale, y / scale) for x, y in corners_analysis]
            box = tuple(int(round(v / scale)) for v in box_analysis)
            ruler_scaled = tuple(int(round(v / scale)) for v in ruler_bounds_analysis)
            final_roi = tuple(int(round(v / scale)) for v in final_analysis)
        else:
            corners = corners_analysis
            box = box_analysis
            ruler_scaled = ruler_bounds_analysis
            final_roi = final_analysis

        debug = self._make_debug(analysis, group, ruler, ruler_bounds_analysis)
        if scale != 1.0:
            debug = {
                name: cv2.resize(frame, (original.shape[1], original.shape[0]), interpolation=cv2.INTER_NEAREST)
                for name, frame in debug.items()
            }

        box_aspect = (group.y2 - group.y1) / max(group.rails[-1] - group.rails[0], 1)
        scores = {
            "three_cell_score": 1.0,
            "ruler_attachment_score": group.attachment_score,
            "spacing_regularity_score": float(
                np.clip(1.0 - group.regularity / max(self.config.max_spacing_regularity, 1e-3), 0.0, 1.0)
            ),
            "rail_strength_score": group.strength,
            "box_aspect_ratio": box_aspect,
            "box_aspect_score": float(
                np.exp(
                    -abs(math.log(max(box_aspect, 1e-4) / self.config.expected_box_aspect_ratio))
                    / max(self.config.box_aspect_tolerance, 1e-3)
                )
            ),
            "target_center_score": group.center_score,
            "box_width_score": group.width_score,
            "header_signature_score": group.header_score,
            "header_top_offset": float(group.y1 - group.header_top),
            "ruler_signature_score": ruler_quality,
            "color_scale_score": color_score,
            "tick_score": tick_score,
        }

        return DetectionResult(
            success=success,
            confidence=confidence,
            corners=order_points(corners),
            bbox=box,
            ruler=ruler_scaled,
            ruler_confidence=ruler_quality,
            final_roi=final_roi,
            reason=reason,
            candidate_scores=scores,
            perspective_recommended=perspective_distortion(corners) >= self.config.perspective_threshold,
            debug_images=debug,
        )

    def _make_debug(
        self,
        image: np.ndarray,
        group: _RailGroup,
        ruler,
        fallback_ruler: tuple[int, int, int, int],
    ) -> dict[str, np.ndarray]:
        frame = image.copy()
        pts = np.asarray(
            [
                [group.rails[0], group.y1],
                [group.rails[-1], group.y1],
                [group.rails[-1], group.y2],
                [group.rails[0], group.y2],
            ],
            dtype=np.int32,
        )
        cv2.polylines(frame, [pts], True, (0, 255, 0), 6)
        for x in group.rails:
            cv2.line(frame, (x, group.y1), (x, group.y2), (255, 0, 255), 4)

        if ruler.found and ruler.bounds:
            x1, y1, x2, y2 = ruler.bounds
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 100, 0), 6)
        else:
            x1, y1, x2, y2 = fallback_ruler
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 100, 0), 5)

        if ruler.color_scale:
            x1, y1, x2, y2 = ruler.color_scale
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 5)

        return {"detected_box": frame}

    @staticmethod
    def _profile_peaks(profile: np.ndarray, min_distance: int, percentile: float = 55) -> list[int]:
        if profile.size < 3:
            return []
        threshold = float(np.percentile(profile, percentile))
        local = [
            i
            for i in range(1, len(profile) - 1)
            if profile[i] >= threshold
            and profile[i] >= profile[i - 1]
            and profile[i] >= profile[i + 1]
        ]
        selected: list[int] = []
        for i in sorted(local, key=lambda idx: float(profile[idx]), reverse=True):
            if all(abs(i - j) >= min_distance for j in selected):
                selected.append(i)
        return sorted(selected)

    @staticmethod
    def _fallback_rectangle(image: np.ndarray) -> tuple[list[Point], float] | None:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h, w = image.shape[:2]
        best = None
        for c in contours:
            area = cv2.contourArea(c)
            if area < 0.08 * w * h:
                continue
            rect = cv2.minAreaRect(c)
            pts = cv2.boxPoints(rect).tolist()
            x = sorted(p[0] for p in pts)
            y = sorted(p[1] for p in pts)
            aspect = max(
                (x[-1] - x[0]) / max(y[-1] - y[0], 1),
                (y[-1] - y[0]) / max(x[-1] - x[0], 1),
            )
            if 1.2 <= aspect <= 8.5:
                score = min(0.8, area / (w * h))
                if best is None or score > best[1]:
                    best = (order_points(pts), score)
        return best
