from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .config import DetectorConfig
from .geometry import bbox_from_points, expand_roi, order_points, perspective_distortion
from .models import DetectionResult, Point
from .ruler_detector import RulerDetector


@dataclass
class _Candidate:
    box: tuple[float, float, float, float]
    confidence: float


class YoloFirstDetector:
    """Fast YOLO candidate selection followed by local geometric refinement.

    The model finds the target region. OpenCV is used only inside the selected
    candidate ROI, so this path does not repeat the old full-image ruler scan.
    """

    def __init__(self, config: DetectorConfig):
        self.config = config
        self._models: dict[str, object] = {}
        self._disabled = False

    def detect(self, image: np.ndarray) -> DetectionResult | None:
        if self._disabled or not self.config.yolo_enabled:
            return None
        primary, fallback = self._weight_paths()
        candidates = self._predict(image, primary, self.config.yolo_imgsz)
        if self._disabled:
            return None
        model_name = primary
        if not candidates and fallback != primary:
            candidates = self._predict(image, fallback, self.config.yolo_fallback_imgsz)
            if self._disabled:
                return None
            model_name = fallback
        if not candidates:
            # None means the optional ML path is unavailable. An explicit
            # failed DetectionResult prevents the legacy global scan when YOLO
            # is installed and weights are present but found no object.
            if primary.exists() or fallback.exists():
                return DetectionResult(
                    False, 0.0, reason="yolo_no_detection",
                    candidate_scores={"orientation_degrees": 0.0},
                )
            return None

        candidate, selection_score = self._select_candidate(candidates, image)
        return self._refine(image, candidate, selection_score, model_name)

    def _weight_paths(self) -> tuple[Path, Path]:
        project = Path(__file__).resolve().parents[1]
        primary = Path(self.config.yolo_primary_weights) if self.config.yolo_primary_weights else (
            project / "runs" / "detect" / "ml" / "runs" / "corebox_v2-2" / "weights" / "best.pt"
        )
        # corebox_v2-2 is the only bundled detector. If it misses, BoxDetector
        # handles the legacy OpenCV fallback; loading an older ML model here
        # would make the active GUI behavior depend on obsolete weights.
        fallback = Path(self.config.yolo_fallback_weights) if self.config.yolo_fallback_weights else primary
        return primary, fallback

    def _load_model(self, path: Path):
        key = str(path.resolve())
        if key in self._models:
            return self._models[key]
        if not path.exists():
            return None
        try:
            from ultralytics import YOLO
        except ImportError:
            self._disabled = True
            return None
        model = YOLO(str(path))
        self._models[key] = model
        return model

    def _predict(self, image: np.ndarray, path: Path, imgsz: int) -> list[_Candidate]:
        model = self._load_model(path)
        if model is None:
            return []
        try:
            result = model.predict(
                source=image,
                imgsz=imgsz,
                conf=self.config.yolo_confidence,
                iou=self.config.yolo_iou,
                max_det=self.config.yolo_max_detections,
                verbose=False,
                device="cpu",
            )[0]
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                return []
            xyxy = boxes.xyxy.detach().cpu().numpy()
            confidence = boxes.conf.detach().cpu().numpy()
            height, width = image.shape[:2]
            candidates = []
            for coords, score in zip(xyxy, confidence):
                x1, y1, x2, y2 = map(float, coords)
                x1, x2 = max(0.0, min(x1, width - 1.0)), max(1.0, min(x2, width))
                y1, y2 = max(0.0, min(y1, height - 1.0)), max(1.0, min(y2, height))
                if x2 > x1 and y2 > y1:
                    candidates.append(_Candidate((x1, y1, x2, y2), float(score)))
            return candidates
        except Exception:
            return []

    def _select_candidate(self, candidates: list[_Candidate], image: np.ndarray) -> tuple[_Candidate, float]:
        if len(candidates) == 1:
            candidate = candidates[0]
            return candidate, candidate.confidence
        height, width = image.shape[:2]
        ruler_detector = RulerDetector(self.config)
        scored: list[tuple[float, _Candidate]] = []
        for candidate in candidates:
            roi = self._expand(candidate.box, image.shape)
            local = self._crop(image, roi)
            small, scale = self._resize(local, self.config.yolo_local_max_dimension)
            ruler = ruler_detector.detect_global(small)
            ruler_score = float(ruler.confidence if ruler.found else 0.0)
            x1, y1, x2, y2 = candidate.box
            center = 1.0 - math.hypot((x1 + x2) / 2 / width - 0.5, (y1 + y2) / 2 / height - 0.5)
            score = 0.72 * candidate.confidence + 0.20 * ruler_score + 0.08 * max(0.0, center)
            scored.append((score, candidate))
        score, candidate = max(scored, key=lambda item: item[0])
        return candidate, float(score)

    def _refine(
        self, image: np.ndarray, candidate: _Candidate, selection_score: float, model_name: Path
    ) -> DetectionResult:
        started = time.perf_counter()
        roi = self._expand(candidate.box, image.shape)
        local = self._crop(image, roi)
        small, scale = self._resize(local, self.config.yolo_local_max_dimension)
        quad = self._find_local_quad(small)
        if quad is None:
            local_corners = [(0.0, 0.0), (float(small.shape[1] - 1), 0.0),
                             (float(small.shape[1] - 1), float(small.shape[0] - 1)),
                             (0.0, float(small.shape[0] - 1))]
        else:
            local_corners = quad
        inv = 1.0 / max(scale, 1e-9)
        corners = [(roi[0] + x * inv, roi[1] + y * inv) for x, y in local_corners]
        corners = order_points(corners)
        box = bbox_from_points(corners, image.shape)

        ruler = RulerDetector(self.config).detect_global(small)
        ruler_bounds = None
        if ruler.found and ruler.bounds:
            rx1, ry1, rx2, ry2 = ruler.bounds
            ruler_bounds = (
                int(round(roi[0] + rx1 * inv)), int(round(roi[1] + ry1 * inv)),
                int(round(roi[0] + rx2 * inv)), int(round(roi[1] + ry2 * inv)),
            )
        full = box if ruler_bounds is None else (
            min(box[0], ruler_bounds[0]), min(box[1], ruler_bounds[1]),
            max(box[2], ruler_bounds[2]), max(box[3], ruler_bounds[3]),
        )
        final_roi = expand_roi(full, image.shape, self.config.crop_margin_percent)
        confidence = float(np.clip(0.78 * candidate.confidence + 0.22 * selection_score, 0.0, 1.0))
        return DetectionResult(
            success=confidence >= self.config.confidence_threshold,
            confidence=confidence,
            corners=corners,
            bbox=box,
            ruler=ruler_bounds,
            ruler_confidence=float(ruler.confidence if ruler.found else 0.0),
            final_roi=final_roi,
            reason="yolo_local_refined" if quad is not None else "yolo_local_bbox_refined",
            candidate_scores={
                "yolo_confidence": candidate.confidence,
                "target_selection_score": selection_score,
                "local_refinement": 1.0 if quad is not None else 0.0,
                "ruler_signature_score": float(ruler.confidence if ruler.found else 0.0),
                "yolo_elapsed_ms": (time.perf_counter() - started) * 1000.0,
                "model_primary": 1.0 if "v2-2" in str(model_name) else 0.0,
                "orientation_degrees": 0.0,
            },
            perspective_recommended=perspective_distortion(corners) >= self.config.perspective_threshold,
        )

    def _find_local_quad(self, image: np.ndarray) -> list[Point] | None:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 45, 140)
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        height, width = gray.shape[:2]
        image_area = float(height * width)
        candidates: list[tuple[float, list[Point]]] = []
        for contour in contours:
            area = abs(float(cv2.contourArea(contour)))
            if area < image_area * 0.18:
                continue
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.025 * perimeter, True)
            if len(approx) != 4 or not cv2.isContourConvex(approx):
                continue
            points = [(float(p[0][0]), float(p[0][1])) for p in approx]
            rect = bbox_from_points(points, image.shape)
            rect_area = max(1, (rect[2] - rect[0]) * (rect[3] - rect[1]))
            fill = area / rect_area
            score = min(1.0, area / image_area) * 0.65 + min(1.0, fill) * 0.35
            candidates.append((score, order_points(points)))
        return max(candidates, key=lambda item: item[0])[1] if candidates else None

    def _expand(self, box, shape):
        height, width = shape[:2]
        x1, y1, x2, y2 = map(float, box)
        box_width = max(1.0, x2 - x1)
        box_height = max(1.0, y2 - y1)
        pad_x = max(self.config.yolo_roi_pad_x, self.config.crop_margin_percent)
        pad_y = max(self.config.yolo_roi_pad_y, self.config.crop_margin_percent)
        return (
            max(0, int(round(x1 - box_width * pad_x))),
            max(0, int(round(y1 - box_height * pad_y))),
            min(width - 1, int(round(x2 + box_width * pad_x))),
            min(height - 1, int(round(y2 + box_height * pad_y))),
        )

    @staticmethod
    def _crop(image, rect):
        x1, y1, x2, y2 = map(int, rect)
        return image[max(0, y1):max(y1 + 1, y2), max(0, x1):max(x1 + 1, x2)]

    @staticmethod
    def _resize(image, max_dimension):
        height, width = image.shape[:2]
        scale = min(1.0, max_dimension / max(height, width))
        if scale == 1.0:
            return image, scale
        return cv2.resize(image, (max(1, round(width * scale)), max(1, round(height * scale))), interpolation=cv2.INTER_AREA), scale