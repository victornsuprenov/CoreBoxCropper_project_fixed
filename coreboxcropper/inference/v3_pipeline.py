from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..config import DetectorConfig
from ..geometry import bbox_from_points, expand_roi, is_valid_quadrilateral, order_points, perspective_distortion
from ..models import DetectionResult
from ..ruler_detector import RulerDetector


@dataclass
class V3Candidate:
    box: tuple[float, float, float, float]
    confidence: float
    corners: list[tuple[float, float]] | None = None
    geometry_score: float = 0.0
    ruler_score: float = 0.0
    quality_score: float = 0.0


class V3Pipeline:
    """Keypoint inference, candidate ranking, and quality gating for V3."""

    def __init__(self, config: DetectorConfig | None = None):
        self.config = config or DetectorConfig()
        self.enabled = bool(self.config.v3_enabled)
        self._model = None
        self._model_path: Path | None = None

    def analyze(self, image: np.ndarray) -> DetectionResult:
        if image is None or image.size == 0:
            return DetectionResult(False, 0.0, reason="invalid_image")
        if len(image.shape) != 3 or image.shape[2] != 3:
            return DetectionResult(False, 0.0, reason="unsupported_image_shape")

        if not self.enabled:
            return DetectionResult(False, 0.0, reason="v3_disabled")
        model = self._load_model()
        if model is None:
            return DetectionResult(False, 0.0, reason="v3_model_unavailable")
        try:
            # Reset Ultralytics' cached predictor between frames. The exported
            # static-shape ONNX model otherwise can retain stale preprocessing
            # state after the first image in a batch.
            if hasattr(model, "predictor"):
                model.predictor = None
            result = model.predict(
                source=image,
                imgsz=self.config.yolo_imgsz,
                conf=self.config.yolo_confidence,
                iou=self.config.yolo_iou,
                max_det=self.config.yolo_max_detections,
                verbose=False,
                device="cpu",
            )[0]
            candidates = self._candidates_from_result(result, image.shape)
        except Exception as exc:
            # Ultralytics caches a predictor between calls. ONNX Runtime can
            # retain a stale input shape after a failed frame, so rebuild the
            # wrapper once before reporting an inference failure.
            self._model = None
            self._model_path = None
            return DetectionResult(
                False,
                0.0,
                reason="v3_inference_error",
                error_detail=f"{type(exc).__name__}: {exc}",
            )
        ranked = self.rank_candidates(candidates, image)
        if not ranked:
            return DetectionResult(False, 0.0, reason="v3_no_detection")
        selected = ranked[0]
        raw_corners = selected.corners or self._box_corners(selected.box)
        if not is_valid_quadrilateral(raw_corners):
            return DetectionResult(False, 0.0, reason="v3_invalid_keypoint_quad")
        corners = order_points(raw_corners)
        if not is_valid_quadrilateral(corners):
            return DetectionResult(False, 0.0, reason="v3_invalid_ordered_quad")
        quality = self._quality_score(corners, image.shape)
        selected.quality_score = quality
        confidence = float(np.clip(
            0.70 * selected.confidence + 0.15 * selected.ruler_score + 0.15 * quality,
            0.0, 1.0,
        ))
        box = bbox_from_points(corners, image.shape)
        final = expand_roi(box, image.shape, self.config.crop_margin_percent)
        passed = self.validate_quality(selected, corners)
        return DetectionResult(
            success=passed,
            confidence=confidence,
            corners=corners,
            bbox=box,
            final_roi=final,
            reason="v3_keypoint_quality_gate" if passed else "v3_quality_gate_failed",
            candidate_scores={
                "v3_candidates": float(len(candidates)),
                "v3_selected_confidence": float(selected.confidence),
                "v3_ruler_score": float(selected.ruler_score),
                "v3_quality_score": float(quality),
                "v3_model_loaded": 1.0,
            },
            perspective_recommended=perspective_distortion(corners) >= self.config.perspective_threshold,
            debug_images={
                "v3_prediction": self.make_debug_frame(image, selected.box, corners),
            },
        )

    def _resolve_model_path(self) -> Path:
        if self.config.v3_keypoint_model:
            return Path(self.config.v3_keypoint_model).expanduser()
        project = Path(__file__).resolve().parents[2]
        return project / "runs" / "pose" / "corebox_keypoints_v2" / "weights" / "best.onnx"

    def _load_model(self):
        path = self._resolve_model_path()
        if not path.exists():
            return None
        try:
            from ultralytics import YOLO
            # Recreate the ONNX wrapper for every frame. ONNX Runtime 1.30
            # can retain stale predictor state between sequential calls on
            # this exported static-shape pose model.
            self._model = YOLO(str(path))
            self._model_path = path
            return self._model
        except Exception:
            return None

    @staticmethod
    def _box_corners(box):
        x1, y1, x2, y2 = box
        return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]

    def _candidates_from_result(self, result, shape) -> list[V3Candidate]:
        boxes = getattr(result, "boxes", None)
        keypoints = getattr(result, "keypoints", None)
        if boxes is None or keypoints is None or len(boxes) == 0:
            return []
        xyxy = boxes.xyxy.detach().cpu().numpy()
        confidence = boxes.conf.detach().cpu().numpy()
        point_tensor = getattr(keypoints, "xy", None)
        if point_tensor is None:
            return []
        points = point_tensor.detach().cpu().numpy()
        height, width = shape[:2]
        candidates = []
        for index, (coords, score) in enumerate(zip(xyxy, confidence)):
            x1, y1, x2, y2 = map(float, coords)
            box = (
                max(0.0, min(x1, width - 1.0)), max(0.0, min(y1, height - 1.0)),
                max(1.0, min(x2, width)), max(1.0, min(y2, height)),
            )
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            corners = None
            if index < len(points) and points[index].shape == (4, 2):
                corners = [(float(x), float(y)) for x, y in points[index]]
            candidates.append(V3Candidate(box=box, confidence=float(score), corners=corners))
        return candidates

    def rank_candidates(self, candidates: list[V3Candidate], image: np.ndarray | None = None) -> list[V3Candidate]:
        if not candidates:
            return []
        if image is not None:
            ruler_detector = RulerDetector(self.config)
            height, width = image.shape[:2]
            for candidate in candidates:
                x1, y1, x2, y2 = map(int, candidate.box)
                roi = image[max(0, y1):max(y1 + 1, y2), max(0, x1):max(x1 + 1, x2)]
                ruler = ruler_detector.detect_global(roi)
                candidate.ruler_score = float(ruler.confidence if ruler.found else 0.0)
                center = 1.0 - abs((x1 + x2) / 2 / max(width, 1) - 0.5)
                candidate.geometry_score = float(max(0.0, center))
        return sorted(
            candidates,
            key=lambda c: (
                c.confidence + 0.35 * c.ruler_score + 0.25 * c.geometry_score + 0.15 * c.quality_score,
                c.confidence,
            ),
            reverse=True,
        )[: max(1, self.config.v3_candidate_limit)]

    @staticmethod
    def _quality_score(corners, shape) -> float:
        if not is_valid_quadrilateral(corners):
            return 0.0
        height, width = shape[:2]
        ordered = order_points(corners)
        box_width = max(1.0, max(p[0] for p in ordered) - min(p[0] for p in ordered))
        box_height = max(1.0, max(p[1] for p in ordered) - min(p[1] for p in ordered))
        area_ratio = (box_width * box_height) / max(float(width * height), 1.0)
        aspect = box_height / box_width
        area_score = 1.0 if 0.05 <= area_ratio <= 0.9 else 0.0
        aspect_score = 1.0 if 1.0 <= aspect <= 8.5 else 0.0
        distortion_score = max(0.0, 1.0 - perspective_distortion(ordered))
        return float(np.clip(0.4 * area_score + 0.3 * aspect_score + 0.3 * distortion_score, 0.0, 1.0))

    def validate_quality(
        self,
        candidate: V3Candidate,
        corners: list[tuple[float, float]] | None = None,
    ) -> bool:
        if candidate.confidence < self.config.v3_quality_gate_min_confidence:
            return False
        if corners is not None and len(corners) == 4:
            return candidate.quality_score >= self.config.v3_perspective_min_score
        return True

    @staticmethod
    def make_debug_frame(
        image: np.ndarray,
        box: tuple[float, float, float, float],
        corners: list[tuple[float, float]] | None = None,
    ) -> np.ndarray:
        annotated = image.copy()
        x1, y1, x2, y2 = map(int, box)
        # Keep the detector bbox visible, but distinguish it from the actual
        # four-point contour used for perspective cropping.
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 0), 2)
        if corners and len(corners) == 4:
            points = np.asarray(corners, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(annotated, [points], isClosed=True, color=(0, 255, 0), thickness=3)
            for index, (x, y) in enumerate(corners, start=1):
                point = (int(round(x)), int(round(y)))
                cv2.circle(annotated, point, 7, (0, 255, 255), -1)
                cv2.putText(
                    annotated, f"P{index}", (point[0] + 8, point[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA,
                )
        return annotated
