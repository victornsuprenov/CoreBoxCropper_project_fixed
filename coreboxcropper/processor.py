from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .config import DetectorConfig
from .geometry import calculate_perspective
from .models import DetectionResult, Point, Rect


class ImageProcessor:
    """Turns a trusted detection into a crop and writes it to disk."""

    def __init__(self, config: DetectorConfig | None = None):
        self.config = config or DetectorConfig()

    def process(self, image: np.ndarray, detection: DetectionResult) -> np.ndarray:
        if image is None or image.size == 0:
            raise ValueError("image is empty")
        if not detection.success or not detection.final_roi:
            raise ValueError("cannot process an unsuccessful detection")
        if detection.perspective_recommended and len(detection.corners) == 4:
            matrix, size = calculate_perspective(detection.corners, self.config.crop_margin_percent)
            return cv2.warpPerspective(image, matrix, size, borderMode=cv2.BORDER_REPLICATE)
        return self._crop_rect(image, detection.final_roi)

    def process_manual(self, image: np.ndarray, corners: list[Point]) -> np.ndarray:
        if len(corners) != 4:
            raise ValueError("manual crop requires four corners")
        # Manual selection is deliberately perspective-aware.
        matrix, size = calculate_perspective(corners, self.config.crop_margin_percent)
        return cv2.warpPerspective(image, matrix, size, borderMode=cv2.BORDER_REPLICATE)

    def save(self, image: np.ndarray, output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        suffix = output_path.suffix.lower() or ".jpg"
        if suffix in {".jpg", ".jpeg"}:
            params = [cv2.IMWRITE_JPEG_QUALITY, int(self.config.jpeg_quality)]
            extension = ".jpg"
        elif suffix == ".png":
            params = [cv2.IMWRITE_PNG_COMPRESSION, 3]
            extension = ".png"
        elif suffix == ".webp":
            params = [cv2.IMWRITE_WEBP_QUALITY, int(self.config.jpeg_quality)]
            extension = ".webp"
        else:
            raise ValueError(f"unsupported output format: {suffix}")
        ok, encoded = cv2.imencode(extension, image, params)
        if not ok:
            raise OSError("failed to encode output image")
        encoded.tofile(str(output_path))
        return output_path

    @staticmethod
    def _crop_rect(image: np.ndarray, roi: Rect) -> np.ndarray:
        height, width = image.shape[:2]
        x1, y1, x2, y2 = roi
        x1, y1 = max(0, min(x1, width - 1)), max(0, min(y1, height - 1))
        x2, y2 = max(x1 + 1, min(x2 + 1, width)), max(y1 + 1, min(y2 + 1, height))
        return image[y1:y2, x1:x2].copy()
