from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Iterable

import cv2
import numpy as np
from PIL import Image, ImageOps

from .config import DetectorConfig
from .detector import BoxDetector
from .logger import get_debug_dir, get_logger
from .models import DetectionResult, ProcessResult
from .processor import ImageProcessor

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


class ApplicationService:
    def __init__(self, config: DetectorConfig | None = None):
        self.config = config or DetectorConfig()
        self.detector = BoxDetector(self.config)
        self.processor = ImageProcessor(self.config)
        self.logger = get_logger()

    def load_image(self, path: Path) -> np.ndarray:
        if not path.exists():
            raise FileNotFoundError(str(path))
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"unsupported image format: {path.suffix}")
        try:
            with Image.open(path) as pil_image:
                normalized = ImageOps.exif_transpose(pil_image).convert("RGB")
                return cv2.cvtColor(np.asarray(normalized), cv2.COLOR_RGB2BGR)
        except Exception as exc:
            raise ValueError(f"invalid image: {path.name}") from exc

    def preview_detection(self, input_path: str | Path) -> tuple[np.ndarray, DetectionResult]:
        path = Path(input_path)
        image = self.load_image(path)
        started = time.perf_counter()
        detection = self.detector.detect(image)
        elapsed = int((time.perf_counter() - started) * 1000)
        self.logger.info(
            "file=%s stage=detect result=%s confidence=%.3f elapsed_ms=%d",
            path.name, detection.reason, detection.confidence, elapsed,
        )
        return image, detection

    def process_image(
        self,
        input_path: str | Path,
        output_path: str | Path | None = None,
        debug: bool | Path = False,
        overwrite: bool = False,
    ) -> ProcessResult:
        source = Path(input_path)
        started = time.perf_counter()
        output = Path(output_path) if output_path else self.default_output_path(source)
        try:
            image = self.load_image(source)
            detection = self.detector.detect(image)
            if not detection.success:
                result = ProcessResult(
                    "error", str(source), str(output), detection.confidence,
                    False, bool(detection.ruler), detection.reason, 0, detection,
                )
                if debug:
                    self._save_debug(source, detection, debug, image)
                self._log_result(result, started)
                return result
            if output.exists() and not overwrite:
                result = ProcessResult(
                    "error", str(source), str(output), detection.confidence,
                    True, bool(detection.ruler), "output_exists", 0, detection,
                )
                self._log_result(result, started)
                return result
            cropped = self.processor.process(image, detection)
            self.processor.save(cropped, output)
            if debug:
                self._save_debug(source, detection, debug, image, cropped)
            result = ProcessResult(
                "success", str(source), str(output), detection.confidence,
                True, bool(detection.ruler), detection.reason, 0, detection,
            )
        except FileNotFoundError:
            result = ProcessResult("error", str(source), str(output), reason="file_not_found")
        except PermissionError:
            result = ProcessResult("error", str(source), str(output), reason="permission_denied")
        except (ValueError, OSError) as exc:
            result = ProcessResult("error", str(source), str(output), reason=str(exc))
        except Exception:
            self.logger.exception("file=%s stage=process result=unexpected_error", source.name)
            result = ProcessResult("error", str(source), str(output), reason="processing_error")
        self._log_result(result, started)
        return result

    def process_paths(
        self,
        paths: Iterable[str | Path],
        output_dir: str | Path,
        overwrite: bool = True,
        debug: bool | Path = False,
        on_result: Callable[[ProcessResult], None] | None = None,
        source_root: str | Path | None = None,
    ) -> list[ProcessResult]:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        root = Path(source_root) if source_root is not None else None
        results: list[ProcessResult] = []
        unique_paths = sorted({Path(p) for p in paths}, key=lambda p: str(p).lower())
        for source in unique_paths:
            if source.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            relative = source.relative_to(root) if root is not None else Path(source.name)
            output = destination / relative
            result = self.process_image(source, output, debug=debug, overwrite=overwrite)
            results.append(result)
            if on_result:
                on_result(result)
        return results

    def process_directory(
        self,
        input_dir: str | Path,
        output_dir: str | Path,
        recursive: bool = False,
        debug: bool | Path = False,
        overwrite: bool = False,
        on_result: Callable[[ProcessResult], None] | None = None,
    ) -> list[ProcessResult]:
        source_dir = Path(input_dir)
        if not source_dir.exists():
            return [ProcessResult("error", str(source_dir), reason="input_directory_not_found")]
        pattern = "**/*" if recursive else "*"
        files = [
            p for p in source_dir.glob(pattern)
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
        return self.process_paths(
            files, output_dir, overwrite=overwrite, debug=debug,
            on_result=on_result, source_root=source_dir,
        )

    def save_manual(
        self, input_path: str | Path, output_path: str | Path, corners: list[tuple[float, float]]
    ) -> ProcessResult:
        source, output = Path(input_path), Path(output_path)
        started = time.perf_counter()
        try:
            image = self.load_image(source)
            cropped = self.processor.process_manual(image, corners)
            self.processor.save(cropped, output)
            result = ProcessResult("success", str(source), str(output), 1.0, True, False, "manual_crop")
        except Exception as exc:
            result = ProcessResult("error", str(source), str(output), reason=str(exc))
        self._log_result(result, started)
        return result

    @staticmethod
    def default_output_path(input_path: Path) -> Path:
        # Keep the original filename, but never overwrite the source merely
        # because the caller omitted an explicit output path.
        return input_path.parent / "Result" / input_path.name

    def _save_debug(
        self, source: Path, detection: DetectionResult, debug: bool | Path,
        image: np.ndarray, cropped: np.ndarray | None = None,
    ) -> None:
        base = Path(debug) if isinstance(debug, (str, Path)) else source.parent / "Debug"
        directory = get_debug_dir(base, source)
        for name, frame in detection.debug_images.items():
            self._write_frame(directory / f"{source.stem}_{name}.jpg", frame)
        if cropped is not None:
            self._write_frame(directory / f"{source.stem}_result.jpg", cropped)
        if detection.final_roi:
            roi = image.copy()
            x1, y1, x2, y2 = detection.final_roi
            cv2.rectangle(roi, (x1, y1), (x2, y2), (0, 255, 255), 5)
            self._write_frame(directory / f"{source.stem}_final_roi.jpg", roi)

    @staticmethod
    def _write_frame(path: Path, image: np.ndarray) -> None:
        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if ok:
            path.parent.mkdir(parents=True, exist_ok=True)
            encoded.tofile(str(path))

    def _log_result(self, result: ProcessResult, started: float) -> None:
        result.elapsed_ms = int((time.perf_counter() - started) * 1000)
        self.logger.info(
            "file=%s stage=process result=%s confidence=%.3f error=%s elapsed_ms=%d",
            Path(result.input_path).name, result.status, result.confidence,
            result.reason, result.elapsed_ms,
        )
