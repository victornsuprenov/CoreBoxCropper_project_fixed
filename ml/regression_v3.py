#!/usr/bin/env python3
"""Regression harness for the V3 keypoint ONNX pipeline."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from coreboxcropper.application import ApplicationService
from coreboxcropper.config import DetectorConfig
from coreboxcropper.geometry import calculate_perspective
from coreboxcropper.inference.v3_pipeline import V3Pipeline
from coreboxcropper.processor import ImageProcessor

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def run(source: Path, v3_model: Path | None = None, baseline: Path | None = None) -> int:
    source = source.expanduser().resolve()
    if source.is_file():
        image_paths = [source]
        source_root = source.parent
    else:
        images_dir = source / "images" if (source / "images").is_dir() else source
        if not images_dir.is_dir():
            raise SystemExit(f"Image directory not found: {images_dir}")
        image_paths = sorted(
            p for p in images_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        source_root = source
    regression_dir = source_root / "_regression_v3"
    predictions_dir = regression_dir / "predictions"
    crops_dir = regression_dir / "crops"
    debug_dir = regression_dir / "debug"
    for directory in (predictions_dir, crops_dir, debug_dir):
        directory.mkdir(parents=True, exist_ok=True)

    config = DetectorConfig(
        v3_enabled=True,
        v3_keypoint_model=str(v3_model) if v3_model else "",
    )
    service = ApplicationService(config)
    pipeline = V3Pipeline(config)
    processor = ImageProcessor(config)
    baseline_service = None
    if baseline:
        baseline_config = DetectorConfig(
            yolo_primary_weights=str(baseline),
            yolo_fallback_weights=str(baseline),
            v3_enabled=False,
        )
        baseline_service = ApplicationService(baseline_config)

    rows = []
    for image_path in image_paths:
        loaded = service.load_image(image_path)
        started = time.perf_counter()
        detection = pipeline.analyze(loaded)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        baseline_detection = None
        baseline_ms = None
        if baseline_service is not None:
            started = time.perf_counter()
            baseline_detection = baseline_service.detector.detect(loaded)
            baseline_ms = (time.perf_counter() - started) * 1000.0

        corners_valid = bool(len(detection.corners) == 4)
        perspective_applied = False
        crop_path = None
        if corners_valid:
            try:
                matrix, size = calculate_perspective(detection.corners, config.crop_margin_percent)
                crop = cv2.warpPerspective(loaded, matrix, size, borderMode=cv2.BORDER_REPLICATE)
                perspective_applied = bool(crop.size and crop.shape[0] > 1 and crop.shape[1] > 1)
                crop_path = crops_dir / f"{image_path.stem}.jpg"
                processor.save(crop, crop_path)
            except Exception:
                perspective_applied = False

        prediction = pipeline.make_debug_frame(
            loaded,
            detection.bbox or (0, 0, loaded.shape[1] - 1, loaded.shape[0] - 1),
            detection.corners if corners_valid else None,
        )
        prediction_path = predictions_dir / f"{image_path.stem}_v3.jpg"
        processor.save(prediction, prediction_path)
        debug_path = debug_dir / f"{image_path.stem}_v3.json"
        debug_path.write_text(
            json.dumps(
                {
                    "image": image_path.name,
                    "success": detection.success,
                    "reason": detection.reason,
                    "error_detail": detection.error_detail,
                    "confidence": round(float(detection.confidence), 4),
                    "corners": [[float(x), float(y)] for x, y in detection.corners],
                    "bbox": list(detection.bbox) if detection.bbox else None,
                    "final_roi": list(detection.final_roi) if detection.final_roi else None,
                    "perspective_applied": perspective_applied,
                    "candidate_scores": detection.candidate_scores,
                    "shape": list(loaded.shape),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        if detection.reason.startswith("v3_inference_error"):
            (debug_dir / "last_inference_error.txt").write_text(
                f"{image_path.name}\n{detection.reason}\n",
                encoding="utf-8",
            )
        effective_success = bool(detection.success and corners_valid and perspective_applied)
        rows.append(
            {
                "image": image_path.name,
            "success": effective_success,
            "gate_success": bool(detection.success),
                "confidence": round(float(detection.confidence), 4),
                "reason": detection.reason,
                "corners_valid": corners_valid,
                "perspective_applied": perspective_applied,
                "prediction": str(prediction_path.relative_to(source)),
                "crop": str(crop_path.relative_to(source)) if crop_path else None,
                "debug": str(debug_path.relative_to(source)),
                "elapsed_ms": round(elapsed_ms, 1),
                "baseline_success": bool(baseline_detection.success) if baseline_detection else None,
                "baseline_confidence": round(float(baseline_detection.confidence), 4) if baseline_detection else None,
                "baseline_ms": round(baseline_ms, 1) if baseline_ms is not None else None,
            }
        )

    regressions = [row for row in rows if row["baseline_success"] and not row["success"]]
    report = {
        "source": str(source),
        "model": str(v3_model or pipeline._resolve_model_path()),
        "images": len(rows),
        "v3_success": sum(row["success"] for row in rows),
        "v3_recall": round(sum(row["success"] for row in rows) / max(len(rows), 1), 4),
        "valid_corners": sum(row["corners_valid"] for row in rows),
        "perspective_crops": sum(row["perspective_applied"] for row in rows),
        "regressions": len(regressions),
        "avg_ms": round(sum(row["elapsed_ms"] for row in rows) / max(len(rows), 1), 1),
        "prediction_dir": str(predictions_dir.relative_to(source)),
        "crop_dir": str(crops_dir.relative_to(source)),
        "debug_dir": str(debug_dir.relative_to(source)),
        "rows": rows,
    }
    output = regression_dir / "report.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"images={report['images']} v3_success={report['v3_success']} "
        f"recall={report['v3_recall']} corners={report['valid_corners']} "
        f"perspective_crops={report['perspective_crops']} regressions={report['regressions']}"
    )
    print(f"report={output}")
    print(f"predictions={predictions_dir}")
    print(f"crops={crops_dir}")
    if regressions:
        print("REGRESSION FILES:")
        for row in regressions:
            print(f"  {row['image']}: {row['reason']}")
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run V3 keypoint ONNX regression")
    parser.add_argument("source", type=Path)
    parser.add_argument("--v3", dest="v3_model", type=Path, default=None, help="V3 ONNX model")
    parser.add_argument("--baseline", type=Path, default=None, help="Optional YOLO baseline model")
    args = parser.parse_args()
    return run(args.source, args.v3_model, args.baseline)


if __name__ == "__main__":
    raise SystemExit(main())
