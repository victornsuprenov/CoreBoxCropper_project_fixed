#!/usr/bin/env python3
"""Compare a selected YOLO model with an optional explicit baseline.

This is a cheap production-safety check: it does not run the legacy global
OpenCV detector and does not modify any files. A frame is reported as a
regression when the baseline finds a confident target and the primary model
does not.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Allow direct execution from the ml directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from coreboxcropper.application import ApplicationService
from coreboxcropper.config import DetectorConfig

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def run(source: Path, primary: Path | None = None, baseline: Path | None = None) -> int:
    source = source.expanduser().resolve()
    images_dir = source / "images" if (source / "images").is_dir() else source
    labels_dir = source / "labels" if (source / "labels").is_dir() else source.parent / "labels"
    regression_dir = source / "_regression"
    predictions_dir = regression_dir / "predictions"
    bbox_dir = regression_dir / "bbox_crops"
    crops_dir = regression_dir / "crops"
    debug_dir = regression_dir / "debug"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    bbox_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)

    primary_config = DetectorConfig()
    if primary:
        primary_config.yolo_primary_weights = str(primary)
    baseline_config = DetectorConfig(
        yolo_primary_weights=str(baseline) if baseline else "",
        yolo_fallback_weights=str(baseline) if baseline else "",
    )
    primary_service = ApplicationService(primary_config)
    baseline_service = ApplicationService(baseline_config)
    rows = []
    for image in sorted(p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS):
        loaded = primary_service.load_image(image)
        started = time.perf_counter()
        current = primary_service.detector.detect(loaded)
        primary_ms = (time.perf_counter() - started) * 1000.0
        started = time.perf_counter()
        reference = baseline_service.detector.detect(loaded)
        baseline_ms = (time.perf_counter() - started) * 1000.0
        regression = bool(reference.success and not current.success)

        prediction_paths = _save_prediction_frames(primary_service, predictions_dir, image, loaded, current)
        bbox_path = None
        if current.success and current.bbox:
            x1, y1, x2, y2 = current.bbox
            bbox_crop = loaded[max(0, y1):max(1, y2), max(0, x1):max(1, x2)].copy()
            bbox_path = bbox_dir / f"{image.stem}.jpg"
            primary_service.processor.save(bbox_crop, bbox_path)
        crop_path = None
        if current.success and current.final_roi:
            crop = primary_service.processor.process(loaded, current)
            crop_path = crops_dir / f"{image.stem}.jpg"
            primary_service.processor.save(crop, crop_path)
        _save_debug_snapshot(debug_dir, image, loaded, current, bbox_path, crop_path)

        label_path = labels_dir / f"{image.stem}.txt"
        label_box = _read_label_box(label_path, loaded.shape) if label_path.exists() else None
        primary_iou = _box_iou(current.bbox, label_box) if label_box else None
        baseline_iou = _box_iou(reference.bbox, label_box) if label_box else None
        rows.append({
            "image": image.name,
            "prediction_paths": [str(path.relative_to(source)) for path in prediction_paths],
            "bbox_path": str(bbox_path.relative_to(source)) if bbox_path else None,
            "crop_path": str(crop_path.relative_to(source)) if crop_path else None,
            "primary_success": current.success,
            "primary_confidence": round(current.confidence, 4),
            "primary_reason": current.reason,
            "primary_ruler_found": current.ruler is not None,
            "baseline_success": reference.success,
            "baseline_confidence": round(reference.confidence, 4),
            "baseline_reason": reference.reason,
            "baseline_ruler_found": reference.ruler is not None,
            "regression": regression,
            "primary_target_iou": round(primary_iou, 4) if primary_iou is not None else None,
            "baseline_target_iou": round(baseline_iou, 4) if baseline_iou is not None else None,
            "primary_ms": round(primary_ms, 1),
            "baseline_ms": round(baseline_ms, 1),
        })
    regressions = [row for row in rows if row["regression"]]
    report = {
        "source": str(source),
        "images": len(rows),
        "primary_success": sum(row["primary_success"] for row in rows),
        "baseline_success": sum(row["baseline_success"] for row in rows),
        "regressions": len(regressions),
        "primary_recall": round(sum(row["primary_success"] for row in rows) / max(len(rows), 1), 4),
        "baseline_recall": round(sum(row["baseline_success"] for row in rows) / max(len(rows), 1), 4),
        "primary_mean_target_iou": _mean(row["primary_target_iou"] for row in rows),
        "baseline_mean_target_iou": _mean(row["baseline_target_iou"] for row in rows),
        "primary_ruler_rate": round(sum(bool(row["primary_ruler_found"]) for row in rows) / max(len(rows), 1), 4),
        "baseline_ruler_rate": round(sum(bool(row["baseline_ruler_found"]) for row in rows) / max(len(rows), 1), 4),
        "primary_avg_ms": round(sum(row["primary_ms"] for row in rows) / max(len(rows), 1), 1),
        "baseline_avg_ms": round(sum(row["baseline_ms"] for row in rows) / max(len(rows), 1), 1),
        "prediction_dir": str(predictions_dir.relative_to(source)),
        "bbox_crop_dir": str(bbox_dir.relative_to(source)),
        "crop_dir": str(crops_dir.relative_to(source)),
        "debug_dir": str(debug_dir.relative_to(source)),
        "rows": rows,
    }
    output = regression_dir / "report.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    legacy_output = source / "_regression_report.json"
    legacy_output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"images={report['images']} primary_success={report['primary_success']} baseline_success={report['baseline_success']}")
    print(f"recall primary={report['primary_recall']} baseline={report['baseline_recall']}")
    print(f"target_iou primary={report['primary_mean_target_iou']} baseline={report['baseline_mean_target_iou']}")
    print(f"regressions={report['regressions']} primary_avg_ms={report['primary_avg_ms']}")
    print(f"report={output}")
    print(f"predictions={predictions_dir}")
    print(f"crops={crops_dir}")
    if regressions:
        print("REGRESSION FILES:")
        for row in regressions:
            print(f"  {row['image']}: {row['primary_reason']} vs {row['baseline_reason']}")
        return 1
    return 0


def _save_prediction_frames(service: ApplicationService, predictions_dir: Path, image: Path, loaded: np.ndarray, detection) -> list[Path]:
    frames = detection.debug_images or {"prediction": _annotate_prediction(loaded, detection)}
    saved: list[Path] = []
    for name, frame in frames.items():
        path = predictions_dir / f"{image.stem}_{name}.jpg"
        service._write_frame(path, frame)
        saved.append(path)
    return saved


def _save_debug_snapshot(debug_dir: Path, image: Path, loaded: np.ndarray, detection, bbox_path: Path | None, crop_path: Path | None) -> None:
    payload = {
        "image": image.name,
        "bbox": list(detection.bbox) if detection.bbox else None,
        "final_roi": list(detection.final_roi) if detection.final_roi else None,
        "corners": [[float(x), float(y)] for x, y in detection.corners] if detection.corners else [],
        "perspective_recommended": bool(detection.perspective_recommended),
        "bbox_crop": str(bbox_path.relative_to(image.parents[0])) if bbox_path else None,
        "crop": str(crop_path.relative_to(image.parents[0])) if crop_path else None,
        "shape": list(loaded.shape),
    }
    (debug_dir / f"{image.stem}_debug.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _annotate_prediction(image: np.ndarray, detection) -> np.ndarray:
    annotated = image.copy()
    if detection.bbox:
        x1, y1, x2, y2 = detection.bbox
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 255), 3)
    if detection.final_roi:
        x1, y1, x2, y2 = detection.final_roi
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 165, 255), 2)
    if getattr(detection, "corners", None):
        points = np.asarray(detection.corners, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(annotated, [points], isClosed=True, color=(0, 255, 0), thickness=2)
    return annotated


def _mean(values):
    values = [float(value) for value in values if value is not None]
    return round(sum(values) / len(values), 4) if values else None


def _read_label_box(path: Path, shape) -> tuple[int, int, int, int] | None:
    rows = [line.split() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if len(rows) != 1 or len(rows[0]) != 5:
        return None
    _, cx, cy, width, height = map(float, rows[0])
    image_height, image_width = shape[:2]
    return (
        int(round((cx - width / 2) * image_width)),
        int(round((cy - height / 2) * image_height)),
        int(round((cx + width / 2) * image_width)),
        int(round((cy + height / 2) * image_height)),
    )


def _box_iou(left, right) -> float:
    if not left or not right:
        return 0.0
    x1 = max(left[0], right[0]); y1 = max(left[1], right[1])
    x2 = min(left[2], right[2]); y2 = min(left[3], right[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    left_area = max(0, left[2] - left[0]) * max(0, left[3] - left[1])
    right_area = max(0, right[2] - right[0]) * max(0, right[3] - right[1])
    return intersection / max(left_area + right_area - intersection, 1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--primary", type=Path, default=None)
    parser.add_argument("--baseline", type=Path, default=None)
    args = parser.parse_args()
    return run(args.source, args.primary, args.baseline)


if __name__ == "__main__":
    raise SystemExit(main())