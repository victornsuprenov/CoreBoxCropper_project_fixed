#!/usr/bin/env python3
"""Validate an Ultralytics 4-keypoint pose dataset without training."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from PIL import Image


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a CoreBoxCropper keypoint dataset")
    parser.add_argument("dataset", type=Path, help="keypoint_dataset directory")
    args = parser.parse_args()
    root = args.dataset.expanduser().resolve()
    errors: list[str] = []
    warnings: list[str] = []
    counts = {}

    if not root.is_dir():
        print(f"Dataset directory not found: {root}")
        print("Сначала выполните prepare_keypoint_training.py или проверьте путь к keypoint_dataset.")
        return 2

    yaml = root / "dataset.yaml"
    if not yaml.exists():
        errors.append("Missing dataset.yaml")
    else:
        text = yaml.read_text(encoding="utf-8")
        for required in ("kpt_shape: [4, 3]", "train:", "val:", "names:"):
            if required not in text:
                errors.append(f"dataset.yaml missing '{required}'")

    for split in ("train", "val"):
        images = root / "images" / split
        labels = root / "labels" / split
        if not images.is_dir() or not labels.is_dir():
            errors.append(f"Missing split directories for {split}")
            continue
        image_files = [p for p in images.iterdir() if p.is_file()]
        counts[split] = len(image_files)
        for image in image_files:
            label = labels / f"{image.stem}.txt"
            if not label.exists():
                errors.append(f"{split}/{image.name}: missing label")
                continue
            try:
                with Image.open(image) as opened:
                    width, height = opened.size
                    opened.verify()
                rows = [line.split() for line in label.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
                if len(rows) != 1 or len(rows[0]) != 17:
                    errors.append(f"{split}/{label.name}: expected one row with 17 values")
                    continue
                values = [float(value) for value in rows[0]]
                if int(values[0]) != 0 or not all(math.isfinite(value) for value in values):
                    errors.append(f"{split}/{label.name}: invalid class or non-finite values")
                    continue
                coordinates = values[1:5] + [values[index] for index in (5, 6, 8, 9, 11, 12, 14, 15)]
                visibility = values[7:17:3]
                if not all(0.0 <= value <= 1.0 for value in coordinates):
                    errors.append(f"{split}/{label.name}: normalized coordinates outside [0,1]")
                if not all(value in {0.0, 1.0, 2.0} for value in visibility):
                    errors.append(f"{split}/{label.name}: keypoint visibility must be 0, 1 or 2")
                if values[3] <= 0 or values[4] <= 0:
                    errors.append(f"{split}/{label.name}: bbox has non-positive size")
                keypoints = [(values[index], values[index + 1]) for index in range(5, 17, 3)]
                if len({point for point in keypoints}) != 4:
                    errors.append(f"{split}/{label.name}: duplicate keypoints")
                if width < 32 or height < 32:
                    warnings.append(f"{split}/{image.name}: very small image {width}x{height}")
            except Exception as exc:
                errors.append(f"{split}/{image.name}: {exc}")

    report = {
        "dataset": str(root),
        "ok": not errors and all(counts.get(split, 0) > 0 for split in ("train", "val")),
        "errors": errors,
        "warnings": warnings,
        "images": counts,
        "keypoints": ["top_left", "top_right", "bottom_right", "bottom_left"],
    }
    output = root / "keypoint_validation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Dataset: {root}")
    print(f"Train: {counts.get('train', 0)} | Val: {counts.get('val', 0)}")
    print(f"RESULT: {'OK' if report['ok'] else 'ERRORS FOUND'}")
    for error in errors:
        print(f"ERROR: {error}")
    for warning in warnings:
        print(f"WARNING: {warning}")
    print(f"Report: {output}")
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
