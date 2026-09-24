#!/usr/bin/env python3
"""Build an Ultralytics pose dataset from CoreBoxCropper quadrilateral annotations."""
from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

from PIL import Image

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def _annotation_map(path: Path) -> dict[str, dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    annotations = data.get("annotations")
    if not isinstance(annotations, dict):
        raise ValueError("annotations.json must contain an object named 'annotations'")
    return {Path(name).stem: item for name, item in annotations.items() if isinstance(item, dict)}


def _normalized_points(item: dict) -> list[tuple[float, float]]:
    points = item.get("corners")
    if not isinstance(points, list) or len(points) != 4:
        raise ValueError("annotation must contain exactly four corners")
    result = []
    for point in points:
        if not isinstance(point, list) or len(point) != 2:
            raise ValueError("each corner must contain x and y")
        x, y = float(point[0]), float(point[1])
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            raise ValueError("corner coordinates must be normalized to [0, 1]")
        result.append((x, y))
    return result


def _pose_row(item: dict) -> str:
    points = _normalized_points(item)
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    values = [0, (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1]
    for x, y in points:
        values.extend((x, y, 2))
    return " ".join(f"{value:.8f}" if index else str(int(value)) for index, value in enumerate(values))


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare a 4-keypoint CoreBoxCropper pose dataset")
    parser.add_argument("dataset", type=Path, help="Dataset root containing images/ and annotations.json")
    parser.add_argument("--val-ratio", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    root = args.dataset.expanduser().resolve()
    output = (args.output or root / "keypoint_dataset").expanduser().resolve()
    if not 0.05 <= args.val_ratio <= 0.4:
        raise SystemExit("--val-ratio must be between 0.05 and 0.4")
    images_dir = root / "images"
    annotations_path = root / "annotations.json"
    if not images_dir.is_dir():
        raise SystemExit(f"Missing images directory: {images_dir}")
    if not annotations_path.exists():
        raise SystemExit(f"Missing annotations.json: {annotations_path}")

    annotations = _annotation_map(annotations_path)
    images = sorted(p for p in images_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    records = []
    for image in images:
        item = annotations.get(image.stem)
        if item is None:
            raise SystemExit(f"Missing quadrilateral annotation for {image.name}")
        try:
            with Image.open(image) as opened:
                opened.verify()
            _normalized_points(item)
            row = _pose_row(item)
        except Exception as exc:
            raise SystemExit(f"Invalid annotation/image {image.name}: {exc}") from exc
        records.append((image, row))
    if len(records) < 2:
        raise SystemExit("At least two annotated images are required")

    rng = random.Random(args.seed)
    rng.shuffle(records)
    val_count = max(1, round(len(records) * args.val_ratio))
    splits = {"val": records[:val_count], "train": records[val_count:]}
    if not splits["train"]:
        raise SystemExit("Training split is empty")

    for split in splits:
        for folder in (output / "images" / split, output / "labels" / split):
            if folder.exists():
                shutil.rmtree(folder)

    for split, items in splits.items():
        image_out = output / "images" / split
        label_out = output / "labels" / split
        image_out.mkdir(parents=True, exist_ok=True)
        label_out.mkdir(parents=True, exist_ok=True)
        for source, row in items:
            shutil.copy2(source, image_out / source.name)
            (label_out / f"{source.stem}.txt").write_text(row + "\n", encoding="utf-8")

    dataset_yaml = output / "dataset.yaml"
    dataset_yaml.write_text(
        "path: " + json.dumps(output.as_posix()) + "\n"
        "train: images/train\n"
        "val: images/val\n"
        "kpt_shape: [4, 3]\n"
        "flip_idx: [0, 1, 2, 3]\n"
        "names:\n"
        "  0: target_corebox\n"
        "kpt_names:\n"
        "  0: [top_left, top_right, bottom_right, bottom_left]\n",
        encoding="utf-8",
    )
    manifest = {
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "keypoints": ["top_left", "top_right", "bottom_right", "bottom_left"],
        "train": [source.name for source, _ in splits["train"]],
        "val": [source.name for source, _ in splits["val"]],
    }
    (output / "split_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Prepared pose dataset: {output}")
    print(f"Train: {len(splits['train'])} | Val: {len(splits['val'])} | Keypoints: 4")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
