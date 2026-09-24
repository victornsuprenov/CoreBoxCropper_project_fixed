#!/usr/bin/env python3
"""Offline validator for a CoreBoxCropper YOLO dataset.

Usage:
    python ml/validate_dataset.py D:\\DTM

Creates DTM/_validation/validation_report.txt and validation_report.json.
The script does not train anything and does not modify images or labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def sha1_file(path: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def image_size(path: Path):
    try:
        from PIL import Image
        with Image.open(path) as im:
            return int(im.width), int(im.height), str(im.format or "")
    except Exception as exc:
        return None, None, f"ERROR: {exc}"


def parse_label(path: Path):
    rows = []
    errors = []
    text = path.read_text(encoding="utf-8-sig")
    lines = text.splitlines()
    for lineno, raw in enumerate(lines, 1):
        s = raw.strip()
        if not s:
            continue
        parts = s.split()
        if len(parts) != 5:
            errors.append(f"line {lineno}: expected 5 values, got {len(parts)}")
            continue
        try:
            cls = int(parts[0])
            vals = [float(x) for x in parts[1:]]
        except Exception:
            errors.append(f"line {lineno}: non-numeric value")
            continue
        cx, cy, w, h = vals
        if cls != 0:
            errors.append(f"line {lineno}: class={cls}, expected 0")
        if not all(math.isfinite(x) for x in vals):
            errors.append(f"line {lineno}: non-finite coordinate")
        if w <= 0 or h <= 0:
            errors.append(f"line {lineno}: width/height must be > 0")
        if not (0 <= cx <= 1 and 0 <= cy <= 1):
            errors.append(f"line {lineno}: center outside [0,1]")
        # Bounding box must remain inside image after denormalisation.
        if not (0 <= cx - w / 2 <= 1 and 0 <= cx + w / 2 <= 1 and
                0 <= cy - h / 2 <= 1 and 0 <= cy + h / 2 <= 1):
            errors.append(f"line {lineno}: bbox exceeds image bounds")
        rows.append((cls, cx, cy, w, h))
    return rows, errors


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate CoreBoxCropper ML dataset")
    ap.add_argument("dataset", type=Path, help="DTM dataset folder")
    args = ap.parse_args()
    root = args.dataset.expanduser().resolve()
    out = root / "_validation"
    out.mkdir(exist_ok=True)

    report = {
        "dataset": str(root),
        "ok": True,
        "errors": [],
        "warnings": [],
        "stats": {},
        "files": [],
    }

    images_dir = root / "images"
    labels_dir = root / "labels"
    if not images_dir.is_dir():
        report["errors"].append("Missing images/ directory")
    if not labels_dir.is_dir():
        report["errors"].append("Missing labels/ directory")

    classes = []
    classes_path = root / "classes.txt"
    if classes_path.exists():
        classes = [x.strip() for x in classes_path.read_text(encoding="utf-8-sig").splitlines() if x.strip()]
        if classes != ["target_corebox"]:
            report["errors"].append(f"classes.txt should contain exactly target_corebox; got {classes!r}")
    else:
        report["errors"].append("Missing classes.txt")

    images = sorted([p for p in images_dir.glob("**/*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]) if images_dir.is_dir() else []
    labels = sorted(labels_dir.glob("*.txt")) if labels_dir.is_dir() else []
    image_stems = {p.stem: p for p in images}
    label_stems = {p.stem: p for p in labels}

    missing_labels = sorted(set(image_stems) - set(label_stems))
    orphan_labels = sorted(set(label_stems) - set(image_stems))
    if missing_labels:
        report["errors"].append(f"Images without labels: {len(missing_labels)} ({', '.join(missing_labels[:10])})")
    if orphan_labels:
        report["errors"].append(f"Labels without images: {len(orphan_labels)} ({', '.join(orphan_labels[:10])})")

    seen_hashes = defaultdict(list)
    valid_boxes = []
    no_box = []
    duplicate_images = []
    orientations = Counter()
    aspect_bins = Counter()

    for img in images:
        w, h, fmt = image_size(img)
        item = {"image": img.name, "width": w, "height": h, "format": fmt, "label": None, "errors": [], "warnings": []}
        if w is None:
            item["errors"].append(fmt)
            report["files"].append(item)
            continue
        orientations["landscape" if w > h else "portrait" if h > w else "square"] += 1
        ratio = w / h
        if ratio >= 1.5:
            aspect_bins["wide"] += 1
        elif ratio <= 2/3:
            aspect_bins["tall"] += 1
        else:
            aspect_bins["normal"] += 1
        digest = sha1_file(img)
        seen_hashes[digest].append(img.name)
        lp = label_stems.get(img.stem)
        if lp:
            item["label"] = lp.name
            rows, errs = parse_label(lp)
            item["errors"].extend(errs)
            if len(rows) == 0:
                no_box.append(img.name)
            elif len(rows) == 1 and not errs:
                valid_boxes.append(img.name)
                _, cx, cy, bw, bh = rows[0]
                item["bbox_normalized"] = [cx, cy, bw, bh]
                item["bbox_percent"] = [round(100*cx,2), round(100*cy,2), round(100*bw,2), round(100*bh,2)]
                if bw > 0.98 or bh > 0.98:
                    item["warnings"].append("bbox nearly spans the whole image")
                if bw < 0.05 or bh < 0.05:
                    item["warnings"].append("bbox is unusually small")
            elif len(rows) > 1:
                item["warnings"].append(f"{len(rows)} boxes in one label; current training setup expects one target box")
        report["files"].append(item)

    for names in seen_hashes.values():
        if len(names) > 1:
            duplicate_images.append(names)
    if duplicate_images:
        report["warnings"].append(f"Exact duplicate image content groups: {len(duplicate_images)}")

    ann_path = root / "annotations.json"
    if ann_path.exists():
        try:
            ann = json.loads(ann_path.read_text(encoding="utf-8"))
            annotations = ann.get("annotations", {})
            if not isinstance(annotations, dict):
                report["errors"].append("annotations.json: annotations is not an object")
            else:
                ann_missing = sorted(set(image_stems) - {Path(k).stem for k in annotations})
                if ann_missing:
                    report["warnings"].append(f"annotations.json has no entry for {len(ann_missing)} image(s)")
        except Exception as exc:
            report["errors"].append(f"annotations.json cannot be parsed: {exc}")
    else:
        report["warnings"].append("annotations.json is missing")

    yaml_path = root / "dataset.yaml"
    if not yaml_path.exists():
        report["errors"].append("Missing dataset.yaml")
    else:
        ytxt = yaml_path.read_text(encoding="utf-8-sig")
        for required in ("train:", "val:", "names:"):
            if required not in ytxt:
                report["errors"].append(f"dataset.yaml missing '{required}'")

    report["stats"] = {
        "images": len(images),
        "labels": len(labels),
        "valid_single_box_labels": len(valid_boxes),
        "images_without_box": len(no_box),
        "orientation": dict(orientations),
        "aspect_groups": dict(aspect_bins),
        "duplicate_groups": len(duplicate_images),
    }
    report["ok"] = not report["errors"] and len(images) > 0 and len(valid_boxes) == len(images)
    if len(images) < 20:
        report["warnings"].append("Dataset is small; first training run should be treated as a prototype.")
    if len(valid_boxes) >= 1:
        widths = [f["bbox_normalized"][2] for f in report["files"] if "bbox_normalized" in f]
        heights = [f["bbox_normalized"][3] for f in report["files"] if "bbox_normalized" in f]
        report["stats"]["bbox_width_mean"] = round(sum(widths)/len(widths), 4)
        report["stats"]["bbox_height_mean"] = round(sum(heights)/len(heights), 4)
        report["stats"]["bbox_width_min"] = round(min(widths), 4)
        report["stats"]["bbox_width_max"] = round(max(widths), 4)
        report["stats"]["bbox_height_min"] = round(min(heights), 4)
        report["stats"]["bbox_height_max"] = round(max(heights), 4)

    (out / "validation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = []
    lines.append("CoreBoxCropper — ML DATASET VALIDATION")
    lines.append("=" * 60)
    lines.append(f"Dataset: {root}")
    lines.append(f"RESULT: {'OK — можно переходить к подготовке обучения' if report['ok'] else 'НЕ ГОТОВ — сначала исправить ошибки'}")
    lines.append("")
    lines.append("СТАТИСТИКА")
    for k, v in report["stats"].items():
        lines.append(f"  {k}: {v}")
    if report["errors"]:
        lines.append("\nОШИБКИ")
        lines.extend(f"  [ERROR] {x}" for x in report["errors"])
    if report["warnings"]:
        lines.append("\nПРЕДУПРЕЖДЕНИЯ")
        lines.extend(f"  [WARN] {x}" for x in report["warnings"])
    bad = [f for f in report["files"] if f["errors"]]
    warn = [f for f in report["files"] if f["warnings"]]
    if bad:
        lines.append("\nПРОБЛЕМНЫЕ ФАЙЛЫ")
        for f in bad:
            lines.append(f"  {f['image']}: {'; '.join(f['errors'])}")
    if warn:
        lines.append("\nФАЙЛЫ С ПРЕДУПРЕЖДЕНИЯМИ")
        for f in warn:
            lines.append(f"  {f['image']}: {'; '.join(f['warnings'])}")
    lines.append("\nПодробности: _validation/validation_report.json")
    (out / "validation_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
