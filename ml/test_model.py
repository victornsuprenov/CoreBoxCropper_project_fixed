from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    print("Ultralytics is not installed. Run: python -m pip install ultralytics")
    raise SystemExit(2)

# Allow running directly from the project root: python .\\ml\\test_model.py ...
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from coreboxcropper.config import DetectorConfig
from coreboxcropper.ruler_detector import RulerDetector

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def parse_args():
    p = argparse.ArgumentParser(
        description="Fast external test of YOLO CoreBoxCropper with optional local ruler-aware selection."
    )
    p.add_argument("source", type=Path, help="Folder with external test photos")
    p.add_argument("--weights", type=Path, default=None, help="Path to best.pt")
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--conf", type=float, default=0.20,
                   help="YOLO confidence threshold for collecting candidates (default: 0.20)")
    p.add_argument("--device", default="cpu")
    p.add_argument("--output", type=Path, default=None,
                   help="Output folder. Default: <source>/_ml_test_fast")
    p.add_argument("--local-ruler-max-dim", type=int, default=1800,
                   help="Max dimension for each local ruler-search ROI")
    p.add_argument("--local-ruler-pad-x", type=float, default=0.16,
                   help="Horizontal padding around a YOLO candidate for local ruler search")
    p.add_argument("--local-ruler-pad-y", type=float, default=0.24,
                   help="Vertical padding around a YOLO candidate for local ruler search")
    p.add_argument("--ruler-bonus", type=float, default=0.40,
                   help="Maximum score contribution of a locally detected ruler")
    p.add_argument("--ruler-rotate", action="store_true",
                   help="If local 0-degree ruler search fails, also try a 90-degree rotation")
    p.add_argument("--save-crops", action="store_true",
                   help="Also save the selected YOLO bbox crop")
    return p.parse_args()


def default_weights() -> Path:
    project = Path(__file__).resolve().parents[1]
    candidates = [
        project / "runs" / "detect" / "ml" / "runs" / "corebox_v2-2" / "weights" / "best.pt",
    ]
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]


def collect_images(source: Path):
    if not source.exists() or not source.is_dir():
        raise FileNotFoundError(f"Test folder not found: {source}")
    return sorted(p for p in source.iterdir()
                  if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def rect_area(r):
    x1, y1, x2, y2 = r
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def intersection(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return (x2 - x1) * (y2 - y1)


def center_distance_score(box, image_shape):
    h, w = image_shape[:2]
    cx = (box[0] + box[2]) * 0.5 / max(w, 1)
    cy = (box[1] + box[3]) * 0.5 / max(h, 1)
    d = math.hypot(cx - 0.5, cy - 0.5) / math.sqrt(0.5 ** 2 + 0.5 ** 2)
    return float(np.clip(1.0 - d, 0.0, 1.0))


def expand_box(box, image_shape, pad_x: float, pad_y: float):
    h, w = image_shape[:2]
    x1, y1, x2, y2 = box
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    return (
        max(0.0, x1 - bw * pad_x),
        max(0.0, y1 - bh * pad_y),
        min(float(w), x2 + bw * pad_x),
        min(float(h), y2 + bh * pad_y),
    )


def crop_rect(image, rect):
    x1, y1, x2, y2 = [int(round(v)) for v in rect]
    h, w = image.shape[:2]
    x1, y1 = max(0, min(x1, w - 1)), max(0, min(y1, h - 1))
    x2, y2 = max(x1 + 1, min(x2, w)), max(y1 + 1, min(y2, h))
    return image[y1:y2, x1:x2].copy(), (x1, y1, x2, y2)


def resize_for_search(image, max_dim: int):
    h, w = image.shape[:2]
    scale = min(1.0, float(max_dim) / max(h, w))
    if scale >= 0.999:
        return image, 1.0
    return cv2.resize(image, (max(1, int(round(w * scale))), max(1, int(round(h * scale)))), interpolation=cv2.INTER_AREA), scale


def map_local_rect_to_image(local_rect, crop_rect_xyxy, scale):
    x1, y1, x2, y2 = local_rect
    cx1, cy1, _, _ = crop_rect_xyxy
    inv = 1.0 / max(scale, 1e-9)
    return (
        cx1 + x1 * inv,
        cy1 + y1 * inv,
        cx1 + x2 * inv,
        cy1 + y2 * inv,
    )


def local_ruler_search(image, candidate_box, detector, args):
    """Search for the ruler only around one YOLO candidate.

    This is intentionally local. The previous test searched the whole image in
    four orientations, which is diagnostic but far too expensive for production.
    Returns (mapped ruler bounds, confidence, elapsed_ms, orientation_used).
    """
    roi = expand_box(candidate_box, image.shape, args.local_ruler_pad_x, args.local_ruler_pad_y)
    local, crop_xyxy = crop_rect(image, roi)
    small, scale = resize_for_search(local, args.local_ruler_max_dim)

    started = time.perf_counter()
    attempts = [(0, small)]
    if args.ruler_rotate:
        attempts.append((90, cv2.rotate(small, cv2.ROTATE_90_CLOCKWISE)))

    best = None
    for angle, variant in attempts:
        rd = detector.detect_global(variant)
        if not (rd.found and rd.bounds):
            continue
        candidate = (float(rd.confidence), rd, angle)
        if best is None or candidate[0] > best[0]:
            best = candidate

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    if best is None:
        return None, 0.0, elapsed_ms, None

    confidence, rd, angle = best
    x1, y1, x2, y2 = map(float, rd.bounds)
    if angle == 0:
        mapped = map_local_rect_to_image((x1, y1, x2, y2), crop_xyxy, scale)
    else:
        # For the optional 90-degree probe, rotate the detected rectangle back
        # into the local 0-degree coordinates, then into image coordinates.
        rh, rw = small.shape[:2]
        pts = [
            (rw - 1 - y1, x1),
            (rw - 1 - y2, x1),
            (rw - 1 - y2, x2),
            (rw - 1 - y1, x2),
        ]
        local_rect = (min(p[0] for p in pts), min(p[1] for p in pts),
                      max(p[0] for p in pts), max(p[1] for p in pts))
        mapped = map_local_rect_to_image(local_rect, crop_xyxy, scale)
    return mapped, confidence, elapsed_ms, angle


def candidate_metrics(candidate, ruler_bounds, image_shape):
    box = candidate["box"]
    if ruler_bounds:
        ruler_area = rect_area(ruler_bounds)
        containment = intersection(box, ruler_bounds) / ruler_area if ruler_area > 0 else 0.0
        overlap = intersection(box, ruler_bounds) / rect_area(box) if rect_area(box) > 0 else 0.0
        ruler_score = 0.70 * containment + 0.30 * overlap
    else:
        containment = 0.0
        overlap = 0.0
        ruler_score = 0.0
    center_score = center_distance_score(box, image_shape)
    return containment, overlap, ruler_score, center_score


def choose_candidates(candidates, image, ruler_by_candidate, ruler_bonus):
    scored = []
    for idx, c in enumerate(candidates):
        ruler_bounds, ruler_conf, ruler_ms, angle = ruler_by_candidate.get(idx, (None, 0.0, 0.0, None))
        containment, overlap, ruler_score, center_score = candidate_metrics(c, ruler_bounds, image.shape)
        score = 0.75 * c["confidence"] + ruler_bonus * ruler_score + 0.05 * center_score
        item = dict(c)
        item.update({
            "score": score,
            "ruler_bounds": ruler_bounds,
            "ruler_confidence": ruler_conf,
            "ruler_containment": containment,
            "ruler_overlap": overlap,
            "ruler_search_ms": ruler_ms,
            "ruler_angle": angle,
        })
        scored.append(item)
    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored


def draw_result(image, scored):
    out = image.copy()
    if not scored:
        return out

    selected = scored[0]
    for rank, c in enumerate(scored, start=1):
        x1, y1, x2, y2 = map(int, c["box"])
        is_selected = c is selected
        color = (0, 220, 0) if is_selected else (0, 140, 255)
        thickness = 5 if is_selected else 2
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        rb = c.get("ruler_bounds")
        if rb:
            rx1, ry1, rx2, ry2 = map(int, rb)
            cv2.rectangle(out, (rx1, ry1), (rx2, ry2), (255, 180, 0), 2)
        text = (
            f"#{rank} conf={c['confidence']:.2f} score={c['score']:.2f} "
            f"ruler={c['ruler_containment']:.2f}"
        )
        ty = max(25, y1 - 8)
        cv2.putText(out, text, (x1, ty), cv2.FONT_HERSHEY_SIMPLEX,
                    0.62, color, 2, cv2.LINE_AA)
    return out


def save_crop(image, box, path: Path):
    crop, _ = crop_rect(image, box)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), crop)


def main():
    args = parse_args()
    source = args.source.resolve()
    weights = (args.weights or default_weights()).resolve()

    if not weights.exists():
        print(f"ERROR: best.pt not found: {weights}")
        return 2

    images = collect_images(source)
    if not images:
        print(f"ERROR: no images in {source}")
        return 2

    output = (args.output or (source / "_ml_test_fast")).resolve()
    predictions = output / "predictions"
    crops = output / "crops"
    output.mkdir(parents=True, exist_ok=True)
    predictions.mkdir(parents=True, exist_ok=True)
    if args.save_crops:
        crops.mkdir(parents=True, exist_ok=True)

    csv_path = output / "report.csv"
    summary_path = output / "summary.txt"

    print("=" * 70)
    print("CoreBoxCropper ML external test — FAST local selection")
    print("=" * 70)
    print(f"Model : {weights}")
    print(f"Source: {source}")
    print(f"Images: {len(images)}")
    print(f"Conf  : {args.conf}")
    print("Mode  : YOLO first; local ruler search only for multiple candidates")
    print()

    model = YOLO(str(weights))
    ruler_detector = RulerDetector(DetectorConfig())

    rows = []
    total_start = time.perf_counter()
    ruler_calls = 0

    for idx, path in enumerate(images, 1):
        started = time.perf_counter()
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            rows.append({"file": path.name, "status": "ERROR", "error": "image_read_failed"})
            print(f"[{idx:>3}/{len(images)}] ERROR {path.name}")
            continue

        try:
            yolo_started = time.perf_counter()
            results = model.predict(
                source=image,
                imgsz=args.imgsz,
                conf=args.conf,
                device=args.device,
                verbose=False,
            )
            yolo_ms = (time.perf_counter() - yolo_started) * 1000.0
            result = results[0]
            boxes = result.boxes

            candidates = []
            if boxes is not None and len(boxes):
                xyxy = boxes.xyxy.detach().cpu().numpy()
                confs = boxes.conf.detach().cpu().numpy()
                for j, (b, conf) in enumerate(zip(xyxy, confs)):
                    candidates.append({
                        "box": tuple(float(v) for v in b),
                        "confidence": float(conf),
                        "index": j,
                    })

            if not candidates:
                elapsed = (time.perf_counter() - started) * 1000.0
                rows.append({
                    "file": path.name, "status": "NOT_FOUND", "detections": 0,
                    "selected_confidence": "", "selected_score": "",
                    "ruler_confidence": "", "ruler_containment": "", "ruler_overlap": "",
                    "ruler_calls": 0, "ruler_search_ms": 0.0, "yolo_ms": round(yolo_ms, 1),
                    "x1": "", "y1": "", "x2": "", "y2": "",
                    "width": "", "height": "", "total_ms": round(elapsed, 1), "error": "",
                })
                cv2.imwrite(str(predictions / path.name), image)
                print(f"[{idx:>3}/{len(images)}] NOT_FOUND {path.name} time={elapsed:.0f}ms")
                continue

            ruler_by_candidate = {}
            # Critical speed rule: a single YOLO candidate does NOT trigger the
            # expensive ruler detector. We only need context to disambiguate
            # competing candidates.
            if len(candidates) > 1:
                for candidate_idx, candidate in enumerate(candidates):
                    ruler_calls += 1
                    rb, rc, rms, angle = local_ruler_search(
                        image, candidate["box"], ruler_detector, args
                    )
                    ruler_by_candidate[candidate_idx] = (rb, rc, rms, angle)

            # If there was exactly one candidate, it is selected directly. For
            # multiple candidates, use the local ruler-aware score.
            if len(candidates) == 1:
                c = dict(candidates[0])
                c.update({
                    "score": c["confidence"],
                    "ruler_bounds": None,
                    "ruler_confidence": 0.0,
                    "ruler_containment": 0.0,
                    "ruler_overlap": 0.0,
                    "ruler_search_ms": 0.0,
                    "ruler_angle": None,
                })
                scored = [c]
            else:
                scored = choose_candidates(candidates, image, ruler_by_candidate, args.ruler_bonus)

            selected = scored[0]
            x1, y1, x2, y2 = selected["box"]
            elapsed = (time.perf_counter() - started) * 1000.0
            total_ruler_ms = sum(float(c.get("ruler_search_ms", 0.0)) for c in scored)

            rows.append({
                "file": path.name,
                "status": "FOUND",
                "detections": len(candidates),
                "selected_confidence": round(selected["confidence"], 5),
                "selected_score": round(selected["score"], 5),
                "ruler_confidence": round(selected["ruler_confidence"], 5),
                "ruler_containment": round(selected["ruler_containment"], 5),
                "ruler_overlap": round(selected["ruler_overlap"], 5),
                "ruler_calls": len(candidates) if len(candidates) > 1 else 0,
                "ruler_search_ms": round(total_ruler_ms, 1),
                "yolo_ms": round(yolo_ms, 1),
                "x1": round(x1, 1), "y1": round(y1, 1),
                "x2": round(x2, 1), "y2": round(y2, 1),
                "width": round(x2 - x1, 1), "height": round(y2 - y1, 1),
                "total_ms": round(elapsed, 1), "error": "",
            })

            out = draw_result(image, scored)
            cv2.imwrite(str(predictions / path.name), out)
            if args.save_crops:
                save_crop(image, selected["box"], crops / path.name)

            print(
                f"[{idx:>3}/{len(images)}] FOUND {path.name} "
                f"conf={selected['confidence']:.2f} "
                f"candidates={len(candidates)} "
                f"ruler_calls={len(candidates) if len(candidates) > 1 else 0} "
                f"time={elapsed:.0f}ms"
            )

        except Exception as exc:
            elapsed = (time.perf_counter() - started) * 1000.0
            rows.append({
                "file": path.name, "status": "ERROR", "detections": "",
                "selected_confidence": "", "selected_score": "",
                "ruler_confidence": "", "ruler_containment": "", "ruler_overlap": "",
                "ruler_calls": "", "ruler_search_ms": "", "yolo_ms": "",
                "x1": "", "y1": "", "x2": "", "y2": "",
                "width": "", "height": "", "total_ms": round(elapsed, 1),
                "error": repr(exc),
            })
            print(f"[{idx:>3}/{len(images)}] ERROR {path.name}: {exc}")

    fields = [
        "file", "status", "detections", "selected_confidence", "selected_score",
        "ruler_confidence", "ruler_containment", "ruler_overlap", "ruler_calls",
        "ruler_search_ms", "yolo_ms", "x1", "y1", "x2", "y2", "width", "height",
        "total_ms", "error"
    ]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    found = sum(r.get("status") == "FOUND" for r in rows)
    nf = sum(r.get("status") == "NOT_FOUND" for r in rows)
    errors = sum(r.get("status") == "ERROR" for r in rows)
    multi = sum(int(r["detections"]) > 1 for r in rows if str(r.get("detections", "")).isdigit())
    confs = [float(r["selected_confidence"]) for r in rows if r.get("selected_confidence") not in (None, "")]
    times = [float(r["total_ms"]) for r in rows if r.get("total_ms") not in (None, "")]
    yolo_times = [float(r["yolo_ms"]) for r in rows if r.get("yolo_ms") not in (None, "")]
    ruler_times = [float(r["ruler_search_ms"]) for r in rows if r.get("ruler_search_ms") not in (None, "")]

    total = time.perf_counter() - total_start
    avg = lambda values: (sum(values) / len(values)) if values else 0.0
    summary = f"""CoreBoxCropper external ML test — FAST
=======================================
Mode: YOLO first; local ruler search only when there are multiple candidates
Images: {len(images)}
Found: {found}
Not found: {nf}
Errors: {errors}
Multiple YOLO candidates: {multi}
Detection coverage: {found / len(images) * 100:.1f}%
Average selected confidence: {avg(confs):.4f}
Minimum selected confidence: {min(confs) if confs else 0:.4f}
Average total time: {avg(times):.1f} ms/image
Average YOLO time: {avg(yolo_times):.1f} ms/image
Average local ruler search time: {avg(ruler_times):.1f} ms/image
Total local ruler calls: {ruler_calls}
Total wall time: {total:.1f} s

Speed strategy:
- one YOLO candidate -> select directly, no global ruler search;
- multiple candidates -> local ruler search inside each candidate ROI;
- optional 90° local retry only with --ruler-rotate;
- no four full-frame orientation passes.

Review _ml_test_fast\\predictions visually. This is still a test harness,
not the final production cropper.
"""
    summary_path.write_text(summary, encoding="utf-8")

    print("\n" + "=" * 70)
    print("RESULT")
    print("=" * 70)
    print(f"Found                 : {found}/{len(images)} ({found/len(images)*100:.1f}%)")
    print(f"Not found             : {nf}")
    print(f"Errors                : {errors}")
    print(f"Multiple candidates   : {multi}")
    print(f"Avg selected conf     : {avg(confs):.4f}")
    print(f"Min selected conf     : {min(confs) if confs else 0:.4f}")
    print(f"Avg time              : {avg(times):.1f} ms/image")
    print(f"Avg YOLO time         : {avg(yolo_times):.1f} ms/image")
    print(f"Avg local ruler time  : {avg(ruler_times):.1f} ms/image")
    print(f"Ruler calls           : {ruler_calls}")
    print(f"Predictions           : {predictions}")
    print(f"CSV                   : {csv_path}")
    print(f"Summary               : {summary_path}")
    if args.save_crops:
        print(f"Crops                 : {crops}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
