#!/usr/bin/env python3
"""Train and export the four-corner CoreBoxCropper pose model."""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the CoreBoxCropper 4-keypoint model")
    parser.add_argument("data", type=Path, help="keypoint_dataset/dataset.yaml")
    parser.add_argument("--weights", default="yolo11n-pose.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="cpu", help="cpu, 0, or another Ultralytics device")
    parser.add_argument("--project", type=Path, default=Path("runs/pose"))
    parser.add_argument("--name", default="corebox_keypoints_v3")
    parser.add_argument("--export-onnx", action="store_true")
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()
    if args.epochs < 1 or args.imgsz < 320 or args.batch < 1:
        raise SystemExit("epochs, imgsz and batch must be positive; imgsz must be at least 320")
    if not args.data.exists():
        raise SystemExit(f"Dataset YAML not found: {args.data}")
    project = args.project.expanduser().resolve()

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Install training dependencies first: python -m pip install ultralytics") from exc

    model = YOLO(args.weights)
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(project),
        name=args.name,
        pretrained=True,
        workers=0,
        patience=30,
        seed=20260917,
    )
    best = project / args.name / "weights" / "best.pt"
    print(f"Best weights: {best}")
    if args.export_onnx:
        exported = YOLO(str(best)).export(format="onnx", opset=args.opset, simplify=True, dynamic=False)
        print(f"ONNX model: {exported}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
