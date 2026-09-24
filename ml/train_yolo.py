#!/usr/bin/env python3
"""Train the first local CoreBoxCropper detector.

This script is intentionally separate from the GUI. Training is a development
operation; the final application will ship only the trained ONNX model.

Requires: ultralytics (training machine only).
"""
from __future__ import annotations
import argparse
from pathlib import Path

def main():
    ap=argparse.ArgumentParser(description='Train target_corebox detector')
    ap.add_argument('data', type=Path, help='dataset.yaml from prepare_training.py')
    ap.add_argument('--epochs', type=int, default=80)
    ap.add_argument('--imgsz', type=int, default=960)
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--project', type=Path, default=Path('ml/runs'))
    ap.add_argument('--name', default='corebox_v3')
    args=ap.parse_args()
    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit('Ultralytics is not installed. Install it only on the training PC: pip install ultralytics')
    # The initial model is a small YOLO detector. The pretrained COCO weights
    # are only an initialization; the resulting weights are trained on DTM.
    model=YOLO('yolo11n.pt')
    model.train(data=str(args.data), epochs=args.epochs, imgsz=args.imgsz,
                batch=args.batch, device=args.device, project=str(args.project),
                name=args.name, pretrained=True, workers=0)
    print('Training finished. Best weights are in the run directory.')

if __name__=='__main__': main()
