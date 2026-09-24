#!/usr/bin/env python3
"""Prepare reproducible train/validation split after dataset validation."""
from __future__ import annotations
import argparse, json, random, shutil
from pathlib import Path

EXTS={'.jpg','.jpeg','.png','.bmp','.webp','.tif','.tiff'}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('dataset', type=Path)
    ap.add_argument('--val-ratio', type=float, default=0.20)
    ap.add_argument('--seed', type=int, default=20260910)
    ap.add_argument('--output', type=Path, default=None)
    args=ap.parse_args()
    root=args.dataset.expanduser().resolve()
    output=(args.output or (root/'yolo_dataset')).resolve()
    images=sorted(p for p in (root/'images').glob('*') if p.is_file() and p.suffix.lower() in EXTS)
    labels=root/'labels'
    missing=[p.stem for p in images if not (labels/f'{p.stem}.txt').exists()]
    if missing: raise SystemExit(f'Missing labels for {len(missing)} images; run validate_dataset.py first.')
    if not 0.05 <= args.val_ratio <= 0.4: raise SystemExit('--val-ratio must be between 0.05 and 0.4')
    rng=random.Random(args.seed); rng.shuffle(images)
    nval=max(1, round(len(images)*args.val_ratio))
    val=images[:nval]; train=images[nval:]
    if not train: raise SystemExit('Not enough images for train split')
    for split,items in [('train',train),('val',val)]:
        (output/'images'/split).mkdir(parents=True,exist_ok=True)
        (output/'labels'/split).mkdir(parents=True,exist_ok=True)
        for p in items:
            shutil.copy2(p, output/'images'/split/p.name)
            shutil.copy2(labels/f'{p.stem}.txt', output/'labels'/split/f'{p.stem}.txt')
    yaml=(output/'dataset.yaml')
    yaml.write_text(f"path: {output.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n  0: target_corebox\n",encoding='utf-8')
    manifest={'seed':args.seed,'val_ratio':args.val_ratio,'train':[p.name for p in train],'val':[p.name for p in val]}
    (output/'split_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'Prepared: {output}')
    print(f'Train: {len(train)} | Val: {len(val)} | Seed: {args.seed}')

if __name__=='__main__': main()
