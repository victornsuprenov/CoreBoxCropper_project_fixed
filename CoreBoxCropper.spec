# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all


datas = []
binaries = []
hiddenimports = []
for package in ("cv2", "PIL", "numpy"):
    d, b, h = collect_all(package)
    datas += d
    binaries += b
    hiddenimports += h

# Ship only the current production models with the onedir build.
datas += [
    ("runs/detect/ml/runs/corebox_v2-2/weights/best.pt", "runs/detect/ml/runs/corebox_v2-2/weights"),
    ("runs/pose/corebox_keypoints_v2/weights/best.onnx", "runs/pose/corebox_keypoints_v2/weights"),
]


analysis = Analysis(
    ["main.py"],
    pathex=["."],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=["matplotlib", "pandas", "scipy"],
)

pyz = PYZ(analysis.pure)

exe = EXE(
    pyz,
    analysis.scripts,
    [],
    exclude_binaries=True,
    name="CoreBoxCropper",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
)

coll = COLLECT(
    exe,
    analysis.binaries,
    analysis.datas,
    strip=False,
    upx=True,
    name="CoreBoxCropper",
)
