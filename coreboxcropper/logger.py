from __future__ import annotations

import logging
from pathlib import Path

from .config import app_data_dir


def get_logger() -> logging.Logger:
    logger = logging.getLogger("coreboxcropper")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    log_dir = app_data_dir() / "logs"
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(log_dir / "coreboxcropper.log", encoding="utf-8")
    except OSError:
        handler = logging.NullHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)
    return logger


def get_debug_dir(base: Path, input_path: Path) -> Path:
    directory = base / input_path.stem
    directory.mkdir(parents=True, exist_ok=True)
    return directory
