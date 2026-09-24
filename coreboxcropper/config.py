from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


def app_data_dir() -> Path:
    """Return a writable per-user directory on Windows and other platforms."""
    if os.name == "nt":
        root = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
        if root:
            return Path(root) / "CoreBoxCropper"
    return Path.home() / ".coreboxcropper"


@dataclass
class DetectorConfig:
    max_analysis_dimension: int = 5000
    canny_threshold_1: int = 50
    canny_threshold_2: int = 150
    hough_threshold: int = 45
    min_line_length_ratio: float = 0.16
    max_line_gap_ratio: float = 0.025
    min_box_area_ratio: float = 0.08
    max_box_area_ratio: float = 0.88
    min_aspect_ratio: float = 1.15
    max_aspect_ratio: float = 8.5
    confidence_threshold: float = 0.64
    crop_margin_percent: float = 0.01
    cell_crop_margin_percent: float = 0.0
    ruler_search_margin: float = 0.30
    ruler_signature_enabled: bool = True
    ruler_tick_step_cm: int = 10
    ruler_length_cm: int = 100
    ruler_color_min_saturation: int = 70
    ruler_signature_weight: float = 0.34
    ruler_signature_priority_threshold: float = 0.80
    ruler_labels_required: bool = True
    expected_core_cells: int = 3
    expected_box_aspect_ratio: float = 4.0
    box_aspect_tolerance: float = 0.75
    perspective_threshold: float = 0.055

    # Three-cell rail detector. Values are relative to the image width/height.
    target_search_left_ratio: float = 0.05
    target_search_right_ratio: float = 0.92
    target_center_ratio: float = 0.50
    target_center_sigma: float = 0.18
    preferred_box_width_ratio: float = 0.31
    min_box_width_ratio: float = 0.22
    max_box_width_ratio: float = 0.40
    min_cell_spacing_ratio: float = 0.065
    max_cell_spacing_ratio: float = 0.125
    max_spacing_regularity: float = 0.22
    rail_min_distance_ratio: float = 0.018
    rail_peak_percentile: float = 48.0
    rail_strength_reference: float = 1.0
    hough_min_vertical_length_ratio: float = 0.38
    hough_max_vertical_gap_ratio: float = 0.035
    ruler_box_search_width_ratio: float = 0.72
    ruler_box_gap_ratio_min: float = 0.002
    ruler_box_overlap_search_ratio: float = 0.018
    ruler_box_overlap_tolerance: float = 0.020
    ruler_group_confidence_threshold: float = 0.52
    geometry_group_confidence_threshold: float = 0.54
    geometry_fallback_success_threshold: float = 0.50
    expected_ruler_width_ratio: float = 0.09
    target_box_position: str = "leftmost"
    target_min_x_ratio: float = 0.12
    jpeg_quality: int = 95

    # Fast ML-first path. Empty weight paths resolve relative to the project.
    yolo_enabled: bool = True
    yolo_primary_weights: str = ""
    yolo_fallback_weights: str = ""
    yolo_imgsz: int = 960
    yolo_fallback_imgsz: int = 1280
    yolo_confidence: float = 0.20
    yolo_iou: float = 0.50
    yolo_max_detections: int = 8
    yolo_roi_pad_x: float = 0.12
    yolo_roi_pad_y: float = 0.20
    yolo_local_max_dimension: int = 1800
    yolo_legacy_fallback: bool = True

    # V3 production pipeline: optional staged optimizer that ranks YOLO
    # candidates, validates a local quad, and gates the final crop before
    # perspective warping. The default remains off so the stable production
    # pipeline is unchanged until a trained key-point model is available.
    v3_enabled: bool = False
    v3_candidate_limit: int = 4
    v3_quality_gate_min_confidence: float = 0.75
    v3_perspective_min_score: float = 0.45
    v3_keypoint_model: str = ""

    def validate(self) -> None:
        if self.max_analysis_dimension < 400:
            raise ValueError("max_analysis_dimension must be at least 400")
        if not 0 < self.min_box_area_ratio < self.max_box_area_ratio <= 1:
            raise ValueError("box area ratios must be between 0 and 1")
        if not 0 < self.min_aspect_ratio < self.max_aspect_ratio:
            raise ValueError("aspect ratio limits are invalid")
        if not 0 <= self.crop_margin_percent <= 0.25:
            raise ValueError("crop margin must be between 0 and 25 percent")
        if not 0 <= self.cell_crop_margin_percent <= 0.25:
            raise ValueError("three-cell crop margin must be between 0 and 25 percent")
        if not 0 <= self.confidence_threshold <= 1:
            raise ValueError("confidence threshold must be between 0 and 1")
        if self.ruler_tick_step_cm <= 0 or self.ruler_length_cm <= 0:
            raise ValueError("ruler dimensions must be positive")
        if self.ruler_tick_step_cm >= self.ruler_length_cm:
            raise ValueError("ruler tick step must be smaller than ruler length")
        if not 0 <= self.ruler_color_min_saturation <= 255:
            raise ValueError("ruler color saturation must be between 0 and 255")
        if not 0 <= self.ruler_signature_weight <= 1:
            raise ValueError("ruler signature weight must be between 0 and 1")
        if not 0 <= self.ruler_signature_priority_threshold <= 1:
            raise ValueError("ruler signature priority threshold must be between 0 and 1")
        if self.expected_core_cells < 1:
            raise ValueError("expected core cell count must be positive")
        if self.expected_box_aspect_ratio <= 0:
            raise ValueError("expected_box_aspect_ratio must be positive")
        if not 0.1 <= self.box_aspect_tolerance <= 3.0:
            raise ValueError("box_aspect_tolerance is invalid")
        if not 0 <= self.target_min_x_ratio < 0.5:
            raise ValueError("target_min_x_ratio must be between 0 and 50 percent")
        if self.target_box_position not in {"leftmost", "topmost", "first", "configured"}:
            raise ValueError("unknown target_box_position")
        if not 0 < self.target_search_left_ratio < self.target_search_right_ratio <= 1:
            raise ValueError("target search ratios are invalid")
        if not 0 < self.min_box_width_ratio < self.max_box_width_ratio < 1:
            raise ValueError("box width ratios are invalid")
        if not self.min_cell_spacing_ratio < self.max_cell_spacing_ratio < 0.5:
            raise ValueError("cell spacing ratios are invalid")
        if not 0.05 <= self.max_spacing_regularity <= 0.5:
            raise ValueError("max_spacing_regularity is invalid")
        if not 0 < self.expected_ruler_width_ratio < 0.25:
            raise ValueError("expected_ruler_width_ratio is invalid")
        if not 0 < self.ruler_box_overlap_search_ratio <= 0.05:
            raise ValueError("ruler_box_overlap_search_ratio is invalid")
        if not 0 < self.ruler_box_overlap_tolerance <= 0.08:
            raise ValueError("ruler_box_overlap_tolerance is invalid")
        if self.yolo_imgsz < 320 or self.yolo_fallback_imgsz < 320:
            raise ValueError("YOLO image size must be at least 320")
        if not 0 < self.yolo_confidence < 1 or not 0 < self.yolo_iou < 1:
            raise ValueError("YOLO thresholds must be between 0 and 1")
        if self.yolo_max_detections < 1:
            raise ValueError("YOLO max detections must be positive")
        if not 0 <= self.yolo_roi_pad_x <= 1 or not 0 <= self.yolo_roi_pad_y <= 1:
            raise ValueError("YOLO ROI padding must be between 0 and 1")
        if self.yolo_local_max_dimension < 400:
            raise ValueError("YOLO local refinement dimension must be at least 400")
        if self.v3_candidate_limit < 1:
            raise ValueError("V3 candidate limit must be positive")
        if not 0 <= self.v3_quality_gate_min_confidence <= 1:
            raise ValueError("V3 quality gate confidence must be between 0 and 1")
        if not 0 <= self.v3_perspective_min_score <= 1:
            raise ValueError("V3 perspective score must be between 0 and 1")


def config_path() -> Path:
    return app_data_dir() / "config.json"


def default_v3_model_path() -> Path:
    return Path(__file__).resolve().parents[1] / "runs" / "pose" / "corebox_keypoints_v2" / "weights" / "best.onnx"


def load_config(path: Path | None = None) -> DetectorConfig:
    config = DetectorConfig()
    selected = path or config_path()
    values: dict[str, Any] = {}
    if selected.exists():
        try:
            loaded = json.loads(selected.read_text(encoding="utf-8"))
            values = loaded
            if not isinstance(values, dict):
                raise ValueError("configuration root must be an object")
            config = DetectorConfig(**{k: v for k, v in values.items() if k in asdict(config)})
            config.validate()
        except (OSError, ValueError, TypeError, AttributeError):
            # A corrupt user config must not prevent the application from opening.
            config = DetectorConfig()
            values = {}
    # GUI/CLI use the latest trained V3 model automatically. An explicit false
    # remains an escape hatch for diagnostics and rollback.
    if "v3_enabled" not in values and default_v3_model_path().exists():
        config.v3_enabled = True
    config.validate()
    return config


def save_config(config: DetectorConfig, path: Path | None = None) -> Path:
    config.validate()
    selected = path or config_path()
    selected.parent.mkdir(parents=True, exist_ok=True)
    selected.write_text(json.dumps(asdict(config), ensure_ascii=False, indent=2), encoding="utf-8")
    return selected


def update_config(config: DetectorConfig, values: dict[str, Any]) -> DetectorConfig:
    merged = asdict(config)
    merged.update({key: value for key, value in values.items() if key in merged})
    updated = DetectorConfig(**merged)
    updated.validate()
    return updated
