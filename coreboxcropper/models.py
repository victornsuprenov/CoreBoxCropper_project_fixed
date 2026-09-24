from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

Point = tuple[float, float]
Rect = tuple[int, int, int, int]


@dataclass
class RulerDetection:
    found: bool = False
    bounds: Rect | None = None
    confidence: float = 0.0
    reason: str = "not_found"
    color_scale: Rect | None = None
    color_scale_confidence: float = 0.0
    tick_score: float = 0.0
    label_region: Rect | None = None


@dataclass
class DetectionResult:
    success: bool
    confidence: float
    corners: list[Point] = field(default_factory=list)
    bbox: Rect | None = None
    ruler: Rect | None = None
    ruler_confidence: float = 0.0
    final_roi: Rect | None = None
    reason: str = ""
    candidate_scores: dict[str, float] = field(default_factory=dict)
    perspective_recommended: bool = False
    debug_images: dict[str, Any] = field(default_factory=dict, repr=False)
    error_detail: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "confidence": round(float(self.confidence), 4),
            "corners": [
                [round(float(pair[0]), 2), round(float(pair[1]), 2)]
                for pair in (self.corners or [])
                if hasattr(pair, "__len__") and len(pair) >= 2
            ],
            "bbox": list(self.bbox) if self.bbox else None,
            "ruler": list(self.ruler) if self.ruler else None,
            "ruler_found": self.ruler is not None,
            "ruler_confidence": round(float(self.ruler_confidence), 4),
            "final_roi": list(self.final_roi) if self.final_roi else None,
            "reason": self.reason,
            "candidate_scores": {k: round(float(v), 4) for k, v in self.candidate_scores.items()},
            "perspective_recommended": self.perspective_recommended,
            "error_detail": self.error_detail,
        }


@dataclass
class ProcessResult:
    status: str
    input_path: str
    output_path: str | None = None
    confidence: float = 0.0
    box_found: bool = False
    ruler_found: bool = False
    reason: str = ""
    elapsed_ms: int = 0
    detection: DetectionResult | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "input": self.input_path,
            "output": self.output_path,
            "confidence": round(float(self.confidence), 4),
            "box_found": self.box_found,
            "ruler_found": self.ruler_found,
            "reason": self.reason,
            "elapsed_ms": self.elapsed_ms,
        }
