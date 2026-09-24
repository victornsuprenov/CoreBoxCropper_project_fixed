"""CoreBoxCropper: offline box-and-core photo cropping for Windows."""

from .application import ApplicationService
from .inference.v3_pipeline import V3Pipeline
from .models import DetectionResult, ProcessResult

__all__ = ["ApplicationService", "DetectionResult", "ProcessResult", "V3Pipeline"]
