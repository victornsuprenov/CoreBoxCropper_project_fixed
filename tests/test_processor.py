import numpy as np

from coreboxcropper.config import DetectorConfig
from coreboxcropper.models import DetectionResult
from coreboxcropper.processor import ImageProcessor


def test_processor_crops_axis_aligned_roi():
    image = np.zeros((100, 140, 3), dtype=np.uint8)
    detection = DetectionResult(True, 0.9, [(20, 20), (100, 20), (100, 80), (20, 80)], (20, 20, 100, 80), None, final_roi=(10, 10, 110, 90))
    output = ImageProcessor(DetectorConfig()).process(image, detection)
    assert output.shape[:2] == (81, 101)
