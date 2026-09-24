import cv2
import numpy as np

from coreboxcropper.config import DetectorConfig
from coreboxcropper.yolo_detector import YoloFirstDetector


def test_yolo_pipeline_refines_a_local_quadrilateral():
    image = np.full((500, 700, 3), 30, dtype=np.uint8)
    points = np.array([[180, 40], [520, 65], [500, 450], [160, 430]], dtype=np.int32)
    cv2.fillConvexPoly(image, points, (150, 120, 90))
    cv2.polylines(image, [points], True, (235, 210, 170), 8)

    detector = YoloFirstDetector(DetectorConfig())
    quad = detector._find_local_quad(image)

    assert quad is not None
    assert len(quad) == 4


def test_yolo_config_points_to_new_model_and_baseline():
    config = DetectorConfig()
    primary, fallback = YoloFirstDetector(config)._weight_paths()
    assert primary.name == "best.pt"
    assert "corebox_v2-2" in str(primary)
    assert fallback == primary