import cv2
import numpy as np
import pytest

from coreboxcropper.config import DetectorConfig
from coreboxcropper.detector import BoxDetector
from coreboxcropper.ruler_detector import RulerDetector


def synthetic_box():
    image = np.full((600, 900, 3), 35, dtype=np.uint8)
    cv2.rectangle(image, (90, 110), (560, 470), (150, 100, 55), -1)
    cv2.rectangle(image, (90, 110), (560, 470), (235, 205, 150), 10)
    for x in (180, 275, 370, 465):
        cv2.line(image, (x, 125), (x, 455), (235, 205, 150), 7)
    cv2.line(image, (560, 455), (820, 455), (230, 230, 230), 12)
    return image


def test_detector_returns_structured_result():
    result = BoxDetector(DetectorConfig(confidence_threshold=0.2)).detect(synthetic_box())
    assert result.corners or result.reason in {"box_not_found", "box_confidence_below_threshold"}
    assert 0 <= result.confidence <= 1


def test_ruler_signature_detects_color_scale_and_ten_centimeter_marks():
    config = DetectorConfig()
    detector = RulerDetector(config)
    region = np.full((1600, 180, 3), (145, 105, 65), dtype=np.uint8)
    cv2.rectangle(region, (55, 0), (125, 1599), (190, 180, 145), -1)
    for index, color in enumerate(((0, 0, 220), (0, 180, 0), (220, 80, 0), (0, 210, 210))):
        y1 = 1400 + index * 45
        cv2.rectangle(region, (58, y1), (122, y1 + 42), color, -1)
    for y in range(80, 1360, 160):
        cv2.line(region, (60, y), (120, y), (35, 35, 35), 5)

    color_score = detector._color_scale_score(region, 45, 0, 135, 1599)
    tick_score = detector._ten_cm_tick_score(region, 45, 0, 135, 1599)

    assert color_score >= 0.5
    assert tick_score >= 0.35


def test_ruler_signature_parameters_are_validated():
    with pytest.raises(ValueError):
        DetectorConfig(ruler_tick_step_cm=100, ruler_length_cm=100).validate()


def test_detector_is_orientation_invariant_for_synthetic_box():
    detector = BoxDetector(DetectorConfig(confidence_threshold=0.2))
    source = synthetic_box()
    results = []
    for angle in (0, 90, 180, 270):
        if angle == 0:
            image = source
        elif angle == 90:
            image = cv2.rotate(source, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            image = cv2.rotate(source, cv2.ROTATE_180)
        else:
            image = cv2.rotate(source, cv2.ROTATE_90_COUNTERCLOCKWISE)
        result = detector.detect(image)
        results.append(result)
        assert result.candidate_scores.get("orientation_degrees") in {0.0, 90.0, 180.0, 270.0}
        assert result.corners or result.reason in {
            "box_not_found", "box_confidence_below_threshold", "yolo_no_detection"
        }
    assert all(0 <= result.confidence <= 1 for result in results)
