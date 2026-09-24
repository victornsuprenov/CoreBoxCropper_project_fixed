from pathlib import Path

import pytest

from coreboxcropper.application import ApplicationService


PHOTO_DIR = Path("attached_assets")
EXPECTED = {
    "826_6001-1_(0.00-3.00)_W_1788890262148.jpg": (1051, 1942),
    "826_6001-2_(3.00-6.00)_W_1788890262148.jpg": (1040, 1930),
    "826_6001-3_(6.00-9.00)_W_1788890262148.jpg": (989, 1899),
}


@pytest.mark.parametrize("name, expected_x", EXPECTED.items())
def test_reference_target_is_the_three_cell_box_next_to_ruler(name, expected_x):
    photo = PHOTO_DIR / name
    if not photo.exists():
        pytest.skip("reference photo is optional calibration asset")

    image, detection = ApplicationService().preview_detection(photo)
    assert detection.success, detection.reason
    assert detection.bbox is not None
    assert detection.ruler is not None

    left_expected, right_expected = expected_x
    left, _, right, _ = detection.bbox
    # Allow small changes if thresholds are tuned later, but prevent the
    # detector from silently moving to the neighbouring box.
    assert abs(left - left_expected) <= 70
    assert abs(right - right_expected) <= 70
