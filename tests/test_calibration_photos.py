from pathlib import Path

import pytest

from coreboxcropper.application import ApplicationService


PHOTO_DIR = Path("attached_assets")
PHOTOS = sorted(PHOTO_DIR.glob("826_6001-*_W_1788890262148.jpg"))


@pytest.mark.parametrize("photo", PHOTOS, ids=lambda path: path.name)
def test_reference_photo_finds_three_cell_box_and_ruler(photo, tmp_path):
    if not photo.exists():
        pytest.skip("reference photos are optional calibration assets")
    result = ApplicationService().process_image(
        photo, tmp_path / f"{photo.stem}_test.jpg", overwrite=True
    )
    assert result.status == "success", result.reason
    assert result.box_found
    assert result.ruler_found
    assert result.detection is not None
    assert result.detection.candidate_scores["three_cell_score"] == 1.0
    assert result.detection.final_roi is not None
    assert result.detection.final_roi[1] > 0
