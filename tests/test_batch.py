from pathlib import Path

from coreboxcropper.application import ApplicationService


def test_process_paths_handles_multiple_images(tmp_path):
    source_dir = Path("attached_assets")
    photos = sorted(source_dir.glob("826_6001-*_W_1788890262148.jpg"))[:2]
    results = ApplicationService().process_paths(photos, tmp_path, overwrite=True)
    assert len(results) == 2
    assert all(result.status == "success" for result in results)
    assert all(Path(result.output_path).exists() for result in results)


def test_process_paths_preserves_original_filename(tmp_path):
    source_dir = Path("attached_assets")
    photo = sorted(source_dir.glob("826_6001-*_W_1788890262148.jpg"))[0]
    results = ApplicationService().process_paths([photo], tmp_path, overwrite=True)
    assert results[0].status == "success"
    assert Path(results[0].output_path).name == photo.name
