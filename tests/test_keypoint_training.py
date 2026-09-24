import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "ml" / "prepare_keypoint_training.py"
SPEC = importlib.util.spec_from_file_location("prepare_keypoint_training", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_pose_row_contains_bbox_and_four_visible_keypoints():
    row = MODULE._pose_row({"corners": [[0.1, 0.2], [0.8, 0.2], [0.8, 0.9], [0.1, 0.9]]})
    values = row.split()

    assert len(values) == 17
    assert values[0] == "0"
    assert [float(values[index]) for index in (1, 2, 3, 4)] == [0.45, 0.55, 0.7, 0.7]
    assert [float(values[index]) for index in (7, 10, 13, 16)] == [2.0, 2.0, 2.0, 2.0]


def test_pose_validator_accepts_visibility_value_two(tmp_path):
    dataset = tmp_path / "keypoint_dataset"
    (dataset / "images" / "train").mkdir(parents=True)
    (dataset / "images" / "val").mkdir(parents=True)
    (dataset / "labels" / "train").mkdir(parents=True)
    (dataset / "labels" / "val").mkdir(parents=True)
    yaml = dataset / "dataset.yaml"
    yaml.write_text("kpt_shape: [4, 3]\ntrain: images/train\nval: images/val\nnames:\n", encoding="utf-8")
    from PIL import Image

    for split in ("train", "val"):
        image = dataset / "images" / split / "sample.jpg"
        Image.new("RGB", (100, 100)).save(image)
        (dataset / "labels" / split / "sample.txt").write_text(
            MODULE._pose_row({"corners": [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]]}) + "\n",
            encoding="utf-8",
        )

    validator_path = Path(__file__).parents[1] / "ml" / "validate_keypoint_dataset.py"
    validator_spec = importlib.util.spec_from_file_location("validate_keypoint_dataset", validator_path)
    validator = importlib.util.module_from_spec(validator_spec)
    assert validator_spec.loader is not None
    validator_spec.loader.exec_module(validator)
    import sys

    old_argv = sys.argv
    try:
        sys.argv = [str(validator_path), str(dataset)]
        assert validator.main() == 0
    finally:
        sys.argv = old_argv