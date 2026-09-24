import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "ml" / "regression_v3.py"
SPEC = importlib.util.spec_from_file_location("regression_v3", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_v3_harness_defines_image_extensions_and_runner():
    assert ".jpg" in MODULE.IMAGE_EXTENSIONS
    assert callable(MODULE.run)