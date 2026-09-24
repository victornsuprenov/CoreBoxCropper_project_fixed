import json

from coreboxcropper.config import load_config


def test_explicit_v3_disable_is_preserved(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"v3_enabled": False}), encoding="utf-8")

    assert load_config(path).v3_enabled is False