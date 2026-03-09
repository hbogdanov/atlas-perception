from pathlib import Path

from src.utils.config import deep_merge_dicts, load_config


def test_load_config():
    config = load_config("configs/default.yaml")
    assert config["input"]["mode"] == "webcam"
    assert config["ros2"]["depth_topic"] == "/atlas/depth"
    assert config["camera"]["fx"] == 525.0


def test_deep_merge_dicts_recursively_overrides_nested_values():
    merged = deep_merge_dicts(
        {"input": {"mode": "webcam", "fps": 30}, "ros2": {"enabled": True}},
        {"input": {"mode": "ros2"}, "ros2": {"image_topic": "/camera/image_raw"}},
    )
    assert merged == {
        "input": {"mode": "ros2", "fps": 30},
        "ros2": {"enabled": True, "image_topic": "/camera/image_raw"},
    }


def test_load_config_with_override(tmp_path: Path):
    base = tmp_path / "base.yaml"
    override = tmp_path / "override.yaml"
    base.write_text("input:\n  mode: webcam\n  width: 640\nros2:\n  enabled: true\n", encoding="utf-8")
    override.write_text("input:\n  mode: ros2\nros2:\n  image_topic: /sim/camera\n", encoding="utf-8")

    config = load_config(base, override)

    assert config["input"]["mode"] == "ros2"
    assert config["input"]["width"] == 640
    assert config["ros2"]["enabled"] is True
    assert config["ros2"]["image_topic"] == "/sim/camera"
