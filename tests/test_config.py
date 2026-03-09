from pathlib import Path

import pytest

from src.utils.config import deep_merge_dicts, load_config, validate_config


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
    base.write_text(
        "\n".join(
            [
                "input:",
                "  mode: webcam",
                "  width: 640",
                "camera:",
                "  fx: 525.0",
                "  fy: 525.0",
                "  cx: 319.5",
                "  cy: 239.5",
                "depth:",
                "  output_mode: raw",
                "slam:",
                "  mode: dummy",
                "mapping: {}",
                "ros2:",
                "  enabled: true",
                "output: {}",
            ]
        ),
        encoding="utf-8",
    )
    override.write_text("input:\n  mode: ros2\nros2:\n  image_topic: /sim/camera\n", encoding="utf-8")

    config = load_config(base, override)

    assert config["input"]["mode"] == "ros2"
    assert config["input"]["width"] == 640
    assert config["ros2"]["enabled"] is True
    assert config["ros2"]["image_topic"] == "/sim/camera"


def test_validate_config_rejects_invalid_camera_intrinsics():
    with pytest.raises(ValueError):
        validate_config(
            {
                "input": {"mode": "webcam"},
                "camera": {"fx": 0.0, "fy": 1.0},
                "depth": {"output_mode": "raw"},
                "slam": {"mode": "dummy"},
                "mapping": {},
                "ros2": {},
                "output": {},
            }
        )
