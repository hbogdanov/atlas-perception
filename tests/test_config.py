from pathlib import Path

import pytest

from src.utils.config import deep_merge_dicts, load_config, validate_config


def test_load_config():
    config = load_config("configs/default.yaml")
    assert config["input"]["mode"] == "webcam"
    assert config["ros2"]["depth_topic"] == "/atlas/depth"
    assert config["camera"]["fx"] == 525.0
    assert config["output"]["save_rgb_snapshot"] is False


def test_deep_merge_dicts_recursively_overrides_nested_values():
    merged = deep_merge_dicts(
        {"input": {"mode": "webcam", "fps": 30}, "ros2": {"enabled": True}},
        {"input": {"mode": "ros2", "source": "/camera/image_raw"}},
    )
    assert merged == {
        "input": {"mode": "ros2", "fps": 30, "source": "/camera/image_raw"},
        "ros2": {"enabled": True},
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
                "mapping:",
                "  stride: 1",
                "  max_points: 10",
                "ros2:",
                "  enabled: true",
                "output: {}",
            ]
        ),
        encoding="utf-8",
    )
    override.write_text("input:\n  mode: ros2\n  source: /sim/camera\n", encoding="utf-8")

    config = load_config(base, override)

    assert config["input"]["mode"] == "ros2"
    assert config["input"]["width"] == 640
    assert config["input"]["source"] == "/sim/camera"
    assert config["ros2"]["enabled"] is True


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


def test_validate_config_rejects_negative_principal_point():
    with pytest.raises(ValueError):
        validate_config(
            {
                "input": {"mode": "webcam"},
                "camera": {"fx": 1.0, "fy": 1.0, "cx": -1.0, "cy": 0.0},
                "depth": {"output_mode": "raw"},
                "slam": {"mode": "dummy"},
                "mapping": {"stride": 1, "max_points": 10},
                "ros2": {},
                "output": {},
            }
        )


def test_validate_config_rejects_non_positive_mapping_settings():
    with pytest.raises(ValueError):
        validate_config(
            {
                "input": {"mode": "webcam"},
                "camera": {"fx": 1.0, "fy": 1.0, "cx": 0.0, "cy": 0.0},
                "depth": {"output_mode": "raw"},
                "slam": {"mode": "dummy"},
                "mapping": {"stride": 0, "max_points": 10},
                "ros2": {},
                "output": {},
            }
        )
    with pytest.raises(ValueError):
        validate_config(
            {
                "input": {"mode": "webcam"},
                "camera": {"fx": 1.0, "fy": 1.0, "cx": 0.0, "cy": 0.0},
                "depth": {"output_mode": "raw"},
                "slam": {"mode": "dummy"},
                "mapping": {"stride": 1, "max_points": 0},
                "ros2": {},
                "output": {},
            }
        )


def test_validate_config_requires_source_for_video_and_ros2():
    for mode in ("video", "ros2"):
        with pytest.raises(ValueError):
            validate_config(
                {
                    "input": {"mode": mode},
                    "camera": {"fx": 1.0, "fy": 1.0, "cx": 0.0, "cy": 0.0},
                    "depth": {"output_mode": "raw"},
                    "slam": {"mode": "dummy"},
                    "mapping": {"stride": 1, "max_points": 10},
                    "ros2": {},
                    "output": {},
                }
            )


def test_validate_config_rejects_invalid_depth_postprocess_alpha():
    with pytest.raises(ValueError):
        validate_config(
            {
                "input": {"mode": "webcam"},
                "camera": {"fx": 1.0, "fy": 1.0, "cx": 0.0, "cy": 0.0},
                "depth": {
                    "output_mode": "raw",
                    "postprocess": {"enabled": True, "temporal_alpha": 1.5},
                },
                "slam": {"mode": "dummy"},
                "mapping": {"stride": 1, "max_points": 10},
                "ros2": {},
                "output": {},
            }
        )
