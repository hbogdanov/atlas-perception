from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import yaml

REQUIRED_SECTIONS = ("input", "camera", "depth", "slam", "mapping", "ros2", "output")
VALID_INPUT_MODES = {"webcam", "video", "ros2"}
VALID_DEPTH_OUTPUT_MODES = {"relative_normalized", "raw"}
VALID_SLAM_MODES = {"disabled", "dummy", "orbslam_wrapper"}


def _read_yaml(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config at {path} must deserialize to a dictionary.")
    return data


def deep_merge_dicts(base: dict, override: dict) -> dict:
    merged = deepcopy(base)
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = deep_merge_dicts(current, value)
        else:
            merged[key] = deepcopy(value)
    return merged


def validate_config(config: dict) -> dict:
    missing = [section for section in REQUIRED_SECTIONS if section not in config]
    if missing:
        raise ValueError(f"Config is missing required sections: {', '.join(missing)}")

    input_mode = str(config["input"].get("mode", "")).lower()
    if input_mode not in VALID_INPUT_MODES:
        raise ValueError(f"input.mode must be one of {sorted(VALID_INPUT_MODES)}, got {input_mode!r}.")

    fx = float(config["camera"].get("fx", 0.0))
    fy = float(config["camera"].get("fy", 0.0))
    if fx <= 0.0 or fy <= 0.0:
        raise ValueError("camera.fx and camera.fy must be positive.")

    output_mode = str(config["depth"].get("output_mode", "")).lower()
    if output_mode not in VALID_DEPTH_OUTPUT_MODES:
        raise ValueError(
            f"depth.output_mode must be one of {sorted(VALID_DEPTH_OUTPUT_MODES)}, got {output_mode!r}."
        )

    slam_mode = str(config["slam"].get("mode", "")).lower()
    if slam_mode not in VALID_SLAM_MODES:
        raise ValueError(f"slam.mode must be one of {sorted(VALID_SLAM_MODES)}, got {slam_mode!r}.")

    return config


def load_config(path: str | Path, override_path: str | Path | None = None) -> dict:
    config = _read_yaml(path)
    if override_path is None:
        return validate_config(config)
    override = _read_yaml(override_path)
    return validate_config(deep_merge_dicts(config, override))
