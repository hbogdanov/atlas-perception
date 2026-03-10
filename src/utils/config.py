from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import yaml

REQUIRED_SECTIONS = ("input", "camera", "depth", "slam", "mapping", "ros2", "output")
VALID_INPUT_MODES = {"webcam", "video", "ros2"}
VALID_DEPTH_OUTPUT_MODES = {"relative_normalized", "raw"}
VALID_SLAM_MODES = {"disabled", "dummy", "rtabmap"}
VALID_MAPPING_REPRESENTATIONS = {"pointcloud", "tsdf"}
VALID_SEMANTIC_BACKENDS = {"disabled", "yolov8_seg"}


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
    if input_mode in {"video", "ros2"} and not config["input"].get("source"):
        raise ValueError("input.source is required when input.mode is 'video' or 'ros2'.")

    fx = float(config["camera"].get("fx", 0.0))
    fy = float(config["camera"].get("fy", 0.0))
    cx = float(config["camera"].get("cx", -1.0))
    cy = float(config["camera"].get("cy", -1.0))
    if fx <= 0.0 or fy <= 0.0:
        raise ValueError("camera.fx and camera.fy must be positive.")
    if cx < 0.0 or cy < 0.0:
        raise ValueError("camera.cx and camera.cy must be non-negative.")

    output_mode = str(config["depth"].get("output_mode", "")).lower()
    if output_mode not in VALID_DEPTH_OUTPUT_MODES:
        raise ValueError(
            f"depth.output_mode must be one of {sorted(VALID_DEPTH_OUTPUT_MODES)}, got {output_mode!r}."
        )
    depth_model = str(config["depth"].get("depth_model", config["depth"].get("model", "midas"))).lower()
    config["depth"]["depth_model"] = depth_model
    _validate_depth_postprocess(config["depth"].get("postprocess", {}))
    _validate_semantics(config.get("semantics", {}))

    slam_mode = str(config["slam"].get("mode", "")).lower()
    if slam_mode not in VALID_SLAM_MODES:
        raise ValueError(f"slam.mode must be one of {sorted(VALID_SLAM_MODES)}, got {slam_mode!r}.")

    representation = str(config["mapping"].get("representation", "pointcloud")).lower()
    if representation not in VALID_MAPPING_REPRESENTATIONS:
        raise ValueError(
            f"mapping.representation must be one of {sorted(VALID_MAPPING_REPRESENTATIONS)}, got {representation!r}."
        )
    config["mapping"]["representation"] = representation
    stride = int(config["mapping"].get("stride", 0))
    max_points = int(config["mapping"].get("max_points", 0))
    if stride <= 0:
        raise ValueError("mapping.stride must be greater than 0.")
    if max_points <= 0:
        raise ValueError("mapping.max_points must be greater than 0.")
    if representation == "tsdf":
        voxel_length = float(config["mapping"].get("tsdf_voxel_length", 0.0))
        sdf_trunc = float(config["mapping"].get("tsdf_sdf_trunc", 0.0))
        depth_trunc = float(config["mapping"].get("tsdf_depth_trunc", 0.0))
        if voxel_length <= 0.0:
            raise ValueError("mapping.tsdf_voxel_length must be greater than 0.")
        if sdf_trunc <= 0.0:
            raise ValueError("mapping.tsdf_sdf_trunc must be greater than 0.")
        if depth_trunc <= 0.0:
            raise ValueError("mapping.tsdf_depth_trunc must be greater than 0.")

    return config


def _validate_depth_postprocess(postprocess: dict) -> None:
    if not postprocess:
        return
    if not isinstance(postprocess, dict):
        raise ValueError("depth.postprocess must be a dictionary when provided.")

    if "bilateral_diameter" in postprocess and int(postprocess["bilateral_diameter"]) <= 0:
        raise ValueError("depth.postprocess.bilateral_diameter must be greater than 0.")
    if "bilateral_sigma_color" in postprocess and float(postprocess["bilateral_sigma_color"]) <= 0.0:
        raise ValueError("depth.postprocess.bilateral_sigma_color must be greater than 0.")
    if "bilateral_sigma_space" in postprocess and float(postprocess["bilateral_sigma_space"]) <= 0.0:
        raise ValueError("depth.postprocess.bilateral_sigma_space must be greater than 0.")
    if "guided_radius" in postprocess and int(postprocess["guided_radius"]) < 0:
        raise ValueError("depth.postprocess.guided_radius must be non-negative.")
    if "guided_eps" in postprocess and float(postprocess["guided_eps"]) <= 0.0:
        raise ValueError("depth.postprocess.guided_eps must be greater than 0.")
    if "temporal_alpha" in postprocess:
        alpha = float(postprocess["temporal_alpha"])
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("depth.postprocess.temporal_alpha must be between 0 and 1 inclusive.")


def _validate_semantics(semantics: dict) -> None:
    if not semantics:
        return
    if not isinstance(semantics, dict):
        raise ValueError("semantics must be a dictionary when provided.")
    backend = str(semantics.get("backend", "disabled" if not semantics.get("enabled", False) else "yolov8_seg")).lower()
    if backend not in VALID_SEMANTIC_BACKENDS:
        raise ValueError(f"semantics.backend must be one of {sorted(VALID_SEMANTIC_BACKENDS)}, got {backend!r}.")
    if "confidence" in semantics:
        confidence = float(semantics["confidence"])
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("semantics.confidence must be between 0 and 1 inclusive.")
    if "iou" in semantics:
        iou = float(semantics["iou"])
        if not 0.0 <= iou <= 1.0:
            raise ValueError("semantics.iou must be between 0 and 1 inclusive.")


def load_config(path: str | Path, override_path: str | Path | None = None) -> dict:
    config = _read_yaml(path)
    if override_path is None:
        return validate_config(config)
    override = _read_yaml(override_path)
    return validate_config(deep_merge_dicts(config, override))
