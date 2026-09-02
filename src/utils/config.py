from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import yaml

REQUIRED_SECTIONS = ("input", "camera", "depth", "slam", "mapping", "ros2", "output")
VALID_INPUT_MODES = {"webcam", "video", "ros2", "rgbd_dataset"}
VALID_DEPTH_OUTPUT_MODES = {"relative_normalized", "raw"}
VALID_DEPTH_SOURCE_MODES = {"estimate", "input"}
VALID_SLAM_MODES = {"disabled", "dummy", "rtabmap", "groundtruth"}
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
    if input_mode in {"video", "ros2", "rgbd_dataset"} and not config["input"].get("source"):
        raise ValueError("input.source is required when input.mode is 'video', 'ros2', or 'rgbd_dataset'.")

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
        raise ValueError(f"depth.output_mode must be one of {sorted(VALID_DEPTH_OUTPUT_MODES)}, got {output_mode!r}.")
    source_mode = str(config["depth"].get("source_mode", "estimate")).lower()
    if source_mode not in VALID_DEPTH_SOURCE_MODES:
        raise ValueError(f"depth.source_mode must be one of {sorted(VALID_DEPTH_SOURCE_MODES)}, got {source_mode!r}.")
    config["depth"]["source_mode"] = source_mode
    depth_model = str(config["depth"].get("depth_model", config["depth"].get("model", "midas"))).lower()
    config["depth"]["depth_model"] = depth_model
    _validate_depth_postprocess(config["depth"].get("postprocess", {}))
    _validate_semantics(config.get("semantics", {}))
    _validate_pose_perturbation(config.get("evaluation", {}).get("pose_perturbation", {}))
    _validate_visual_localization(config.get("visual_localization", {}))

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
    confidence_fusion = config["mapping"].get("confidence_fusion", {})
    if confidence_fusion and not isinstance(confidence_fusion, dict):
        raise ValueError("mapping.confidence_fusion must be a dictionary when provided.")
    if confidence_fusion:
        min_confidence = float(confidence_fusion.get("min_confidence", 0.2))
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError("mapping.confidence_fusion.min_confidence must be between 0 and 1.")
        if float(confidence_fusion.get("edge_scale", 0.15)) <= 0.0:
            raise ValueError("mapping.confidence_fusion.edge_scale must be greater than 0.")
    multi_view = config["mapping"].get("multi_view_consistency", {})
    if multi_view and not isinstance(multi_view, dict):
        raise ValueError("mapping.multi_view_consistency must be a dictionary when provided.")
    if multi_view and float(multi_view.get("relative_error_scale", 0.1)) <= 0.0:
        raise ValueError("mapping.multi_view_consistency.relative_error_scale must be greater than 0.")

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


def _validate_pose_perturbation(settings: dict) -> None:
    if not settings:
        return
    if not isinstance(settings, dict):
        raise ValueError("evaluation.pose_perturbation must be a dictionary when provided.")
    for key in ("translation_std_m", "rotation_std_deg"):
        if float(settings.get(key, 0.0)) < 0.0:
            raise ValueError(f"evaluation.pose_perturbation.{key} must be non-negative.")
    dropout_probability = float(settings.get("dropout_probability", 0.0))
    if not 0.0 <= dropout_probability <= 1.0:
        raise ValueError("evaluation.pose_perturbation.dropout_probability must be between 0 and 1.")
    if int(settings.get("latency_frames", 0)) < 0:
        raise ValueError("evaluation.pose_perturbation.latency_frames must be non-negative.")


def _validate_visual_localization(settings: dict) -> None:
    if not settings:
        return
    if not isinstance(settings, dict):
        raise ValueError("visual_localization must be a dictionary when provided.")
    if float(settings.get("marker_length_m", 0.16)) <= 0.0:
        raise ValueError("visual_localization.marker_length_m must be greater than 0.")
    if float(settings.get("max_reprojection_error", 3.0)) <= 0.0:
        raise ValueError("visual_localization.max_reprojection_error must be greater than 0.")
    for landmark in settings.get("landmarks", []):
        if "id" not in landmark or "T_world_marker" not in landmark:
            raise ValueError("Each visual_localization landmark requires id and T_world_marker.")
        if np.asarray(landmark["T_world_marker"]).shape != (4, 4):
            raise ValueError("visual_localization landmark T_world_marker must have shape (4, 4).")
    correction = settings.get("pose_correction", {})
    if correction and not isinstance(correction, dict):
        raise ValueError("visual_localization.pose_correction must be a dictionary when provided.")
    if correction and not 0.0 <= float(correction.get("blend_weight", 1.0)) <= 1.0:
        raise ValueError("visual_localization.pose_correction.blend_weight must be between 0 and 1.")
    for key in (
        "max_timestamp_delta_sec",
        "max_translation_innovation_m",
        "max_rotation_innovation_deg",
        "max_translation_std_m",
        "max_rotation_std_deg",
    ):
        if correction and float(correction.get(key, 1.0)) <= 0.0:
            raise ValueError(f"visual_localization.pose_correction.{key} must be greater than 0.")


def load_config(path: str | Path, override_path: str | Path | None = None) -> dict:
    config = _read_yaml(path)
    if override_path is None:
        return validate_config(config)
    override = _read_yaml(override_path)
    return validate_config(deep_merge_dicts(config, override))
