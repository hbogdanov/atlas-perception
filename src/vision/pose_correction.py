from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from src.slam.odometry import PoseEstimate
from src.vision.landmark_pose import VisualPoseMeasurement


@dataclass(frozen=True)
class VisualPoseCorrection:
    pose: PoseEstimate
    applied: bool
    reason: str
    translation_innovation_m: float
    rotation_innovation_deg: float


class VisualPoseCorrector:
    """Gate and blend absolute landmark poses into a tracked camera trajectory."""

    def __init__(self, config: dict | None = None) -> None:
        settings = config or {}
        self.enabled = bool(settings.get("apply_to_mapping", False))
        self.blend_weight = float(settings.get("blend_weight", 1.0))
        self.max_timestamp_delta_sec = float(settings.get("max_timestamp_delta_sec", 0.05))
        self.max_translation_innovation_m = float(settings.get("max_translation_innovation_m", 2.0))
        self.max_rotation_innovation_deg = float(settings.get("max_rotation_innovation_deg", 45.0))
        self.max_translation_std_m = float(settings.get("max_translation_std_m", 0.5))
        self.max_rotation_std_deg = float(settings.get("max_rotation_std_deg", 20.0))

    def correct(
        self, predicted_pose: PoseEstimate, measurement: VisualPoseMeasurement | None, timestamp: float
    ) -> VisualPoseCorrection:
        if not self.enabled:
            return _rejected(predicted_pose, "disabled")
        if measurement is None:
            return _rejected(predicted_pose, "no_measurement")
        if abs(float(measurement.timestamp) - float(timestamp)) > self.max_timestamp_delta_sec:
            return _rejected(predicted_pose, "timestamp_mismatch")
        if not predicted_pose.tracking_ok:
            relocalized = PoseEstimate(measurement.T_world_camera.copy(), float(timestamp), tracking_ok=True)
            return VisualPoseCorrection(relocalized, True, "relocalized", 0.0, 0.0)
        translation_delta, rotation_delta = _pose_innovation(predicted_pose.matrix, measurement.T_world_camera)
        if translation_delta > self.max_translation_innovation_m:
            return _rejected(predicted_pose, "translation_innovation", translation_delta, rotation_delta)
        if rotation_delta > self.max_rotation_innovation_deg:
            return _rejected(predicted_pose, "rotation_innovation", translation_delta, rotation_delta)
        translation_std = float(np.sqrt(np.mean(np.diag(measurement.covariance)[:3])))
        rotation_std_deg = float(np.rad2deg(np.sqrt(np.mean(np.diag(measurement.covariance)[3:]))))
        if translation_std > self.max_translation_std_m:
            return _rejected(predicted_pose, "translation_covariance", translation_delta, rotation_delta)
        if rotation_std_deg > self.max_rotation_std_deg:
            return _rejected(predicted_pose, "rotation_covariance", translation_delta, rotation_delta)
        corrected = _blend_poses(predicted_pose, measurement.T_world_camera, self.blend_weight, timestamp)
        return VisualPoseCorrection(corrected, True, "accepted", translation_delta, rotation_delta)


def _rejected(
    pose: PoseEstimate, reason: str, translation_innovation_m: float = 0.0, rotation_innovation_deg: float = 0.0
) -> VisualPoseCorrection:
    return VisualPoseCorrection(pose, False, reason, translation_innovation_m, rotation_innovation_deg)


def _pose_innovation(predicted: np.ndarray, measured: np.ndarray) -> tuple[float, float]:
    translation_delta = float(np.linalg.norm(measured[:3, 3] - predicted[:3, 3]))
    relative_rotation = predicted[:3, :3].T @ measured[:3, :3]
    cosine = float(np.clip((np.trace(relative_rotation) - 1.0) * 0.5, -1.0, 1.0))
    return translation_delta, float(np.rad2deg(np.arccos(cosine)))


def _blend_poses(predicted: PoseEstimate, measured: np.ndarray, weight: float, timestamp: float) -> PoseEstimate:
    blend = float(np.clip(weight, 0.0, 1.0))
    transform = predicted.matrix.copy()
    transform[:3, 3] = (1.0 - blend) * predicted.matrix[:3, 3] + blend * measured[:3, 3]
    relative_rotation = predicted.matrix[:3, :3].T @ measured[:3, :3]
    rotation_vector, _ = cv2.Rodrigues(relative_rotation.astype(np.float64))
    delta_rotation, _ = cv2.Rodrigues(rotation_vector * blend)
    transform[:3, :3] = (predicted.matrix[:3, :3] @ delta_rotation).astype(np.float32)
    return PoseEstimate(T_world_camera=transform, timestamp=float(timestamp), tracking_ok=predicted.tracking_ok)
