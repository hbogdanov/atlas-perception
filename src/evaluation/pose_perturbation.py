from __future__ import annotations

from collections import deque

import cv2
import numpy as np

from src.slam.odometry import PoseEstimate


class PosePerturber:
    """Inject deterministic pose errors to isolate mapping sensitivity from depth quality."""

    def __init__(self, config: dict | None = None) -> None:
        settings = config or {}
        self.enabled = bool(settings.get("enabled", False))
        self.translation_std = float(settings.get("translation_std_m", 0.0))
        self.rotation_std_deg = float(settings.get("rotation_std_deg", 0.0))
        self.dropout_probability = float(settings.get("dropout_probability", 0.0))
        self.latency_frames = int(settings.get("latency_frames", 0))
        self._rng = np.random.default_rng(int(settings.get("seed", 0)))
        self._history: deque[PoseEstimate] = deque(maxlen=self.latency_frames + 1)

    def perturb(self, pose: PoseEstimate) -> PoseEstimate | None:
        if not self.enabled:
            return pose
        self._history.append(_copy_pose(pose))
        if self._rng.random() < self.dropout_probability:
            return None
        delayed = self._history[0]
        transform = delayed.matrix.copy()
        transform[:3, 3] += self._rng.normal(0.0, self.translation_std, 3).astype(np.float32)
        rotation_std_rad = np.deg2rad(self.rotation_std_deg)
        if rotation_std_rad > 0.0:
            rotation_vector = self._rng.normal(0.0, rotation_std_rad, 3).astype(np.float64)
            rotation_noise, _ = cv2.Rodrigues(rotation_vector)
            transform[:3, :3] = (rotation_noise.astype(np.float32) @ transform[:3, :3]).astype(np.float32)
        return PoseEstimate(T_world_camera=transform, timestamp=pose.timestamp, tracking_ok=pose.tracking_ok)


def _copy_pose(pose: PoseEstimate) -> PoseEstimate:
    return PoseEstimate(T_world_camera=pose.matrix.copy(), timestamp=pose.timestamp, tracking_ok=pose.tracking_ok)
