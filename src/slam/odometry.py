from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PoseEstimate:
    T_world_camera: np.ndarray
    timestamp: float
    tracking_ok: bool = True

    @property
    def matrix(self) -> np.ndarray:
        return self.T_world_camera


def identity_pose(timestamp: float) -> PoseEstimate:
    return PoseEstimate(T_world_camera=np.eye(4, dtype=np.float32), timestamp=timestamp, tracking_ok=True)
