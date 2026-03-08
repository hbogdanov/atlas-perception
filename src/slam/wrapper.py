from __future__ import annotations

from pathlib import Path

import numpy as np

from src.slam.odometry import PoseEstimate, identity_pose
from src.slam.trajectory import Trajectory


class SlamWrapper:
    """Integration boundary for visual odometry or external SLAM systems."""

    def __init__(self, config: dict) -> None:
        self.config = config
        self.enabled = bool(config.get("enabled", False))
        self.trajectory = Trajectory()
        self._step = 0

    def update(self, image: np.ndarray, depth_map: np.ndarray, timestamp: float) -> PoseEstimate:
        del image, depth_map
        pose = identity_pose(timestamp)
        pose.matrix[0, 3] = float(self._step) * 0.05
        self._step += 1
        self.trajectory.append(pose)
        return pose

    def export_trajectory(self, path: Path) -> None:
        self.trajectory.export(path)

