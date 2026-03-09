from __future__ import annotations

from pathlib import Path

import numpy as np

from src.slam.odometry import PoseEstimate, identity_pose
from src.slam.trajectory import Trajectory


class SlamWrapper:
    """Integration boundary for visual odometry or external SLAM systems."""

    def __init__(self, config: dict) -> None:
        self.config = config
        self.mode = str(config.get("mode", "disabled")).lower()
        self.trajectory = Trajectory()
        self._step = 0

    def update(self, image: np.ndarray, depth_map: np.ndarray, timestamp: float) -> PoseEstimate:
        if self.mode == "disabled":
            pose = self._disabled_pose(timestamp)
        elif self.mode == "dummy":
            pose = self._dummy_pose(timestamp)
        else:
            pose = self._backend_pose(image, depth_map, timestamp)
        self.trajectory.append(pose)
        return pose

    def export_trajectory(self, path: Path) -> None:
        self.trajectory.export(path)

    @staticmethod
    def _disabled_pose(timestamp: float) -> PoseEstimate:
        return identity_pose(timestamp)

    def _dummy_pose(self, timestamp: float) -> PoseEstimate:
        pose = identity_pose(timestamp)
        pose.matrix[0, 3] = float(self._step) * 0.05
        self._step += 1
        return pose

    def _backend_pose(self, image: np.ndarray, depth_map: np.ndarray, timestamp: float) -> PoseEstimate:
        del image, depth_map
        if self.mode == "orbslam_wrapper":
            raise NotImplementedError("ORB-SLAM backend integration is not implemented yet.")
        raise ValueError(f"Unsupported SLAM mode: {self.mode}")
