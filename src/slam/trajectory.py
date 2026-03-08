from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.slam.odometry import PoseEstimate


@dataclass
class Trajectory:
    poses: list[PoseEstimate] = field(default_factory=list)

    def append(self, pose: PoseEstimate) -> None:
        self.poses.append(pose)

    def export(self, path: Path) -> None:
        stacked = np.stack([pose.matrix for pose in self.poses]) if self.poses else np.empty((0, 4, 4))
        np.save(path, stacked)

