from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PoseEstimate:
    matrix: np.ndarray
    timestamp: float


def identity_pose(timestamp: float) -> PoseEstimate:
    return PoseEstimate(matrix=np.eye(4, dtype=np.float32), timestamp=timestamp)

