from __future__ import annotations

import numpy as np


def pointcloud_to_occupancy(points: np.ndarray, voxel_size: float = 0.25) -> np.ndarray:
    if points.size == 0:
        return np.empty((0, 3), dtype=np.int32)
    return np.floor(points / voxel_size).astype(np.int32)

