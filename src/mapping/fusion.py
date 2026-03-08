from __future__ import annotations

import numpy as np


def fuse_pointclouds(*clouds: np.ndarray) -> np.ndarray:
    valid = [cloud for cloud in clouds if cloud.size]
    if not valid:
        return np.empty((0, 3), dtype=np.float32)
    return np.vstack(valid)

