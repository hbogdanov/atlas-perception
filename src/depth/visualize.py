from __future__ import annotations

import cv2
import numpy as np


def colorize_depth(depth_map: np.ndarray, *, invert: bool = True) -> np.ndarray:
    depth = np.asarray(depth_map, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0.0)
    if not np.any(valid):
        depth_uint8 = np.zeros(depth.shape, dtype=np.uint8)
        return cv2.applyColorMap(depth_uint8, cv2.COLORMAP_TURBO)

    valid_depth = depth[valid]
    min_value = float(np.percentile(valid_depth, 2.0))
    max_value = float(np.percentile(valid_depth, 98.0))
    if max_value <= min_value:
        min_value = float(valid_depth.min())
        max_value = float(valid_depth.max())

    scale = max(max_value - min_value, 1e-6)

    normalized = np.zeros(depth.shape, dtype=np.float32)
    clipped = np.clip(depth[valid], min_value, max_value)
    normalized[valid] = (clipped - min_value) / scale
    if invert:
        normalized[valid] = 1.0 - normalized[valid]

    depth_uint8 = np.clip(normalized * 255.0, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(depth_uint8, cv2.COLORMAP_TURBO)
