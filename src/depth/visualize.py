from __future__ import annotations

import cv2
import numpy as np


def colorize_depth(depth_map: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth_map, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0.0)
    if not np.any(valid):
        depth_uint8 = np.zeros(depth.shape, dtype=np.uint8)
        return cv2.applyColorMap(depth_uint8, cv2.COLORMAP_TURBO)
    min_depth = float(depth[valid].min())
    max_depth = float(depth[valid].max())
    scale = max(max_depth - min_depth, 1e-6)
    normalized = np.zeros(depth.shape, dtype=np.float32)
    normalized[valid] = (depth[valid] - min_depth) / scale
    depth_uint8 = np.clip(normalized * 255.0, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(depth_uint8, cv2.COLORMAP_TURBO)
