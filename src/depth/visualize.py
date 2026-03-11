from __future__ import annotations

import cv2
import numpy as np


def colorize_depth(
    depth_map: np.ndarray,
    *,
    min_depth: float | None = None,
    max_depth: float | None = None,
    invert: bool = True,
    blur_ksize: int = 0,
) -> np.ndarray:
    depth = np.asarray(depth_map, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0.0)
    if not np.any(valid):
        depth_uint8 = np.zeros(depth.shape, dtype=np.uint8)
        return cv2.applyColorMap(depth_uint8, cv2.COLORMAP_TURBO)

    valid_depth = depth[valid]
    if min_depth is None or max_depth is None:
        min_value = float(np.percentile(valid_depth, 2.0))
        max_value = float(np.percentile(valid_depth, 98.0))
        if max_value <= min_value:
            min_value = float(valid_depth.min())
            max_value = float(valid_depth.max())
    else:
        min_value = float(min_depth)
        max_value = float(max_depth)

    scale = max(max_value - min_value, 1e-6)

    normalized = np.zeros(depth.shape, dtype=np.float32)
    clipped = np.clip(depth[valid], min_value, max_value)
    normalized[valid] = (clipped - min_value) / scale
    if invert:
        normalized[valid] = 1.0 - normalized[valid]

    kernel_size = int(blur_ksize)
    if kernel_size > 1:
        if kernel_size % 2 == 0:
            kernel_size += 1
        normalized = cv2.GaussianBlur(normalized, (kernel_size, kernel_size), 0)

    depth_uint8 = np.clip(normalized * 255.0, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(depth_uint8, cv2.COLORMAP_TURBO)
