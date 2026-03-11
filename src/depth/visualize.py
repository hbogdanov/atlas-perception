from __future__ import annotations

import cv2
import numpy as np


def normalize_depth_for_display(depth_map: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth_map, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0.0)
    normalized = np.zeros(depth.shape, dtype=np.float32)
    if not np.any(valid):
        return normalized

    normalized[valid] = cv2.normalize(depth[valid], None, 0.0, 1.0, cv2.NORM_MINMAX).reshape(-1)
    normalized[valid] = 1.0 - normalized[valid]

    if np.any(~valid):
        depth_uint8 = np.clip(normalized * 255.0, 0, 255).astype(np.uint8)
        invalid_mask = (~valid).astype(np.uint8) * 255
        depth_uint8 = cv2.inpaint(depth_uint8, invalid_mask, 3, cv2.INPAINT_TELEA)
        normalized = depth_uint8.astype(np.float32) / 255.0
    return normalized


def colorize_depth(depth_map: np.ndarray) -> np.ndarray:
    depth_uint8 = np.clip(np.asarray(depth_map, dtype=np.float32) * 255.0, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(depth_uint8, cv2.COLORMAP_TURBO)
