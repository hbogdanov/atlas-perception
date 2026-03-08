from __future__ import annotations

import cv2
import numpy as np


def colorize_depth(depth_map: np.ndarray) -> np.ndarray:
    depth_uint8 = np.clip(depth_map * 255.0, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(depth_uint8, cv2.COLORMAP_TURBO)

