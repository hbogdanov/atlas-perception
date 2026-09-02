from __future__ import annotations

import cv2
import numpy as np


def compute_depth_confidence(depth_map: np.ndarray, edge_scale: float = 0.15) -> np.ndarray:
    """Estimate fusion confidence from valid support and local relative depth discontinuity."""
    depth = np.asarray(depth_map, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0.0)
    safe_depth = np.where(valid, depth, 0.0)
    grad_x = cv2.Sobel(safe_depth, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(safe_depth, cv2.CV_32F, 0, 1, ksize=3)
    relative_gradient = np.hypot(grad_x, grad_y) / np.maximum(safe_depth, 1e-6)
    confidence = np.exp(-relative_gradient / max(float(edge_scale), 1e-6)).astype(np.float32)
    confidence[~valid] = 0.0
    return confidence
