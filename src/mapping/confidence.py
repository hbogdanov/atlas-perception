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


def compute_multiview_confidence(
    current_depth: np.ndarray,
    current_pose: np.ndarray,
    previous_depth: np.ndarray,
    previous_pose: np.ndarray,
    intrinsics: dict,
    relative_error_scale: float = 0.1,
) -> tuple[np.ndarray, int]:
    """Score current depth by reprojection agreement with the prior camera view.

    Pixels that are not co-visible remain neutral; new surfaces should not be rejected
    merely because they were absent from the previous frame.
    """
    current = np.asarray(current_depth, dtype=np.float32)
    previous = np.asarray(previous_depth, dtype=np.float32)
    if current.shape != previous.shape:
        raise ValueError("Multi-view consistency requires matching current and previous depth shapes.")
    height, width = current.shape
    v_coords, u_coords = np.mgrid[0:height, 0:width]
    valid = np.isfinite(current) & (current > 0.0)
    fx, fy = float(intrinsics["fx"]), float(intrinsics["fy"])
    cx, cy = float(intrinsics["cx"]), float(intrinsics["cy"])
    x = (u_coords - cx) * current / fx
    y = (v_coords - cy) * current / fy
    points = np.stack([x, y, current, np.ones_like(current)], axis=-1).reshape(-1, 4)
    T_previous_current = np.linalg.inv(previous_pose) @ current_pose
    previous_points = points @ T_previous_current.T
    previous_z = previous_points[:, 2].reshape(height, width)
    projected_u = np.rint(fx * previous_points[:, 0] / np.maximum(previous_points[:, 2], 1e-6) + cx).astype(int)
    projected_v = np.rint(fy * previous_points[:, 1] / np.maximum(previous_points[:, 2], 1e-6) + cy).astype(int)
    projected_u = projected_u.reshape(height, width)
    projected_v = projected_v.reshape(height, width)
    in_bounds = (projected_u >= 0) & (projected_u < width) & (projected_v >= 0) & (projected_v < height)
    sampled_previous = np.zeros_like(current)
    sampled_previous[in_bounds] = previous[projected_v[in_bounds], projected_u[in_bounds]]
    co_visible = valid & in_bounds & (previous_z > 0.0) & np.isfinite(sampled_previous) & (sampled_previous > 0.0)
    confidence = np.ones_like(current, dtype=np.float32)
    relative_error = np.abs(previous_z - sampled_previous) / np.maximum(sampled_previous, 1e-6)
    confidence[co_visible] = np.exp(-relative_error[co_visible] / max(float(relative_error_scale), 1e-6))
    confidence[~valid] = 0.0
    return confidence, int(np.count_nonzero(co_visible))
