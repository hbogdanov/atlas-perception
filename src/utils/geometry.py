from __future__ import annotations

import numpy as np


def homogenize(points: np.ndarray) -> np.ndarray:
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Expected an array with shape (N, 3)")
    ones = np.ones((points.shape[0], 1), dtype=points.dtype)
    return np.hstack([points, ones])


def pixel_to_camera_coords(
    u: np.ndarray,
    v: np.ndarray,
    depth: np.ndarray,
    intrinsics: dict,
) -> np.ndarray:
    fx = float(intrinsics["fx"])
    fy = float(intrinsics["fy"])
    cx = float(intrinsics["cx"])
    cy = float(intrinsics["cy"])
    x = (u - cx) * depth / fx
    y = (v - cy) * depth / fy
    return np.stack([x, y, depth], axis=-1)


def depth_to_pointcloud(
    depth_map: np.ndarray,
    rgb_image: np.ndarray,
    intrinsics: dict,
    stride: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    depth_map = np.asarray(depth_map, dtype=np.float32)
    if depth_map.ndim == 3 and depth_map.shape[2] == 1:
        depth_map = depth_map[:, :, 0]
    if depth_map.ndim != 2:
        raise ValueError(f"depth_to_pointcloud expects a 2D depth map, got shape {depth_map.shape}.")
    v_coords, u_coords = np.mgrid[0 : depth_map.shape[0] : stride, 0 : depth_map.shape[1] : stride]
    sampled_depth = depth_map[::stride, ::stride].astype(np.float32)
    sampled_rgb = rgb_image[::stride, ::stride].astype(np.float32) / 255.0
    valid = sampled_depth > 0.0
    points = pixel_to_camera_coords(u_coords[valid], v_coords[valid], sampled_depth[valid], intrinsics)
    colors = sampled_rgb[valid]
    return points.reshape(-1, 3).astype(np.float32), colors.reshape(-1, 3).astype(np.float32)


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    if points.size == 0:
        return points.astype(np.float32)
    hom = homogenize(points)
    transformed = hom @ transform.T
    return transformed[:, :3].astype(np.float32)
