from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class LandmarkObservation:
    landmark_id: str
    world_points: np.ndarray
    image_points: np.ndarray


@dataclass(frozen=True)
class VisualPoseMeasurement:
    timestamp: float
    T_world_camera: np.ndarray
    covariance: np.ndarray
    reprojection_rmse: float
    inlier_count: int
    landmark_ids: tuple[str, ...]


def solve_landmark_pose(
    observations: list[LandmarkObservation],
    intrinsics: dict,
    timestamp: float,
    max_reprojection_error: float = 3.0,
    min_inliers: int = 4,
) -> VisualPoseMeasurement | None:
    """Estimate a world-frame camera pose from known 3D landmarks using robust PnP."""
    if not observations:
        return None
    world_points = np.vstack([np.asarray(observation.world_points, dtype=np.float32) for observation in observations])
    image_points = np.vstack([np.asarray(observation.image_points, dtype=np.float32) for observation in observations])
    if world_points.shape != (image_points.shape[0], 3) or image_points.shape[1] != 2:
        raise ValueError("Landmark observations must provide matching (N, 3) world and (N, 2) image points.")
    if world_points.shape[0] < min_inliers:
        return None
    camera_matrix = np.array(
        [
            [float(intrinsics["fx"]), 0.0, float(intrinsics["cx"])],
            [0.0, float(intrinsics["fy"]), float(intrinsics["cy"])],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    success, rotation_vector, translation_vector, inliers = cv2.solvePnPRansac(
        world_points,
        image_points,
        camera_matrix,
        None,
        reprojectionError=float(max_reprojection_error),
        flags=cv2.SOLVEPNP_EPNP,
    )
    if not success or inliers is None or len(inliers) < min_inliers:
        return None
    inlier_indices = inliers.reshape(-1)
    refined, rotation_vector, translation_vector = cv2.solvePnP(
        world_points[inlier_indices],
        image_points[inlier_indices],
        camera_matrix,
        None,
        rotation_vector,
        translation_vector,
        useExtrinsicGuess=True,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not refined:
        return None
    projected, _ = cv2.projectPoints(
        world_points[inlier_indices], rotation_vector, translation_vector, camera_matrix, None
    )
    errors = np.linalg.norm(projected.reshape(-1, 2) - image_points[inlier_indices], axis=1)
    reprojection_rmse = float(np.sqrt(np.mean(errors**2)))
    if reprojection_rmse > max_reprojection_error:
        return None
    rotation, _ = cv2.Rodrigues(rotation_vector)
    T_camera_world = np.eye(4, dtype=np.float32)
    T_camera_world[:3, :3] = rotation.astype(np.float32)
    T_camera_world[:3, 3] = translation_vector.reshape(3).astype(np.float32)
    T_world_camera = np.linalg.inv(T_camera_world).astype(np.float32)
    covariance = _estimate_pose_covariance(reprojection_rmse, len(inlier_indices))
    return VisualPoseMeasurement(
        timestamp=float(timestamp),
        T_world_camera=T_world_camera,
        covariance=covariance,
        reprojection_rmse=reprojection_rmse,
        inlier_count=len(inlier_indices),
        landmark_ids=tuple(observation.landmark_id for observation in observations),
    )


def _estimate_pose_covariance(reprojection_rmse: float, inlier_count: int) -> np.ndarray:
    """Produce a conservative diagonal quality estimate for downstream measurement gating."""
    image_error = max(reprojection_rmse, 0.25)
    support = max(inlier_count, 1)
    translation_variance = (0.05 * image_error) ** 2 / support
    rotation_variance = np.deg2rad(2.0 * image_error) ** 2 / support
    return np.diag([translation_variance] * 3 + [rotation_variance] * 3).astype(np.float32)
