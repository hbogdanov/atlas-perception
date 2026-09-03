from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.slam.odometry import PoseEstimate


@dataclass
class LoopClosureConstraint:
    source_index: int
    target_index: int
    distance: float
    timestamp: float
    transform: np.ndarray | None = None
    inlier_count: int = 0
    reprojection_rmse: float = 0.0


class LoopClosureDetector:
    def __init__(self, config: dict | None = None) -> None:
        config = config or {}
        self.enabled = bool(config.get("enabled", True))
        self.min_node_gap = int(config.get("min_node_gap", 15))
        self.distance_threshold = float(config.get("distance_threshold", 0.15))

    def detect(self, poses: list[PoseEstimate]) -> LoopClosureConstraint | None:
        if not self.enabled or len(poses) <= self.min_node_gap:
            return None
        current_index = len(poses) - 1
        current_pose = poses[current_index]
        current_position = current_pose.matrix[:3, 3]
        best_constraint: LoopClosureConstraint | None = None
        for target_index, target_pose in enumerate(poses[: current_index - self.min_node_gap + 1]):
            distance = float(np.linalg.norm(current_position - target_pose.matrix[:3, 3]))
            if distance > self.distance_threshold:
                continue
            if best_constraint is None or distance < best_constraint.distance:
                best_constraint = LoopClosureConstraint(
                    source_index=current_index,
                    target_index=target_index,
                    distance=distance,
                    timestamp=float(current_pose.timestamp),
                )
        return best_constraint


class AppearanceLoopClosureDetector:
    """Verify RGB-D loop candidates with ORB matching and depth-backed PnP."""

    def __init__(self, config: dict | None, camera_matrix: np.ndarray) -> None:
        import cv2

        settings = config or {}
        self.enabled = bool(settings.get("enabled", False))
        self.min_node_gap = int(settings.get("min_node_gap", 30))
        self.max_candidates = int(settings.get("max_candidates", 3))
        self.min_matches = int(settings.get("min_matches", 30))
        self.min_inliers = int(settings.get("min_inliers", 15))
        self.max_reprojection_error = float(settings.get("max_reprojection_error", 2.5))
        self.match_ratio = float(settings.get("match_ratio", 0.75))
        self.max_depth_m = float(settings.get("max_depth_m", 8.0))
        self._camera_matrix = camera_matrix.astype(np.float32)
        self._cv2 = cv2
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self._keyframes: list[tuple[list, np.ndarray | None, np.ndarray]] = []

    def update_camera_intrinsics(self, intrinsics: dict) -> None:
        self._camera_matrix = np.array(
            [
                [intrinsics["fx"], 0.0, intrinsics["cx"]],
                [0.0, intrinsics["fy"], intrinsics["cy"]],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

    def detect(self, keypoints, descriptors, depth: np.ndarray, timestamp: float) -> LoopClosureConstraint | None:
        current_index = len(self._keyframes)
        result = None
        if self.enabled and descriptors is not None and current_index >= self.min_node_gap:
            candidates = self._rank_candidates(descriptors, current_index)
            for target_index in candidates:
                result = self._verify(target_index, keypoints, descriptors, timestamp)
                if result is not None:
                    break
        self._keyframes.append((keypoints, descriptors, np.asarray(depth, dtype=np.float32).copy()))
        return result

    def _rank_candidates(self, descriptors: np.ndarray, current_index: int) -> list[int]:
        scored = []
        for index, (_, candidate_descriptors, _) in enumerate(self._keyframes[: current_index - self.min_node_gap + 1]):
            if candidate_descriptors is None:
                continue
            matches = self._matcher.knnMatch(candidate_descriptors, descriptors, k=2)
            score = sum(len(pair) == 2 and pair[0].distance < self.match_ratio * pair[1].distance for pair in matches)
            if score >= self.min_matches:
                scored.append((score, index))
        return [index for _, index in sorted(scored, reverse=True)[: self.max_candidates]]

    def _verify(self, target_index: int, current_keypoints, current_descriptors, timestamp: float):
        target_keypoints, target_descriptors, target_depth = self._keyframes[target_index]
        matches = self._matcher.knnMatch(target_descriptors, current_descriptors, k=2)
        object_points, image_points = [], []
        for pair in matches:
            if len(pair) != 2 or pair[0].distance >= self.match_ratio * pair[1].distance:
                continue
            target = target_keypoints[pair[0].queryIdx].pt
            current = current_keypoints[pair[0].trainIdx].pt
            u, v = int(round(target[0])), int(round(target[1]))
            if not (0 <= v < target_depth.shape[0] and 0 <= u < target_depth.shape[1]):
                continue
            z = float(target_depth[v, u])
            if not np.isfinite(z) or z <= 0.0 or z > self.max_depth_m:
                continue
            object_points.append(
                (
                    (target[0] - self._camera_matrix[0, 2]) * z / self._camera_matrix[0, 0],
                    (target[1] - self._camera_matrix[1, 2]) * z / self._camera_matrix[1, 1],
                    z,
                )
            )
            image_points.append(current)
        if len(object_points) < self.min_matches:
            return None
        success, rvec, translation, inliers = self._cv2.solvePnPRansac(
            np.asarray(object_points, dtype=np.float32),
            np.asarray(image_points, dtype=np.float32),
            self._camera_matrix,
            None,
            iterationsCount=100,
            reprojectionError=self.max_reprojection_error,
            confidence=0.999,
            flags=self._cv2.SOLVEPNP_EPNP,
        )
        if not success or inliers is None or len(inliers) < self.min_inliers:
            return None
        projected, _ = self._cv2.projectPoints(
            np.asarray(object_points, dtype=np.float32)[inliers.reshape(-1)],
            rvec,
            translation,
            self._camera_matrix,
            None,
        )
        errors = np.linalg.norm(
            projected.reshape(-1, 2) - np.asarray(image_points, dtype=np.float32)[inliers.reshape(-1)], axis=1
        )
        rmse = float(np.sqrt(np.mean(errors**2)))
        rotation, _ = self._cv2.Rodrigues(rvec)
        current_target = np.eye(4, dtype=np.float32)
        current_target[:3, :3] = rotation.astype(np.float32)
        current_target[:3, 3] = translation.reshape(3).astype(np.float32)
        return LoopClosureConstraint(
            source_index=len(self._keyframes),
            target_index=target_index,
            distance=0.0,
            timestamp=float(timestamp),
            transform=np.linalg.inv(current_target).astype(np.float32),
            inlier_count=len(inliers),
            reprojection_rmse=rmse,
        )
