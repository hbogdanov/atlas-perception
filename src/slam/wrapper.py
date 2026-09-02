from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from src.ros2.transforms import quaternion_to_rotation_matrix
from src.slam.loop_closure import AppearanceLoopClosureDetector, LoopClosureDetector
from src.slam.odometry import PoseEstimate, identity_pose
from src.slam.pose_graph import PoseGraph
from src.slam.trajectory import Trajectory
from src.vision.pose_correction import VisualPoseCorrector

try:
    import rclpy
    from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
    from rclpy.node import Node
except ImportError:  # pragma: no cover
    rclpy = None
    PoseStamped = None
    PoseWithCovarianceStamped = None
    Node = None


class SlamBackend(ABC):
    def __init__(self, config: dict, visual_localization_config: dict | None = None) -> None:
        self.config = config

    def initialize(self) -> None:
        return None

    @abstractmethod
    def update(self, rgb, depth=None, timestamp=None) -> PoseEstimate:
        raise NotImplementedError

    def get_pose(self) -> PoseEstimate | None:
        return None

    def get_trajectory(self) -> list[PoseEstimate]:
        return []

    def set_pose(self, pose: PoseEstimate) -> None:
        del pose

    def shutdown(self) -> None:
        return None


class DisabledBackend(SlamBackend):
    def update(self, rgb, depth=None, timestamp=None) -> PoseEstimate:
        del rgb, depth
        return identity_pose(float(timestamp or 0.0))


class DummyBackend(SlamBackend):
    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self._step = 0
        self._latest: PoseEstimate | None = None
        self.path_radius_x = float(config.get("path_radius_x", 1.4))
        self.path_radius_y = float(config.get("path_radius_y", 1.4))
        self.path_frequency = float(config.get("path_frequency", 0.035))
        self.vertical_amplitude = float(config.get("vertical_amplitude", 0.02))
        self.vertical_frequency = float(config.get("vertical_frequency", 0.05))
        self.heading_lookahead = float(config.get("heading_lookahead", 0.15))

    def update(self, rgb, depth=None, timestamp=None) -> PoseEstimate:
        del rgb, depth
        pose = identity_pose(float(timestamp or 0.0))
        step = float(self._step)
        theta = step * self.path_frequency
        next_theta = theta + self.heading_lookahead
        x = self.path_radius_x * (1.0 - float(np.cos(theta)))
        y = self.path_radius_y * float(np.sin(theta))
        next_x = self.path_radius_x * (1.0 - float(np.cos(next_theta)))
        next_y = self.path_radius_y * float(np.sin(next_theta))
        yaw = float(np.arctan2(next_y - y, next_x - x))
        cos_yaw = float(np.cos(yaw))
        sin_yaw = float(np.sin(yaw))
        pose.T_world_camera[:3, :3] = np.array(
            [
                [cos_yaw, -sin_yaw, 0.0],
                [sin_yaw, cos_yaw, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        pose.T_world_camera[0, 3] = x
        pose.T_world_camera[1, 3] = y
        pose.T_world_camera[2, 3] = float(np.sin(step * self.vertical_frequency) * self.vertical_amplitude)
        self._step += 1
        self._latest = pose
        return pose

    def get_pose(self) -> PoseEstimate | None:
        return self._latest


class RtabmapBackend(SlamBackend):
    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.pose_topic = str(config.get("pose_topic", "/rtabmap/localization_pose"))
        self.pose_message_type = str(config.get("pose_message_type", "pose_with_covariance")).lower()
        self.timeout_sec = float(config.get("timeout_sec", 0.0))
        self._latest: PoseEstimate | None = None
        self._owns_runtime = False
        self._node = None

    def initialize(self) -> None:
        if rclpy is None or Node is None or PoseStamped is None or PoseWithCovarianceStamped is None:
            raise RuntimeError("rtabmap mode requires ROS2 Python packages and geometry_msgs.")
        if not rclpy.ok():
            rclpy.init(args=None)
            self._owns_runtime = True
        self._node = Node("atlas_perception_slam")
        message_type = PoseWithCovarianceStamped if self.pose_message_type == "pose_with_covariance" else PoseStamped
        self._node.create_subscription(message_type, self.pose_topic, self._pose_callback, 10)

    def update(self, rgb, depth=None, timestamp=None) -> PoseEstimate:
        del rgb, depth
        if self._node is None:
            self.initialize()
        rclpy.spin_once(self._node, timeout_sec=self.timeout_sec)
        if self._latest is None:
            pose = identity_pose(float(timestamp or 0.0))
            pose.tracking_ok = False
            return pose
        return self._latest

    def get_pose(self) -> PoseEstimate | None:
        return self._latest

    def shutdown(self) -> None:
        if self._node is not None:
            self._node.destroy_node()
        if self._owns_runtime and rclpy is not None and rclpy.ok():
            rclpy.shutdown()

    def _pose_callback(self, message) -> None:
        pose_message = message.pose.pose if hasattr(message.pose, "pose") else message.pose
        transform = np.eye(4, dtype=np.float32)
        quaternion = np.array(
            [
                float(pose_message.orientation.x),
                float(pose_message.orientation.y),
                float(pose_message.orientation.z),
                float(pose_message.orientation.w),
            ],
            dtype=np.float32,
        )
        transform[:3, :3] = quaternion_to_rotation_matrix(quaternion)
        transform[0, 3] = float(pose_message.position.x)
        transform[1, 3] = float(pose_message.position.y)
        transform[2, 3] = float(pose_message.position.z)
        timestamp = float(message.header.stamp.sec) + float(message.header.stamp.nanosec) * 1e-9
        self._latest = PoseEstimate(T_world_camera=transform, timestamp=timestamp, tracking_ok=True)


class GroundTruthBackend(SlamBackend):
    def update(self, rgb, depth=None, timestamp=None, pose_hint: np.ndarray | None = None) -> PoseEstimate:
        del rgb, depth
        if pose_hint is None:
            pose = identity_pose(float(timestamp or 0.0))
            pose.tracking_ok = False
            return pose
        return PoseEstimate(
            T_world_camera=np.asarray(pose_hint, dtype=np.float32).copy(),
            timestamp=float(timestamp or 0.0),
        )


class RgbdVisualOdometryBackend(SlamBackend):
    """Sparse calibrated RGB-D odometry using feature tracks and PnP-RANSAC.

    This intentionally consumes metric depth only. Relative monocular depth is
    not a valid source of metric camera motion for this backend.
    """

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        import cv2

        camera = config.get("camera", {})
        self._camera_matrix = np.array(
            [
                [camera.get("fx", 0.0), 0.0, camera.get("cx", 0.0)],
                [0.0, camera.get("fy", 0.0), camera.get("cy", 0.0)],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        if self._camera_matrix[0, 0] <= 0.0 or self._camera_matrix[1, 1] <= 0.0:
            raise ValueError("rgbd_vo mode requires positive camera intrinsics.")
        self._cv2 = cv2
        self._orb = cv2.ORB_create(nfeatures=int(config.get("max_features", 1200)))
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self._ratio_test = float(config.get("match_ratio", 0.75))
        self._min_correspondences = int(config.get("min_correspondences", 20))
        self._min_inliers = int(config.get("min_inliers", 12))
        self._max_depth_m = float(config.get("max_depth_m", 8.0))
        self._previous_keypoints = None
        self._previous_descriptors = None
        self._previous_depth = None
        self._latest = None
        self._loop_detector = AppearanceLoopClosureDetector(
            config.get("pose_graph", {}).get("loop_closure", {}), self._camera_matrix
        )
        self.last_loop_constraint = None

    def update(self, rgb, depth=None, timestamp=None) -> PoseEstimate:
        timestamp = float(timestamp or 0.0)
        if rgb is None or depth is None:
            return self._lost_pose(timestamp)
        gray = self._cv2.cvtColor(np.asarray(rgb), self._cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self._orb.detectAndCompute(gray, None)
        self.last_loop_constraint = self._loop_detector.detect(keypoints, descriptors, depth, timestamp)
        if self._latest is None:
            self._store_frame(keypoints, descriptors, depth)
            self._latest = identity_pose(timestamp)
            self._latest.tracking_ok = False
            return self._latest
        if descriptors is None or self._previous_descriptors is None:
            self._store_frame(keypoints, descriptors, depth)
            return self._lost_pose(timestamp)
        object_points, image_points = self._matched_geometry(keypoints, descriptors)
        self._store_frame(keypoints, descriptors, depth)
        if len(object_points) < self._min_correspondences:
            return self._lost_pose(timestamp)
        success, rvec, translation, inliers = self._cv2.solvePnPRansac(
            object_points,
            image_points,
            self._camera_matrix,
            None,
            iterationsCount=100,
            reprojectionError=2.0,
            confidence=0.999,
            flags=self._cv2.SOLVEPNP_EPNP,
        )
        if not success or inliers is None or len(inliers) < self._min_inliers:
            return self._lost_pose(timestamp)
        rotation, _ = self._cv2.Rodrigues(rvec)
        T_current_previous = np.eye(4, dtype=np.float32)
        T_current_previous[:3, :3] = rotation.astype(np.float32)
        T_current_previous[:3, 3] = translation.reshape(3).astype(np.float32)
        world_current = self._latest.matrix @ np.linalg.inv(T_current_previous)
        self._latest = PoseEstimate(world_current.astype(np.float32), timestamp, tracking_ok=True)
        return self._latest

    def get_pose(self) -> PoseEstimate | None:
        return self._latest

    def set_pose(self, pose: PoseEstimate) -> None:
        self._latest = pose

    def _matched_geometry(self, keypoints, descriptors) -> tuple[np.ndarray, np.ndarray]:
        matches = self._matcher.knnMatch(self._previous_descriptors, descriptors, k=2)
        object_points, image_points = [], []
        for pair in matches:
            if len(pair) != 2 or pair[0].distance >= self._ratio_test * pair[1].distance:
                continue
            previous = self._previous_keypoints[pair[0].queryIdx].pt
            current = keypoints[pair[0].trainIdx].pt
            u, v = int(round(previous[0])), int(round(previous[1]))
            if not (0 <= v < self._previous_depth.shape[0] and 0 <= u < self._previous_depth.shape[1]):
                continue
            z = float(self._previous_depth[v, u])
            if not np.isfinite(z) or z <= 0.0 or z > self._max_depth_m:
                continue
            x = (previous[0] - self._camera_matrix[0, 2]) * z / self._camera_matrix[0, 0]
            y = (previous[1] - self._camera_matrix[1, 2]) * z / self._camera_matrix[1, 1]
            object_points.append((x, y, z))
            image_points.append(current)
        return np.asarray(object_points, dtype=np.float32), np.asarray(image_points, dtype=np.float32)

    def _store_frame(self, keypoints, descriptors, depth) -> None:
        self._previous_keypoints = keypoints
        self._previous_descriptors = descriptors
        self._previous_depth = np.asarray(depth, dtype=np.float32).copy()

    def _lost_pose(self, timestamp: float) -> PoseEstimate:
        if self._latest is None:
            self._latest = identity_pose(timestamp)
        return PoseEstimate(self._latest.matrix.copy(), timestamp, tracking_ok=False)


class SlamWrapper:
    """Integration boundary for visual odometry or external SLAM systems."""

    def __init__(self, config: dict, visual_localization_config: dict | None = None) -> None:
        self.config = config
        self.mode = str(config.get("mode", "disabled")).lower()
        self.trajectory = Trajectory()
        pose_graph_config = config.get("pose_graph", {})
        loop_closure = LoopClosureDetector(pose_graph_config.get("loop_closure", {}))
        self.pose_graph = PoseGraph(
            loop_closure_detector=(
                loop_closure
                if bool(pose_graph_config.get("enabled", True))
                and self.mode != "rgbd_vo"
                and bool(pose_graph_config.get("loop_closure", {}).get("proximity_enabled", True))
                else None
            )
        )
        self.backend = self._build_backend()
        correction_config = (visual_localization_config or {}).get("pose_correction", {})
        self.visual_pose_corrector = VisualPoseCorrector(correction_config)
        self.last_visual_correction = None

    def update(
        self,
        image: np.ndarray,
        depth_map: np.ndarray,
        timestamp: float,
        pose_hint: np.ndarray | None = None,
        visual_measurement=None,
    ) -> PoseEstimate:
        try:
            pose = self.backend.update(image, depth_map, timestamp, pose_hint=pose_hint)
        except TypeError:
            pose = self.backend.update(image, depth_map, timestamp)
        self.last_visual_correction = self.visual_pose_corrector.correct(pose, visual_measurement, timestamp)
        pose = self.last_visual_correction.pose
        self.trajectory.append(pose)
        loop_constraint = getattr(self.backend, "last_loop_constraint", None)
        loop_applied = self.pose_graph.append(pose, loop_constraint=loop_constraint)
        if loop_applied:
            self.trajectory.poses = [
                PoseEstimate(node.pose_matrix.copy(), node.timestamp, self.trajectory.poses[index].tracking_ok)
                for index, node in enumerate(self.pose_graph.nodes)
            ]
            pose = self.trajectory.poses[-1]
            self.backend.set_pose(pose)
        return pose

    def export_trajectory(self, path: Path) -> None:
        self.trajectory.export(path)
        self.trajectory.export_json(path.with_suffix(".json"))
        self.trajectory.export_csv(path.with_suffix(".csv"))
        self.trajectory.export_plot(path.with_name("trajectory_plot.png"))
        self.pose_graph.export_json(path.with_name("pose_graph.json"))
        self.pose_graph.export_csv(path.with_name("pose_graph_edges.csv"))

    def shutdown(self) -> None:
        self.backend.shutdown()

    def _build_backend(self) -> SlamBackend:
        if self.mode == "disabled":
            return DisabledBackend(self.config)
        if self.mode == "dummy":
            return DummyBackend(self.config)
        if self.mode == "rtabmap":
            return RtabmapBackend(self.config)
        if self.mode == "groundtruth":
            return GroundTruthBackend(self.config)
        if self.mode == "rgbd_vo":
            return RgbdVisualOdometryBackend(self.config)
        raise ValueError(f"Unsupported SLAM mode: {self.mode}")
