from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from src.ros2.transforms import quaternion_to_rotation_matrix
from src.slam.loop_closure import LoopClosureDetector
from src.slam.odometry import PoseEstimate, identity_pose
from src.slam.pose_graph import PoseGraph
from src.slam.trajectory import Trajectory

try:
    import rclpy
    from geometry_msgs.msg import PoseStamped
    from rclpy.node import Node
except ImportError:  # pragma: no cover
    rclpy = None
    PoseStamped = None
    Node = None


class SlamBackend(ABC):
    def __init__(self, config: dict) -> None:
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
        self.linear_step = float(config.get("linear_step", 0.05))
        self.lateral_amplitude = float(config.get("lateral_amplitude", 0.3))
        self.lateral_frequency = float(config.get("lateral_frequency", 0.1))
        self.vertical_amplitude = float(config.get("vertical_amplitude", 0.02))
        self.vertical_frequency = float(config.get("vertical_frequency", 0.05))
        self.yaw_rate = float(config.get("yaw_rate", 0.03))

    def update(self, rgb, depth=None, timestamp=None) -> PoseEstimate:
        del rgb, depth
        pose = identity_pose(float(timestamp or 0.0))
        step = float(self._step)
        yaw = step * self.yaw_rate
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
        pose.T_world_camera[0, 3] = step * self.linear_step
        pose.T_world_camera[1, 3] = float(np.sin(step * self.lateral_frequency) * self.lateral_amplitude)
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
        self.timeout_sec = float(config.get("timeout_sec", 0.0))
        self._latest: PoseEstimate | None = None
        self._owns_runtime = False
        self._node = None

    def initialize(self) -> None:
        if rclpy is None or Node is None or PoseStamped is None:
            raise RuntimeError("rtabmap mode requires ROS2 Python packages and geometry_msgs.")
        if not rclpy.ok():
            rclpy.init(args=None)
            self._owns_runtime = True
        self._node = Node("atlas_perception_slam")
        self._node.create_subscription(PoseStamped, self.pose_topic, self._pose_callback, 10)

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

    def _pose_callback(self, message: PoseStamped) -> None:
        transform = np.eye(4, dtype=np.float32)
        quaternion = np.array(
            [
                float(message.pose.orientation.x),
                float(message.pose.orientation.y),
                float(message.pose.orientation.z),
                float(message.pose.orientation.w),
            ],
            dtype=np.float32,
        )
        transform[:3, :3] = quaternion_to_rotation_matrix(quaternion)
        transform[0, 3] = float(message.pose.position.x)
        transform[1, 3] = float(message.pose.position.y)
        transform[2, 3] = float(message.pose.position.z)
        timestamp = float(message.header.stamp.sec) + float(message.header.stamp.nanosec) * 1e-9
        self._latest = PoseEstimate(T_world_camera=transform, timestamp=timestamp, tracking_ok=True)


class SlamWrapper:
    """Integration boundary for visual odometry or external SLAM systems."""

    def __init__(self, config: dict) -> None:
        self.config = config
        self.mode = str(config.get("mode", "disabled")).lower()
        self.trajectory = Trajectory()
        pose_graph_config = config.get("pose_graph", {})
        loop_closure = LoopClosureDetector(pose_graph_config.get("loop_closure", {}))
        self.pose_graph = PoseGraph(
            loop_closure_detector=loop_closure if bool(pose_graph_config.get("enabled", True)) else None
        )
        self.backend = self._build_backend()

    def update(self, image: np.ndarray, depth_map: np.ndarray, timestamp: float) -> PoseEstimate:
        pose = self.backend.update(image, depth_map, timestamp)
        self.trajectory.append(pose)
        self.pose_graph.append(pose)
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
        raise ValueError(f"Unsupported SLAM mode: {self.mode}")
