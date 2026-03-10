from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.ros2.publishers import TopicPublisher
from src.ros2.transforms import rotation_matrix_to_quaternion

try:
    import open3d as o3d
except ImportError:  # pragma: no cover
    o3d = None

try:
    import rclpy
    from cv_bridge import CvBridge
    from geometry_msgs.msg import PoseStamped
    from nav_msgs.msg import Path as RosPath
    from rclpy.node import Node
    from sensor_msgs.msg import Image, PointCloud2, PointField
    from sensor_msgs_py import point_cloud2
    from std_msgs.msg import Header
    from builtin_interfaces.msg import Time as RosTime
except ImportError:  # pragma: no cover
    rclpy = None
    CvBridge = None
    PoseStamped = None
    RosPath = None
    Node = None
    Image = None
    PointCloud2 = None
    PointField = None
    point_cloud2 = None
    Header = None
    RosTime = None


@dataclass
class AtlasRosBridge:
    config: dict
    published: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.enabled = bool(self.config.get("enabled", False))
        self._bridge = CvBridge() if CvBridge is not None else None
        self._node = None
        self.depth_publisher = TopicPublisher(self.config["depth_topic"])
        self.pose_publisher = TopicPublisher(self.config["pose_topic"])
        self.path_publisher = TopicPublisher(self.config["path_topic"])
        self.pointcloud_publisher = TopicPublisher(self.config["pointcloud_topic"])
        if self.enabled and rclpy is not None:
            if not rclpy.ok():
                rclpy.init(args=None)
            self._node = Node("atlas_perception")
            self.depth_publisher.attach_ros(self._node, Image)
            self.pose_publisher.attach_ros(self._node, PoseStamped)
            self.path_publisher.attach_ros(self._node, RosPath)
            self.pointcloud_publisher.attach_ros(self._node, PointCloud2)

    def publish_depth(self, depth_map: np.ndarray, timestamp: float) -> None:
        header = self._header_from_timestamp(timestamp)
        message = {"header": header, "depth": depth_map}
        if self._node is not None and self._bridge is not None:
            ros_image = self._bridge.cv2_to_imgmsg(depth_map.astype(np.float32), encoding="32FC1")
            ros_image.header = header
            self.depth_publisher.publish(ros_image)
        else:
            self.depth_publisher.publish(message)
        self.published["depth"] = message

    def publish_pose(self, pose, timestamp: float) -> None:
        rotation = pose.matrix[:3, :3]
        quaternion = rotation_matrix_to_quaternion(rotation)
        header = self._header_from_timestamp(timestamp)
        if self._node is not None and PoseStamped is not None:
            message = PoseStamped()
            message.header = header
            message.pose.position.x = float(pose.matrix[0, 3])
            message.pose.position.y = float(pose.matrix[1, 3])
            message.pose.position.z = float(pose.matrix[2, 3])
            message.pose.orientation.x = float(quaternion[0])
            message.pose.orientation.y = float(quaternion[1])
            message.pose.orientation.z = float(quaternion[2])
            message.pose.orientation.w = float(quaternion[3])
            self.pose_publisher.publish(message)
        else:
            self.pose_publisher.publish(
                {
                    "header": header,
                    "pose": pose.matrix,
                    "orientation_source": "derived_from_pose_matrix",
                    "quaternion_xyzw": quaternion,
                }
            )
        self.published["pose"] = {"header": header, "pose": pose.matrix, "quaternion_xyzw": quaternion}

    def publish_trajectory(self, trajectory, timestamp: float) -> None:
        header = self._header_from_timestamp(timestamp)
        if self._node is not None and RosPath is not None and PoseStamped is not None:
            message = RosPath()
            message.header = header
            for pose in trajectory.poses:
                pose_msg = PoseStamped()
                pose_msg.header = self._header_from_timestamp(pose.timestamp)
                pose_msg.pose.position.x = float(pose.matrix[0, 3])
                pose_msg.pose.position.y = float(pose.matrix[1, 3])
                pose_msg.pose.position.z = float(pose.matrix[2, 3])
                quat = rotation_matrix_to_quaternion(pose.matrix[:3, :3])
                pose_msg.pose.orientation.x = float(quat[0])
                pose_msg.pose.orientation.y = float(quat[1])
                pose_msg.pose.orientation.z = float(quat[2])
                pose_msg.pose.orientation.w = float(quat[3])
                message.poses.append(pose_msg)
            self.path_publisher.publish(message)
        else:
            self.path_publisher.publish(
                {
                    "header": header,
                    "poses": [pose.matrix for pose in trajectory.poses],
                }
            )
        self.published["path"] = {"header": header, "num_poses": len(trajectory.poses)}

    def publish_pointcloud(self, point_cloud, timestamp: float) -> None:
        header = self._header_from_timestamp(timestamp)
        if self._node is not None and PointCloud2 is not None and point_cloud2 is not None and Header is not None:
            cloud_msg = point_cloud.to_ros_pointcloud2(header, point_cloud2, PointField)
            self.pointcloud_publisher.publish(cloud_msg)
        else:
            colors = getattr(point_cloud, "colors", np.empty((0, 3), dtype=np.float32))
            self.pointcloud_publisher.publish(
                {
                    "header": header,
                    "points": point_cloud.points,
                    "colors": colors,
                    "semantic_labels": getattr(point_cloud, "semantic_labels", None),
                }
            )
        self.published["pointcloud"] = {
            "header": header,
            "points": point_cloud.points,
            "colors": getattr(point_cloud, "colors", np.empty((0, 3), dtype=np.float32)),
            "semantic_labels": getattr(point_cloud, "semantic_labels", None),
        }

    def shutdown(self) -> None:
        if not self.enabled:
            return
        if self._node is not None:
            self._node.destroy_node()
        if rclpy is not None and rclpy.ok():
            rclpy.shutdown()

    def _header_from_timestamp(self, timestamp: float):
        if Header is None:
            return {"stamp": timestamp, "frame_id": self.config["frame_id"]}
        header = Header()
        header.frame_id = self.config["frame_id"]
        if RosTime is not None:
            secs = int(timestamp)
            nanos = int((timestamp - secs) * 1_000_000_000)
            header.stamp = RosTime(sec=secs, nanosec=nanos)
        return header
