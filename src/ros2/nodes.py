from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.ros2.publishers import TopicPublisher

try:
    import open3d as o3d
except ImportError:  # pragma: no cover
    o3d = None

try:
    import rclpy
    from cv_bridge import CvBridge
    from geometry_msgs.msg import PoseStamped
    from rclpy.node import Node
    from sensor_msgs.msg import Image, PointCloud2
    from sensor_msgs_py import point_cloud2
    from std_msgs.msg import Header
except ImportError:  # pragma: no cover
    rclpy = None
    CvBridge = None
    PoseStamped = None
    Node = None
    Image = None
    PointCloud2 = None
    point_cloud2 = None
    Header = None


@dataclass
class AtlasRosBridge:
    config: dict
    published: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._bridge = CvBridge() if CvBridge is not None else None
        self._node = None
        self.depth_publisher = TopicPublisher(self.config["depth_topic"])
        self.pose_publisher = TopicPublisher(self.config["pose_topic"])
        self.pointcloud_publisher = TopicPublisher(self.config["pointcloud_topic"])
        if rclpy is not None:
            if not rclpy.ok():
                rclpy.init(args=None)
            self._node = Node("atlas_perception")
            self.depth_publisher.attach_ros(self._node, Image)
            self.pose_publisher.attach_ros(self._node, PoseStamped)
            self.pointcloud_publisher.attach_ros(self._node, PointCloud2)

    def publish_depth(self, depth_map: np.ndarray, timestamp: float) -> None:
        message = {"timestamp": timestamp, "depth": depth_map}
        if self._node is not None and self._bridge is not None:
            ros_image = self._bridge.cv2_to_imgmsg(depth_map.astype(np.float32), encoding="32FC1")
            ros_image.header.frame_id = self.config["frame_id"]
            self.depth_publisher.publish(ros_image)
        else:
            self.depth_publisher.publish(message)
        self.published["depth"] = depth_map

    def publish_pose(self, pose, timestamp: float) -> None:
        if self._node is not None and PoseStamped is not None:
            message = PoseStamped()
            message.header.frame_id = self.config["frame_id"]
            message.pose.position.x = float(pose.matrix[0, 3])
            message.pose.position.y = float(pose.matrix[1, 3])
            message.pose.position.z = float(pose.matrix[2, 3])
            message.pose.orientation.w = 1.0
            self.pose_publisher.publish(message)
        else:
            self.pose_publisher.publish({"timestamp": timestamp, "pose": pose.matrix})
        self.published["pose"] = pose.matrix

    def publish_pointcloud(self, point_cloud, timestamp: float) -> None:
        points = np.asarray(point_cloud.points, dtype=np.float32)
        if self._node is not None and PointCloud2 is not None and point_cloud2 is not None and Header is not None:
            header = Header()
            header.stamp = self._node.get_clock().now().to_msg()
            header.frame_id = self.config["frame_id"]
            cloud_msg = point_cloud2.create_cloud_xyz32(header, points.tolist())
            self.pointcloud_publisher.publish(cloud_msg)
        else:
            self.pointcloud_publisher.publish({"timestamp": timestamp, "points": points})
        self.published["pointcloud"] = points

    def shutdown(self) -> None:
        if self._node is not None:
            self._node.destroy_node()
        if rclpy is not None and rclpy.ok():
            rclpy.shutdown()
