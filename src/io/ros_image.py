from __future__ import annotations

from dataclasses import dataclass
from time import sleep, time

import numpy as np

from src.io.types import FramePacket

try:
    import rclpy
    from cv_bridge import CvBridge
    from rclpy.node import Node
    from sensor_msgs.msg import Image
except ImportError:  # pragma: no cover
    rclpy = None
    CvBridge = None
    Node = None
    Image = None


@dataclass
class RosImageMessage:
    topic: str
    image: np.ndarray
    timestamp: float


@dataclass
class RosCameraInfoMessage:
    topic: str
    fx: float
    fy: float
    cx: float
    cy: float
    timestamp: float


class RosImageSubscriber:
    """ROS2 image subscription wrapper that converts incoming images to OpenCV frames."""

    def __init__(self, topic: str, camera_info_topic: str | None = None, timeout_sec: float = 5.0) -> None:
        if rclpy is None or CvBridge is None or Node is None or Image is None:
            raise RuntimeError("ROS2 image ingestion requires rclpy, sensor_msgs, and cv_bridge.")
        self._owns_runtime = False
        if not rclpy.ok():
            rclpy.init(args=None)
            self._owns_runtime = True
        self.topic = topic
        self.camera_info_topic = camera_info_topic
        self.timeout_sec = timeout_sec
        self._bridge = CvBridge()
        self._node = Node("atlas_perception_image_source")
        self._latest: RosImageMessage | None = None
        self._latest_camera_info: RosCameraInfoMessage | None = None
        self._sequence = 0
        self._last_yielded_sequence = -1
        self._subscription = self._node.create_subscription(Image, topic, self._callback, 10)
        self._camera_info_subscription = None
        if camera_info_topic:
            from sensor_msgs.msg import CameraInfo

            self._camera_info_subscription = self._node.create_subscription(
                CameraInfo,
                camera_info_topic,
                self._camera_info_callback,
                10,
            )

    def latest(self) -> RosImageMessage | None:
        return self._latest

    def get_camera_intrinsics(self) -> dict | None:
        if self._latest_camera_info is None:
            return None
        return {
            "fx": self._latest_camera_info.fx,
            "fy": self._latest_camera_info.fy,
            "cx": self._latest_camera_info.cx,
            "cy": self._latest_camera_info.cy,
        }

    def wait_for_frame(self) -> FramePacket:
        start = time()
        while time() - start <= self.timeout_sec:
            rclpy.spin_once(self._node, timeout_sec=0.1)
            message = self.latest()
            if message is not None:
                return FramePacket(image=message.image, timestamp=message.timestamp)
            sleep(0.01)
        raise TimeoutError(f"Timed out waiting for ROS2 image on topic {self.topic} after {self.timeout_sec:.1f}s.")

    def frames(self):
        while True:
            rclpy.spin_once(self._node, timeout_sec=0.1)
            if self._latest is None or self._sequence == self._last_yielded_sequence:
                if self._latest is None:
                    continue
                continue
            self._last_yielded_sequence = self._sequence
            yield FramePacket(image=self._latest.image, timestamp=self._latest.timestamp)

    def close(self) -> None:
        if self._node is not None:
            self._node.destroy_node()
        if self._owns_runtime and rclpy is not None and rclpy.ok():
            rclpy.shutdown()

    def _callback(self, message: Image) -> None:
        image = self._bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
        timestamp = float(message.header.stamp.sec) + float(message.header.stamp.nanosec) * 1e-9
        if timestamp == 0.0:
            timestamp = time()
        self._latest = RosImageMessage(topic=self.topic, image=image, timestamp=timestamp)
        self._sequence += 1

    def _camera_info_callback(self, message) -> None:
        timestamp = float(message.header.stamp.sec) + float(message.header.stamp.nanosec) * 1e-9
        if timestamp == 0.0:
            timestamp = time()
        self._latest_camera_info = RosCameraInfoMessage(
            topic=str(self.camera_info_topic),
            fx=float(message.k[0]),
            fy=float(message.k[4]),
            cx=float(message.k[2]),
            cy=float(message.k[5]),
            timestamp=timestamp,
        )
