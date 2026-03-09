import numpy as np

from src.ros2.nodes import AtlasRosBridge


class DummyPointCloud:
    def __init__(self) -> None:
        self.points = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)


class DummyPose:
    def __init__(self) -> None:
        self.matrix = np.eye(4, dtype=np.float32)


def test_bridge_respects_disabled_flag_without_ros_runtime():
    bridge = AtlasRosBridge(
        {
            "enabled": False,
            "depth_topic": "/atlas/depth",
            "pose_topic": "/atlas/pose",
            "pointcloud_topic": "/atlas/pointcloud",
            "frame_id": "atlas_camera",
        }
    )

    bridge.publish_depth(np.ones((2, 2), dtype=np.float32), 1.0)
    bridge.publish_pose(DummyPose(), 1.0)
    bridge.publish_pointcloud(DummyPointCloud(), 1.0)

    assert bridge.enabled is False
    assert bridge._node is None
    assert bridge.depth_publisher.last_message is not None
    assert bridge.pose_publisher.last_message is not None
    assert bridge.pointcloud_publisher.last_message is not None
