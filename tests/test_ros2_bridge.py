import numpy as np

from src.ros2.nodes import AtlasRosBridge
from src.vision.landmark_pose import VisualPoseMeasurement


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
            "path_topic": "/atlas/path",
            "pointcloud_topic": "/atlas/pointcloud",
            "map_frame_id": "map",
            "camera_frame_id": "atlas_camera",
        }
    )

    bridge.publish_depth(np.ones((2, 2), dtype=np.float32), 1.0)
    bridge.publish_pose(DummyPose(), 1.0)
    bridge.publish_pointcloud(DummyPointCloud(), 1.0)
    bridge.publish_trajectory(type("Trajectory", (), {"poses": [DummyPose()]})(), 1.0)

    assert bridge.enabled is False
    assert bridge._node is None
    assert bridge.depth_publisher.last_message is not None
    assert bridge.pose_publisher.last_message is not None
    assert bridge.path_publisher.last_message is not None
    assert bridge.pointcloud_publisher.last_message is not None
    assert bridge.depth_publisher.last_message["header"]["frame_id"] == "atlas_camera"
    assert bridge.pose_publisher.last_message["header"]["frame_id"] == "map"
    assert bridge.pointcloud_publisher.last_message["header"]["frame_id"] == "map"


def test_bridge_publishes_visual_pose_with_covariance_without_ros_runtime():
    bridge = AtlasRosBridge(
        {
            "enabled": False,
            "depth_topic": "/atlas/depth",
            "pose_topic": "/atlas/pose",
            "path_topic": "/atlas/path",
            "pointcloud_topic": "/atlas/pointcloud",
            "map_frame_id": "map",
            "camera_frame_id": "atlas_camera",
        }
    )
    measurement = VisualPoseMeasurement(
        timestamp=2.0,
        T_world_camera=np.eye(4, dtype=np.float32),
        covariance=np.eye(6, dtype=np.float32),
        reprojection_rmse=0.4,
        inlier_count=8,
        landmark_ids=("tag-1", "tag-2"),
    )

    bridge.publish_visual_pose(measurement)

    assert bridge.visual_pose_publisher.topic == "/atlas/visual_pose"
    assert bridge.visual_pose_publisher.last_message["inlier_count"] == 8
