import numpy as np

from src.ros2.nodes import AtlasRosBridge


class DummyPose:
    def __init__(self, matrix: np.ndarray) -> None:
        self.matrix = matrix


def test_publish_pose_fallback_includes_quaternion():
    bridge = AtlasRosBridge(
        {
            "enabled": False,
            "depth_topic": "/atlas/depth",
            "pose_topic": "/atlas/pose",
            "pointcloud_topic": "/atlas/pointcloud",
            "frame_id": "atlas_camera",
        }
    )
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, :3] = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    bridge.publish_pose(DummyPose(matrix), 1.0)

    message = bridge.pose_publisher.last_message
    assert message["header"]["frame_id"] == "atlas_camera"
    assert message["orientation_source"] == "derived_from_pose_matrix"
    assert np.allclose(message["quaternion_xyzw"], np.array([0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)], dtype=np.float32))
