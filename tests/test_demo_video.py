import numpy as np

from src.mapping.pointcloud import PointCloudData
from src.slam.odometry import PoseEstimate
from src.slam.trajectory import Trajectory
from src.utils.demo_video import DemoVideoRecorder


def test_compose_frame_returns_expected_shape():
    rgb = np.zeros((240, 320, 3), dtype=np.uint8)
    depth = np.ones((240, 320), dtype=np.float32)
    pose = PoseEstimate(T_world_camera=np.eye(4, dtype=np.float32), timestamp=1.0, tracking_ok=True)
    trajectory = Trajectory([pose])

    frame = DemoVideoRecorder.compose_frame(
        rgb=rgb,
        depth_map=depth,
        trajectory=trajectory,
        pose=pose,
        metrics={"depth_ms": 1.0, "mapping_ms": 2.0, "fps": 10.0, "points": 123, "frames": 4},
        runtime={
            "input_mode": "ros2",
            "slam_mode": "dummy",
            "frame_id": "gazebo_camera",
            "depth_topic": "/atlas/depth",
            "pose_topic": "/atlas/pose",
            "path_topic": "/atlas/path",
            "pointcloud_topic": "/atlas/pointcloud",
            "semantic_title": "Semantic Overlay",
            "map_title": "Fused Point Cloud Map",
        },
        frame_size=(1280, 720),
    )

    assert frame.shape == (720, 1280, 3)


def test_render_topdown_map_draws_fused_points():
    pose = PoseEstimate(T_world_camera=np.eye(4, dtype=np.float32), timestamp=1.0, tracking_ok=True)
    cloud = PointCloudData(
        points=np.array([[0.0, 0.0, 1.0], [0.2, 0.0, 1.4], [0.4, 0.1, 1.8]], dtype=np.float32),
        colors=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32),
    )
    panel = DemoVideoRecorder.render_topdown_map(
        cloud,
        pose,
        {"mapping_ms": 2.0, "fps": 10.0, "points": 3, "frames": 2},
        {"slam_mode": "disabled"},
    )
    assert panel.shape[0] > 0
    assert panel.shape[1] > 0
    assert np.count_nonzero(panel != 250) > 0
