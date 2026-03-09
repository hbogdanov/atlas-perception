import numpy as np

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
        metrics={"depth_ms": 1.0, "mapping_ms": 2.0, "fps": 10.0, "points": 123},
        runtime={
            "input_mode": "ros2",
            "slam_mode": "dummy",
            "frame_id": "gazebo_camera",
            "depth_topic": "/atlas/depth",
            "pose_topic": "/atlas/pose",
            "path_topic": "/atlas/path",
            "pointcloud_topic": "/atlas/pointcloud",
        },
        frame_size=(1280, 720),
    )

    assert frame.shape == (720, 1280, 3)
