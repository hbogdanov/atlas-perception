import numpy as np

from src.utils.geometry import depth_to_pointcloud, homogenize, transform_points


def test_homogenize_adds_fourth_coordinate():
    points = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    result = homogenize(points)
    assert result.shape == (1, 4)
    assert result[0, 3] == 1.0


def test_depth_to_pointcloud_projects_valid_points():
    depth = np.ones((2, 2), dtype=np.float32)
    rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    intrinsics = {"fx": 1.0, "fy": 1.0, "cx": 0.0, "cy": 0.0}
    points, colors = depth_to_pointcloud(depth, rgb, intrinsics, stride=1)
    assert points.shape == (4, 3)
    assert colors.shape == (4, 3)


def test_transform_points_applies_translation():
    points = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
    transform = np.eye(4, dtype=np.float32)
    transform[0, 3] = 2.0
    moved = transform_points(points, transform)
    assert moved[0, 0] == 2.0
