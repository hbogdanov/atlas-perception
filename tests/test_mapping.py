import numpy as np
from pathlib import Path
import pytest

from src.mapping.pointcloud import PointCloudBuilder

from src.mapping.occupancy import pointcloud_to_occupancy


def test_pointcloud_to_occupancy_quantizes_points():
    points = np.array([[0.5, 0.5, 0.5]], dtype=np.float32)
    voxels = pointcloud_to_occupancy(points, voxel_size=0.25)
    assert voxels.tolist() == [[2, 2, 2]]


def test_pointcloud_builder_exports_ply(tmp_path: Path):
    pytest.importorskip("open3d")
    builder = PointCloudBuilder(
        {"fx": 1.0, "fy": 1.0, "cx": 0.0, "cy": 0.0},
        {"stride": 1, "max_points": 100},
    )
    depth = np.ones((2, 2), dtype=np.float32)
    rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    pose = type("Pose", (), {"matrix": np.eye(4, dtype=np.float32)})()
    builder.integrate(depth, rgb, pose)
    path = tmp_path / "frame_cloud.ply"
    builder.export(path)
    assert path.exists()
