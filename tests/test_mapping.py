from pathlib import Path

import numpy as np
import pytest

from src.mapping.occupancy import pointcloud_to_occupancy
from src.mapping.pointcloud import PointCloudBuilder
from src.semantics.models import SemanticPrediction


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
    point_cloud = builder.integrate(depth, rgb, pose)
    assert point_cloud.points.shape == (4, 3)
    assert point_cloud.colors.shape == (4, 3)
    path = tmp_path / "frame_cloud.ply"
    builder.export_ply(path)
    assert path.exists()


def test_pointcloud_builder_integrates_without_open3d():
    builder = PointCloudBuilder(
        {"fx": 1.0, "fy": 1.0, "cx": 0.0, "cy": 0.0},
        {"stride": 1, "max_points": 100},
    )
    depth = np.ones((2, 2), dtype=np.float32)
    rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    pose = type("Pose", (), {"matrix": np.eye(4, dtype=np.float32)})()
    point_cloud = builder.integrate(depth, rgb, pose)
    assert point_cloud.points.shape == (4, 3)
    assert builder.points.shape == (4, 3)


def test_pointcloud_builder_rejects_unknown_representation():
    with pytest.raises(ValueError):
        PointCloudBuilder(
            {"fx": 1.0, "fy": 1.0, "cx": 0.0, "cy": 0.0},
            {"representation": "meshlets", "stride": 1, "max_points": 100},
        )


def test_tsdf_builder_exports_mesh_and_cloud(tmp_path: Path):
    pytest.importorskip("open3d")
    builder = PointCloudBuilder(
        {"fx": 100.0, "fy": 100.0, "cx": 1.0, "cy": 1.0},
        {
            "representation": "tsdf",
            "stride": 1,
            "max_points": 100,
            "tsdf_voxel_length": 0.05,
            "tsdf_sdf_trunc": 0.1,
            "tsdf_depth_scale": 1.0,
            "tsdf_depth_trunc": 3.0,
        },
    )
    depth = np.ones((4, 4), dtype=np.float32)
    rgb = np.full((4, 4, 3), 255, dtype=np.uint8)
    pose = type("Pose", (), {"matrix": np.eye(4, dtype=np.float32)})()
    point_cloud = builder.integrate(depth, rgb, pose)
    assert point_cloud.points.shape[1] == 3
    ply_path = tmp_path / "tsdf_cloud.ply"
    mesh_path = tmp_path / "tsdf_mesh.ply"
    builder.export_ply(ply_path)
    builder.export_mesh(mesh_path)
    assert ply_path.exists()


def test_pointcloud_builder_ros_adapter_includes_color():
    builder = PointCloudBuilder(
        {"fx": 1.0, "fy": 1.0, "cx": 0.0, "cy": 0.0},
        {"stride": 1, "max_points": 100},
    )
    depth = np.ones((1, 1), dtype=np.float32)
    rgb = np.array([[[255, 0, 0]]], dtype=np.uint8)
    pose = type("Pose", (), {"matrix": np.eye(4, dtype=np.float32)})()
    builder.integrate(depth, rgb, pose)

    class FakePointField:
        FLOAT32 = 7
        UINT32 = 6

        def __init__(self, name, offset, datatype, count):
            self.name = name
            self.offset = offset
            self.datatype = datatype
            self.count = count

    class FakePointCloud2Module:
        @staticmethod
        def create_cloud(header, fields, rows):
            return {"header": header, "fields": fields, "rows": rows}

    msg = builder.to_ros_pointcloud2({"frame_id": "atlas"}, FakePointCloud2Module, FakePointField)
    assert [field.name for field in msg["fields"]] == ["x", "y", "z", "rgb", "label"]
    assert msg["rows"][0][3] == 0xFF0000


def test_pointcloud_builder_produces_semantic_labels_and_colors():
    builder = PointCloudBuilder(
        {"fx": 1.0, "fy": 1.0, "cx": 0.0, "cy": 0.0},
        {"stride": 1, "max_points": 100, "semantic_color_fusion": True},
    )
    depth = np.ones((2, 2), dtype=np.float32)
    rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    semantics = SemanticPrediction(
        labels=np.array([[0, 1], [-1, 1]], dtype=np.int32),
        class_names={0: "desk", 1: "monitor"},
    )
    pose = type("Pose", (), {"matrix": np.eye(4, dtype=np.float32)})()
    point_cloud = builder.integrate(depth, rgb, pose, semantics=semantics)
    assert point_cloud.semantic_labels is not None
    assert point_cloud.semantic_labels.tolist() == [0, 1, -1, 1]
    assert point_cloud.semantic_colors is not None
    assert point_cloud.class_names == {0: "desk", 1: "monitor"}
