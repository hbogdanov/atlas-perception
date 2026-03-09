from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.slam.odometry import PoseEstimate
from src.utils.geometry import depth_to_pointcloud, transform_points

try:
    import open3d as o3d
except ImportError:  # pragma: no cover
    o3d = None


@dataclass
class PointCloudData:
    points: np.ndarray
    colors: np.ndarray

    def to_ros_pointcloud2(self, header, point_cloud2_module, point_field_type):
        fields = [
            point_field_type(name="x", offset=0, datatype=point_field_type.FLOAT32, count=1),
            point_field_type(name="y", offset=4, datatype=point_field_type.FLOAT32, count=1),
            point_field_type(name="z", offset=8, datatype=point_field_type.FLOAT32, count=1),
            point_field_type(name="rgb", offset=12, datatype=point_field_type.UINT32, count=1),
        ]
        rows = []
        for point, color in zip(self.points.astype(np.float32), self.colors.astype(np.float32), strict=False):
            rgb_uint8 = np.clip(color * 255.0, 0, 255).astype(np.uint8)
            packed_rgb = int(rgb_uint8[0]) << 16 | int(rgb_uint8[1]) << 8 | int(rgb_uint8[2])
            rows.append([float(point[0]), float(point[1]), float(point[2]), packed_rgb])
        return point_cloud2_module.create_cloud(header, fields, rows)


class PointCloudBuilder:
    def __init__(self, camera_config: dict, mapping_config: dict) -> None:
        self.camera_config = camera_config
        self.mapping_config = mapping_config
        self._points = np.empty((0, 3), dtype=np.float32)
        self._colors = np.empty((0, 3), dtype=np.float32)

    @property
    def points(self) -> np.ndarray:
        return self._points

    @property
    def colors(self) -> np.ndarray:
        return self._colors

    def integrate(self, depth_map: np.ndarray, rgb: np.ndarray, pose: PoseEstimate) -> PointCloudData:
        stride = int(self.mapping_config.get("stride", 4))
        sample_points, sample_colors = depth_to_pointcloud(depth_map, rgb, self.camera_config, stride=stride)
        transformed = transform_points(sample_points, pose.matrix)
        max_points = int(self.mapping_config.get("max_points", 100000))
        self._points = np.vstack([self._points, transformed])[-max_points:]
        self._colors = np.vstack([self._colors, sample_colors])[-max_points:]
        return self.data()

    def data(self) -> PointCloudData:
        return PointCloudData(points=self._points.copy(), colors=self._colors.copy())

    def to_open3d(self):
        self._require_open3d()
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(self._points.astype(np.float64))
        if self._colors.size:
            cloud.colors = o3d.utility.Vector3dVector(self._colors.astype(np.float64))
        return cloud

    def to_ros_pointcloud2(self, header, point_cloud2_module, point_field_type):
        return self.data().to_ros_pointcloud2(header, point_cloud2_module, point_field_type)

    def export_ply(self, path: Path) -> None:
        self._require_open3d()
        path.parent.mkdir(parents=True, exist_ok=True)
        o3d.io.write_point_cloud(str(path), self.to_open3d())

    @staticmethod
    def _require_open3d() -> None:
        if o3d is None:
            raise RuntimeError("Open3D is required for point cloud export. Install dependencies from requirements.txt.")
