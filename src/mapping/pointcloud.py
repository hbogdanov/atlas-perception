from __future__ import annotations

from pathlib import Path

import numpy as np

from src.slam.odometry import PoseEstimate
from src.utils.geometry import depth_to_pointcloud, transform_points

try:
    import open3d as o3d
except ImportError:  # pragma: no cover
    o3d = None


class PointCloudBuilder:
    def __init__(self, camera_config: dict, mapping_config: dict) -> None:
        self.camera_config = camera_config
        self.mapping_config = mapping_config
        self._points = np.empty((0, 3), dtype=np.float32)
        self._colors = np.empty((0, 3), dtype=np.float32)

    def integrate(self, depth_map: np.ndarray, rgb: np.ndarray, pose: PoseEstimate):
        self._require_open3d()
        stride = int(self.mapping_config.get("stride", 4))
        sample_points, sample_colors = depth_to_pointcloud(depth_map, rgb, self.camera_config, stride=stride)
        transformed = transform_points(sample_points, pose.matrix)
        max_points = int(self.mapping_config.get("max_points", 100000))
        self._points = np.vstack([self._points, transformed])[-max_points:]
        self._colors = np.vstack([self._colors, sample_colors])[-max_points:]
        return self.as_open3d()

    def as_open3d(self):
        self._require_open3d()
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(self._points.astype(np.float64))
        if self._colors.size:
            cloud.colors = o3d.utility.Vector3dVector(self._colors.astype(np.float64))
        return cloud

    def export(self, path: Path) -> None:
        self._require_open3d()
        path.parent.mkdir(parents=True, exist_ok=True)
        o3d.io.write_point_cloud(str(path), self.as_open3d())

    @staticmethod
    def _require_open3d() -> None:
        if o3d is None:
            raise RuntimeError("Open3D is required for point cloud generation. Install dependencies from requirements.txt.")
