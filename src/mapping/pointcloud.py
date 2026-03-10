from __future__ import annotations

from abc import ABC, abstractmethod
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


class MappingBackend(ABC):
    def __init__(self, camera_config: dict, mapping_config: dict) -> None:
        self.camera_config = camera_config
        self.mapping_config = mapping_config

    def update_camera_intrinsics(self, intrinsics: dict | None) -> None:
        if not intrinsics:
            return
        for key in ("fx", "fy", "cx", "cy"):
            if key in intrinsics:
                self.camera_config[key] = float(intrinsics[key])

    @property
    @abstractmethod
    def points(self) -> np.ndarray:
        raise NotImplementedError

    @property
    @abstractmethod
    def colors(self) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def integrate(self, depth_map: np.ndarray, rgb: np.ndarray, pose: PoseEstimate) -> PointCloudData:
        raise NotImplementedError

    @abstractmethod
    def data(self) -> PointCloudData:
        raise NotImplementedError

    @abstractmethod
    def to_open3d(self):
        raise NotImplementedError

    def export_ply(self, path: Path) -> None:
        _require_open3d()
        path.parent.mkdir(parents=True, exist_ok=True)
        o3d.io.write_point_cloud(str(path), self.to_open3d())


class PointCloudFusionBackend(MappingBackend):
    def __init__(self, camera_config: dict, mapping_config: dict) -> None:
        super().__init__(camera_config, mapping_config)
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
        _require_open3d()
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(self._points.astype(np.float64))
        if self._colors.size:
            cloud.colors = o3d.utility.Vector3dVector(self._colors.astype(np.float64))
        return cloud


class TsdfFusionBackend(MappingBackend):
    def __init__(self, camera_config: dict, mapping_config: dict) -> None:
        super().__init__(camera_config, mapping_config)
        _require_open3d()
        self._points = np.empty((0, 3), dtype=np.float32)
        self._colors = np.empty((0, 3), dtype=np.float32)
        self._volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=float(mapping_config.get("tsdf_voxel_length", 0.04)),
            sdf_trunc=float(mapping_config.get("tsdf_sdf_trunc", 0.08)),
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
        )
        self._depth_scale = float(mapping_config.get("tsdf_depth_scale", 1.0))
        self._depth_trunc = float(mapping_config.get("tsdf_depth_trunc", 4.0))

    @property
    def points(self) -> np.ndarray:
        return self._points

    @property
    def colors(self) -> np.ndarray:
        return self._colors

    def integrate(self, depth_map: np.ndarray, rgb: np.ndarray, pose: PoseEstimate) -> PointCloudData:
        rgb_u8 = np.ascontiguousarray(np.clip(rgb, 0, 255).astype(np.uint8))
        depth_f32 = np.ascontiguousarray(np.clip(depth_map, 0.0, self._depth_trunc).astype(np.float32))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(rgb_u8),
            o3d.geometry.Image(depth_f32),
            depth_scale=self._depth_scale,
            depth_trunc=self._depth_trunc,
            convert_rgb_to_intensity=False,
        )
        intrinsics = o3d.camera.PinholeCameraIntrinsic(
            width=int(rgb_u8.shape[1]),
            height=int(rgb_u8.shape[0]),
            fx=float(self.camera_config["fx"]),
            fy=float(self.camera_config["fy"]),
            cx=float(self.camera_config["cx"]),
            cy=float(self.camera_config["cy"]),
        )
        self._volume.integrate(rgbd, intrinsics, np.linalg.inv(pose.matrix).astype(np.float64))
        cloud = self._volume.extract_point_cloud()
        self._sync_cache_from_cloud(cloud, depth_f32, rgb_u8, pose)
        return self.data()

    def data(self) -> PointCloudData:
        return PointCloudData(points=self._points.copy(), colors=self._colors.copy())

    def to_open3d(self):
        cloud = self._volume.extract_point_cloud()
        if not len(cloud.points) and self._points.size:
            cloud.points = o3d.utility.Vector3dVector(self._points.astype(np.float64))
            cloud.colors = o3d.utility.Vector3dVector(self._colors.astype(np.float64))
        if self.mapping_config.get("tsdf_estimate_normals", False) and len(cloud.points):
            cloud.estimate_normals()
        return cloud

    def export_mesh(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        mesh = self._volume.extract_triangle_mesh()
        if not len(mesh.vertices):
            return
        mesh.compute_vertex_normals()
        o3d.io.write_triangle_mesh(str(path), mesh)

    def _sync_cache_from_cloud(self, cloud, depth_map: np.ndarray, rgb: np.ndarray, pose: PoseEstimate) -> None:
        points = np.asarray(cloud.points, dtype=np.float32)
        colors = np.asarray(cloud.colors, dtype=np.float32)
        if points.size:
            self._points = points
            self._colors = colors
            return
        stride = max(1, int(self.mapping_config.get("stride", 4)))
        fallback_points, fallback_colors = depth_to_pointcloud(depth_map, rgb, self.camera_config, stride=stride)
        self._points = transform_points(fallback_points, pose.matrix)
        self._colors = fallback_colors


class PointCloudBuilder:
    def __init__(self, camera_config: dict, mapping_config: dict) -> None:
        self.camera_config = camera_config
        self.mapping_config = mapping_config
        self.representation = str(mapping_config.get("representation", "pointcloud")).lower()
        self.backend = self._build_backend()

    @property
    def points(self) -> np.ndarray:
        return self.backend.points

    @property
    def colors(self) -> np.ndarray:
        return self.backend.colors

    def update_camera_intrinsics(self, intrinsics: dict | None) -> None:
        self.backend.update_camera_intrinsics(intrinsics)

    def integrate(self, depth_map: np.ndarray, rgb: np.ndarray, pose: PoseEstimate) -> PointCloudData:
        return self.backend.integrate(depth_map, rgb, pose)

    def data(self) -> PointCloudData:
        return self.backend.data()

    def to_open3d(self):
        return self.backend.to_open3d()

    def to_ros_pointcloud2(self, header, point_cloud2_module, point_field_type):
        return self.data().to_ros_pointcloud2(header, point_cloud2_module, point_field_type)

    def export_ply(self, path: Path) -> None:
        self.backend.export_ply(path)

    def export_mesh(self, path: Path) -> None:
        if not hasattr(self.backend, "export_mesh"):
            raise RuntimeError("Mesh export is only available for TSDF mapping.")
        self.backend.export_mesh(path)

    def _build_backend(self) -> MappingBackend:
        if self.representation == "pointcloud":
            return PointCloudFusionBackend(self.camera_config, self.mapping_config)
        if self.representation == "tsdf":
            return TsdfFusionBackend(self.camera_config, self.mapping_config)
        raise ValueError(f"Unsupported mapping representation: {self.representation}")


def _require_open3d() -> None:
    if o3d is None:
        raise RuntimeError("Open3D is required for TSDF or point cloud export. Install dependencies from requirements.txt.")
