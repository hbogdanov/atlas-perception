from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.mapping.confidence import compute_depth_confidence, compute_multiview_confidence
from src.slam.odometry import PoseEstimate
from src.utils.geometry import depth_to_pointcloud, depth_to_pointcloud_with_confidence, transform_points

try:
    import open3d as o3d
except ImportError:  # pragma: no cover
    o3d = None


@dataclass
class PointCloudData:
    points: np.ndarray
    colors: np.ndarray
    semantic_labels: np.ndarray | None = None
    semantic_colors: np.ndarray | None = None
    class_names: dict[int, str] | None = None
    confidence: np.ndarray | None = None
    observation_counts: np.ndarray | None = None

    def to_ros_pointcloud2(self, header, point_cloud2_module, point_field_type):
        fields = [
            point_field_type(name="x", offset=0, datatype=point_field_type.FLOAT32, count=1),
            point_field_type(name="y", offset=4, datatype=point_field_type.FLOAT32, count=1),
            point_field_type(name="z", offset=8, datatype=point_field_type.FLOAT32, count=1),
            point_field_type(name="rgb", offset=12, datatype=point_field_type.UINT32, count=1),
            point_field_type(name="label", offset=16, datatype=point_field_type.UINT32, count=1),
        ]
        rows = []
        semantic_labels = (
            self.semantic_labels.astype(np.uint32)
            if self.semantic_labels is not None
            else np.full((self.points.shape[0],), np.uint32(0xFFFFFFFF), dtype=np.uint32)
        )
        point_colors = self.semantic_colors if self.semantic_colors is not None else self.colors
        for point, color, label in zip(
            self.points.astype(np.float32),
            point_colors.astype(np.float32),
            semantic_labels,
            strict=False,
        ):
            rgb_uint8 = np.clip(color * 255.0, 0, 255).astype(np.uint8)
            packed_rgb = int(rgb_uint8[0]) << 16 | int(rgb_uint8[1]) << 8 | int(rgb_uint8[2])
            rows.append([float(point[0]), float(point[1]), float(point[2]), packed_rgb, int(label)])
        return point_cloud2_module.create_cloud(header, fields, rows)

    def to_open3d(self, use_semantic_colors: bool = False):
        _require_open3d()
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(self.points.astype(np.float64))
        selected_colors = (
            self.semantic_colors if use_semantic_colors and self.semantic_colors is not None else self.colors
        )
        if selected_colors.size:
            cloud.colors = o3d.utility.Vector3dVector(selected_colors.astype(np.float64))
        return cloud


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
    def integrate(self, depth_map: np.ndarray, rgb: np.ndarray, pose: PoseEstimate, semantics=None) -> PointCloudData:
        raise NotImplementedError

    @abstractmethod
    def data(self) -> PointCloudData:
        raise NotImplementedError

    @abstractmethod
    def to_open3d(self):
        raise NotImplementedError

    def export_ply(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if o3d is None:
            _write_ascii_ply(path, self.data())
            return
        o3d.io.write_point_cloud(str(path), self.to_open3d())

    def diagnostics(self) -> dict[str, float | int]:
        return {}


@dataclass
class _VoxelState:
    weighted_position: np.ndarray
    weighted_color: np.ndarray
    total_weight: float
    observations: int
    semantic_label: int = -1
    semantic_color: np.ndarray | None = None


class PointCloudFusionBackend(MappingBackend):
    def __init__(self, camera_config: dict, mapping_config: dict) -> None:
        super().__init__(camera_config, mapping_config)
        self.voxel_size = float(mapping_config.get("voxel_size", 0.03))
        self.max_voxels = int(mapping_config.get("max_voxels", mapping_config.get("max_points", 100000)))
        self._voxels: dict[tuple[int, int, int], _VoxelState] = {}
        self._class_names: dict[int, str] = {}
        self._last_diagnostics: dict[str, float | int] = {}
        self._previous_depth: np.ndarray | None = None
        self._previous_pose: np.ndarray | None = None

    @property
    def points(self) -> np.ndarray:
        return self.data().points

    @property
    def colors(self) -> np.ndarray:
        return self.data().colors

    def integrate(self, depth_map: np.ndarray, rgb: np.ndarray, pose: PoseEstimate, semantics=None) -> PointCloudData:
        stride = int(self.mapping_config.get("stride", 4))
        semantic_fusion = bool(self.mapping_config.get("semantic_color_fusion", True))
        color_image = semantics.colorize() if semantics is not None and semantic_fusion else rgb
        confidence_config = self.mapping_config.get("confidence_fusion", {})
        multi_view_config = self.mapping_config.get("multi_view_consistency", {})
        confidence_enabled = bool(confidence_config.get("enabled", False) or multi_view_config.get("enabled", False))
        multi_view_enabled = bool(multi_view_config.get("enabled", False))
        multiview_overlap = 0
        multiview_confidence = None
        if confidence_enabled:
            confidence_map = (
                compute_depth_confidence(depth_map, float(confidence_config.get("edge_scale", 0.15)))
                if confidence_config.get("enabled", False)
                else np.where(np.asarray(depth_map) > 0.0, 1.0, 0.0).astype(np.float32)
            )
            if multi_view_enabled and self._previous_depth is not None and self._previous_pose is not None:
                multiview_confidence, multiview_overlap = compute_multiview_confidence(
                    depth_map,
                    pose.matrix,
                    self._previous_depth,
                    self._previous_pose,
                    self.camera_config,
                    float(multi_view_config.get("relative_error_scale", 0.1)),
                )
                confidence_map *= multiview_confidence
            sample_points, sample_colors, sample_confidence = depth_to_pointcloud_with_confidence(
                depth_map,
                color_image,
                self.camera_config,
                confidence_map,
                stride=stride,
                min_confidence=float(confidence_config.get("min_confidence", 0.2)),
            )
        else:
            sample_points, sample_colors = depth_to_pointcloud(
                depth_map, color_image, self.camera_config, stride=stride
            )
            sample_confidence = np.ones((sample_points.shape[0],), dtype=np.float32)
        transformed = transform_points(sample_points, pose.matrix)
        capacity_rejections = self._integrate_voxels(transformed, sample_colors, sample_confidence, semantics, stride)
        total_samples = int(depth_map[::stride, ::stride].size)
        self._last_diagnostics = {
            "confidence_enabled": int(confidence_enabled),
            "multi_view_enabled": int(multi_view_enabled),
            "multi_view_overlap_pixels": multiview_overlap,
            "multi_view_mean_confidence": (
                float(np.mean(multiview_confidence[multiview_confidence > 0.0]))
                if multiview_confidence is not None and np.any(multiview_confidence > 0.0)
                else 1.0
            ),
            "sampled_pixels": total_samples,
            "accepted_points": int(sample_points.shape[0]),
            "rejected_points": total_samples - int(sample_points.shape[0]),
            "mean_confidence": float(np.mean(sample_confidence)) if sample_confidence.size else 0.0,
            "active_voxels": len(self._voxels),
            "capacity_rejected_voxels": capacity_rejections,
        }
        self._previous_depth = np.asarray(depth_map, dtype=np.float32).copy()
        self._previous_pose = pose.matrix.copy()
        return self.data()

    def data(self) -> PointCloudData:
        if not self._voxels:
            return PointCloudData(
                points=np.empty((0, 3), dtype=np.float32),
                colors=np.empty((0, 3), dtype=np.float32),
                confidence=np.empty((0,), dtype=np.float32),
                observation_counts=np.empty((0,), dtype=np.int32),
            )
        # Dict insertion order keeps the exported point/semantic ordering stable across frames.
        states = list(self._voxels.values())
        weights = np.asarray([state.total_weight for state in states], dtype=np.float32)
        points = np.vstack([state.weighted_position / state.total_weight for state in states]).astype(np.float32)
        colors = np.vstack([state.weighted_color / state.total_weight for state in states]).astype(np.float32)
        labels = np.asarray([state.semantic_label for state in states], dtype=np.int32)
        semantic_colors = np.vstack(
            [
                state.semantic_color if state.semantic_color is not None else color
                for state, color in zip(states, colors, strict=True)
            ]
        ).astype(np.float32)
        return PointCloudData(
            points=points,
            colors=colors,
            semantic_labels=labels if np.any(labels >= 0) else None,
            semantic_colors=semantic_colors if np.any(labels >= 0) else None,
            class_names=self._class_names.copy(),
            confidence=weights
            / np.maximum(np.asarray([state.observations for state in states], dtype=np.float32), 1.0),
            observation_counts=np.asarray([state.observations for state in states], dtype=np.int32),
        )

    def diagnostics(self) -> dict[str, float | int]:
        return self._last_diagnostics.copy()

    def to_open3d(self):
        _require_open3d()
        cloud = o3d.geometry.PointCloud()
        data = self.data()
        cloud.points = o3d.utility.Vector3dVector(data.points.astype(np.float64))
        if data.colors.size:
            cloud.colors = o3d.utility.Vector3dVector(data.colors.astype(np.float64))
        return cloud

    def _integrate_voxels(
        self, points: np.ndarray, colors: np.ndarray, confidence: np.ndarray, semantics, stride: int
    ) -> int:
        if points.size == 0:
            return 0
        keys = np.floor(points / self.voxel_size).astype(np.int64)
        unique_keys, first_indices, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
        labels, semantic_colors = self._sample_semantics(semantics, stride, inverse.shape[0])
        capacity_rejections = 0
        for group_index in np.argsort(first_indices):
            key_array = unique_keys[group_index]
            members = np.flatnonzero(inverse == group_index)
            group_weight = float(np.sum(confidence[members]))
            if group_weight <= 0.0:
                continue
            key = tuple(int(value) for value in key_array)
            state = self._voxels.get(key)
            if state is None:
                if len(self._voxels) >= self.max_voxels:
                    capacity_rejections += 1
                    continue
                state = _VoxelState(
                    weighted_position=np.zeros(3, dtype=np.float64),
                    weighted_color=np.zeros(3, dtype=np.float64),
                    total_weight=0.0,
                    observations=0,
                )
                self._voxels[key] = state
            member_weights = confidence[members].astype(np.float64)
            state.weighted_position += np.sum(points[members] * member_weights[:, None], axis=0)
            state.weighted_color += np.sum(colors[members] * member_weights[:, None], axis=0)
            state.total_weight += group_weight
            state.observations += int(members.size)
            if labels is not None:
                label = int(labels[members[-1]])
                if label >= 0:
                    state.semantic_label = label
                    state.semantic_color = semantic_colors[members[-1]].astype(np.float64)
        return capacity_rejections

    def _sample_semantics(
        self, semantics, stride: int, point_count: int
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        if semantics is None:
            return None, None
        labels, colors = semantics.sample(stride=stride)
        if labels.shape[0] != point_count or colors.shape[0] != point_count:
            return None, None
        self._class_names.update(semantics.class_names)
        return labels, colors


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

    def integrate(self, depth_map: np.ndarray, rgb: np.ndarray, pose: PoseEstimate, semantics=None) -> PointCloudData:
        semantic_fusion = bool(self.mapping_config.get("semantic_color_fusion", True))
        color_image = semantics.colorize() if semantics is not None and semantic_fusion else rgb
        rgb_u8 = np.ascontiguousarray(np.clip(color_image, 0, 255).astype(np.uint8))
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

    def integrate(self, depth_map: np.ndarray, rgb: np.ndarray, pose: PoseEstimate, semantics=None) -> PointCloudData:
        return self.backend.integrate(depth_map, rgb, pose, semantics=semantics)

    def data(self) -> PointCloudData:
        return self.backend.data()

    def to_open3d(self):
        return self.backend.to_open3d()

    def to_ros_pointcloud2(self, header, point_cloud2_module, point_field_type):
        return self.data().to_ros_pointcloud2(header, point_cloud2_module, point_field_type)

    def export_ply(self, path: Path) -> None:
        self.backend.export_ply(path)

    def export_semantic_ply(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if o3d is None:
            cloud = self.data()
            _write_ascii_ply(
                path,
                PointCloudData(
                    points=cloud.points,
                    colors=cloud.semantic_colors if cloud.semantic_colors is not None else cloud.colors,
                ),
            )
            return
        cloud = self.data().to_open3d(use_semantic_colors=True)
        o3d.io.write_point_cloud(str(path), cloud)

    def export_mesh(self, path: Path) -> None:
        if not hasattr(self.backend, "export_mesh"):
            raise RuntimeError("Mesh export is only available for TSDF mapping.")
        self.backend.export_mesh(path)

    def diagnostics(self) -> dict[str, float | int]:
        return self.backend.diagnostics()

    def _build_backend(self) -> MappingBackend:
        if self.representation == "pointcloud":
            return PointCloudFusionBackend(self.camera_config, self.mapping_config)
        if self.representation == "tsdf":
            return TsdfFusionBackend(self.camera_config, self.mapping_config)
        raise ValueError(f"Unsupported mapping representation: {self.representation}")


def _require_open3d() -> None:
    if o3d is None:
        raise RuntimeError(
            "Open3D is required for TSDF or point cloud export. Install dependencies from requirements.txt."
        )


def _write_ascii_ply(path: Path, cloud: PointCloudData) -> None:
    """Export point clouds without making basic RGB-D artifacts depend on Open3D."""
    points = np.asarray(cloud.points, dtype=np.float32)
    colors = np.clip(np.asarray(cloud.colors, dtype=np.float32) * 255.0, 0, 255).astype(np.uint8)
    if points.shape[0] != colors.shape[0]:
        raise ValueError("Point-cloud export requires one color for every point.")
    header = [
        "ply",
        "format ascii 1.0",
        f"element vertex {points.shape[0]}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "end_header",
    ]
    rows = [
        f"{point[0]:.8f} {point[1]:.8f} {point[2]:.8f} {color[0]} {color[1]} {color[2]}"
        for point, color in zip(points, colors, strict=True)
    ]
    path.write_text("\n".join([*header, *rows, ""]), encoding="ascii")
