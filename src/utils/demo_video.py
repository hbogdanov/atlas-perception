from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.depth.visualize import colorize_depth
from src.slam.odometry import PoseEstimate
from src.slam.trajectory import Trajectory


class DemoVideoRecorder:
    def __init__(self, path: Path, fps: float = 15.0, width: int = 1280, height: int = 720) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.frame_size = (int(width), int(height))
        self.writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(max(fps, 1.0)),
            self.frame_size,
        )
        if not self.writer.isOpened():
            raise RuntimeError(f"Failed to open demo video writer at {path}.")

    def write(
        self,
        rgb: np.ndarray,
        depth_map: np.ndarray,
        trajectory: Trajectory,
        pose: PoseEstimate,
        metrics: dict[str, float | int],
        runtime: dict[str, str],
        semantic_image: np.ndarray | None = None,
        map_image: np.ndarray | None = None,
    ) -> None:
        frame = self.compose_frame(
            rgb,
            depth_map,
            trajectory,
            pose,
            metrics,
            runtime,
            self.frame_size,
            semantic_image=semantic_image,
            map_image=map_image,
        )
        self.writer.write(frame)

    def close(self) -> None:
        self.writer.release()

    @staticmethod
    def compose_frame(
        rgb: np.ndarray,
        depth_map: np.ndarray,
        trajectory: Trajectory,
        pose: PoseEstimate,
        metrics: dict[str, float | int],
        runtime: dict[str, str],
        frame_size: tuple[int, int] = (1280, 720),
        semantic_image: np.ndarray | None = None,
        map_image: np.ndarray | None = None,
    ) -> np.ndarray:
        width, height = frame_size
        cell_w = width // 2
        cell_h = height // 2

        canvas = np.full((height, width, 3), 245, dtype=np.uint8)
        depth_vis = colorize_depth(
            depth_map,
            min_depth=_optional_float(runtime.get("depth_viz_min")),
            max_depth=_optional_float(runtime.get("depth_viz_max")),
            invert=bool(runtime.get("depth_viz_invert", True)),
            fill_invalid=bool(runtime.get("depth_viz_fill_invalid", False)),
            smooth_ksize=int(runtime.get("depth_viz_smooth_ksize", 0)),
        )
        semantic_vis = (
            semantic_image
            if semantic_image is not None
            else DemoVideoRecorder._build_semantic_panel(
                runtime.get("semantic_mode", "disabled"),
                runtime.get("semantic_summary", "disabled"),
            )
        )
        map_vis = (
            map_image
            if map_image is not None
            else DemoVideoRecorder._build_map_panel(cell_w, cell_h, pose, metrics, runtime)
        )
        rgb_title = runtime.get("rgb_title", "Simulator Camera Feed")
        depth_title = runtime.get("depth_title", "Depth Output")
        semantic_title = runtime.get("semantic_title", "Semantic Overlay")
        map_title = runtime.get("map_title", "Fused Point Cloud Map")

        tiles = [
            DemoVideoRecorder._fit_tile(rgb, cell_w, cell_h, rgb_title),
            DemoVideoRecorder._fit_tile(depth_vis, cell_w, cell_h, depth_title),
            DemoVideoRecorder._fit_tile(semantic_vis, cell_w, cell_h, semantic_title),
            DemoVideoRecorder._fit_tile(map_vis, cell_w, cell_h, map_title),
        ]

        positions = [(0, 0), (cell_w, 0), (0, cell_h), (cell_w, cell_h)]
        for tile, (x, y) in zip(tiles, positions, strict=False):
            canvas[y : y + cell_h, x : x + cell_w] = tile
        return canvas

    @staticmethod
    def _fit_tile(image: np.ndarray, width: int, height: int, title: str) -> np.ndarray:
        tile = np.full((height, width, 3), 255, dtype=np.uint8)
        content_h = height - 44
        scale = min(width / image.shape[1], content_h / image.shape[0])
        resized = cv2.resize(image, (max(1, int(image.shape[1] * scale)), max(1, int(image.shape[0] * scale))))
        y = 38 + max(0, (content_h - resized.shape[0]) // 2)
        x = max(0, (width - resized.shape[1]) // 2)
        tile[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
        cv2.putText(tile, title, (18, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (25, 25, 25), 2)
        cv2.rectangle(tile, (0, 0), (width - 1, height - 1), (210, 210, 210), 2)
        return tile

    @staticmethod
    def _build_status_panel(
        width: int,
        height: int,
        pose: PoseEstimate,
        metrics: dict[str, float | int],
        runtime: dict[str, str],
    ) -> np.ndarray:
        return DemoVideoRecorder._build_map_panel(width, height, pose, metrics, runtime)

    @staticmethod
    def _build_semantic_panel(mode: str = "disabled", summary: str = "disabled") -> np.ndarray:
        panel = np.full((320, 520, 3), 248, dtype=np.uint8)
        if mode == "empty":
            lines = [
                ("No recognized objects", 0.9, (70, 70, 70), 2),
                ("YOLOv8 ran, but this frame produced", 0.66, (95, 95, 95), 1),
                ("no COCO segmentation masks.", 0.66, (95, 95, 95), 1),
                (summary, 0.62, (110, 110, 110), 1),
            ]
        else:
            lines = [
                ("Semantic segmentation disabled", 0.9, (70, 70, 70), 2),
                ("Enable semantics.backend to render", 0.68, (95, 95, 95), 1),
                ("live semantic overlays here.", 0.68, (95, 95, 95), 1),
            ]
        y = 118
        for text, scale, color, thickness in lines:
            cv2.putText(panel, text, (38, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)
            y += 48
        cv2.rectangle(panel, (24, 24), (panel.shape[1] - 24, panel.shape[0] - 24), (210, 210, 210), 2)
        return panel

    @staticmethod
    def _build_map_panel(
        width: int,
        height: int,
        pose: PoseEstimate,
        metrics: dict[str, float | int],
        runtime: dict[str, str],
    ) -> np.ndarray:
        panel = np.full((height, width, 3), 250, dtype=np.uint8)
        tx, ty, tz = pose.matrix[0, 3], pose.matrix[1, 3], pose.matrix[2, 3]
        lines = [
            f"slam mode: {runtime['slam_mode']}",
            f"pose xyz: ({tx:.2f}, {ty:.2f}, {tz:.2f})",
            f"tracking_ok: {pose.tracking_ok}",
            f"fused points: {int(metrics['points'])}",
            f"fps: {metrics['fps']:.2f}",
            f"mapping_ms: {metrics['mapping_ms']:.2f}",
        ]
        y = 42
        for line in lines:
            cv2.putText(panel, line, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (30, 30, 30), 1)
            y += 34
        return panel

    @staticmethod
    def render_topdown_map(
        point_cloud,
        pose: PoseEstimate,
        metrics: dict[str, float | int],
        runtime: dict[str, str],
        size: tuple[int, int] = (520, 320),
    ) -> np.ndarray:
        width, height = size
        panel = np.full((height, width, 3), 250, dtype=np.uint8)
        cv2.rectangle(panel, (16, 16), (width - 16, height - 16), (220, 220, 220), 2)
        points = getattr(point_cloud, "points", None)
        colors = getattr(point_cloud, "semantic_colors", None)
        if colors is None or not getattr(colors, "size", 0):
            colors = getattr(point_cloud, "colors", None)
        if points is not None and getattr(points, "size", 0):
            points = np.asarray(points, dtype=np.float32)
            colors = np.asarray(colors, dtype=np.float32) if colors is not None and getattr(colors, "size", 0) else None
            projection = str(runtime.get("map_projection", "auto")).lower()
            if projection == "xy":
                axes = points[:, [0, 1]]
                pose_marker = np.array([pose.matrix[0, 3], pose.matrix[1, 3]], dtype=np.float32)
                heading_2d = np.array([pose.matrix[0, 0], pose.matrix[1, 0]], dtype=np.float32)
                height_values = points[:, 2] if points.shape[1] >= 3 else axes[:, 1]
            elif projection == "xz":
                axes = points[:, [0, 2]]
                pose_marker = np.array([pose.matrix[0, 3], pose.matrix[2, 3]], dtype=np.float32)
                heading_2d = np.array([pose.matrix[0, 2], pose.matrix[2, 2]], dtype=np.float32)
                height_values = points[:, 1] if points.shape[1] >= 3 else axes[:, 1]
            else:
                axes = points[:, [0, 2]]
                spans = axes.max(axis=0) - axes.min(axis=0)
                if float(spans[1]) < 1e-4:
                    axes = points[:, [0, 1]]
                    pose_marker = np.array([pose.matrix[0, 3], pose.matrix[1, 3]], dtype=np.float32)
                    heading_2d = np.array([pose.matrix[0, 0], pose.matrix[1, 0]], dtype=np.float32)
                    height_values = points[:, 2] if points.shape[1] >= 3 else axes[:, 1]
                else:
                    pose_marker = np.array([pose.matrix[0, 3], pose.matrix[2, 3]], dtype=np.float32)
                    heading_2d = np.array([pose.matrix[0, 2], pose.matrix[2, 2]], dtype=np.float32)
                    height_values = points[:, 1] if points.shape[1] >= 3 else axes[:, 1]
            configured_bounds = runtime.get("map_bounds")
            if isinstance(configured_bounds, (list, tuple)) and len(configured_bounds) == 4:
                mins = np.array([float(configured_bounds[0]), float(configured_bounds[2])], dtype=np.float32)
                maxs = np.array([float(configured_bounds[1]), float(configured_bounds[3])], dtype=np.float32)
            else:
                mins = np.percentile(axes, 1.0, axis=0).astype(np.float32)
                maxs = np.percentile(axes, 99.0, axis=0).astype(np.float32)
                mins = np.minimum(mins, pose_marker)
                maxs = np.maximum(maxs, pose_marker)
            spans = np.maximum(maxs - mins, 1e-6)
            mins = mins - spans * 0.05
            maxs = maxs + spans * 0.05
            spans = np.maximum(maxs - mins, 1e-6)
            margin = 28.0
            plot_w = width - 2 * margin
            plot_h = height - 110.0
            scale = min(plot_w / spans[0], plot_h / spans[1])
            x_offset = (plot_w - spans[0] * scale) * 0.5
            y_offset = (plot_h - spans[1] * scale) * 0.5
            projected_x = ((axes[:, 0] - mins[0]) * scale + margin + x_offset).astype(np.int32)
            projected_y = (height - 58.0 - ((axes[:, 1] - mins[1]) * scale + y_offset)).astype(np.int32)
            valid = (projected_x >= 24) & (projected_x < width - 24) & (projected_y >= 48) & (projected_y < height - 28)
            projected_x = projected_x[valid]
            projected_y = projected_y[valid]
            height_values = height_values[valid]
            plot_layer = np.zeros_like(panel)
            plot_mask = np.zeros((height, width), dtype=np.uint8)
            if height_values.size:
                normalized_height = (height_values - float(height_values.min())) / max(
                    float(height_values.max() - height_values.min()), 1e-6
                )
                height_colors = cv2.applyColorMap(
                    np.clip(normalized_height * 255.0, 0, 255).astype(np.uint8),
                    cv2.COLORMAP_VIRIDIS,
                )
                plot_layer[projected_y, projected_x] = height_colors[:, 0, :]
            elif colors is not None and colors.size:
                rgb = np.clip(colors[valid] * 255.0, 0, 255).astype(np.uint8)
                plot_layer[projected_y, projected_x] = rgb
            else:
                plot_layer[projected_y, projected_x] = (55, 110, 220)
            plot_mask[projected_y, projected_x] = 255
            kernel = np.ones((3, 3), dtype=np.uint8)
            plot_mask = cv2.dilate(plot_mask, kernel, iterations=1)
            plot_layer = cv2.dilate(plot_layer, kernel, iterations=1)
            panel[plot_mask > 0] = plot_layer[plot_mask > 0]
            pose_x = int((pose_marker[0] - mins[0]) * scale + margin + x_offset)
            pose_y = int(height - 58.0 - ((pose_marker[1] - mins[1]) * scale + y_offset))
            if 24 <= pose_x < width - 24 and 48 <= pose_y < height - 28:
                cv2.circle(panel, (pose_x, pose_y), 6, (0, 0, 220), -1)
                norm = float(np.linalg.norm(heading_2d))
                if norm > 1e-6:
                    heading_2d = heading_2d / norm
                axis_length = 18
                end = (
                    int(pose_x + heading_2d[0] * axis_length),
                    int(pose_y - heading_2d[1] * axis_length),
                )
                cv2.arrowedLine(panel, (pose_x, pose_y), end, (0, 0, 220), 2, tipLength=0.35)
        else:
            cv2.putText(panel, "No fused points yet", (40, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (70, 70, 70), 2)

        summary = [
            f"{runtime['slam_mode']}",
            f"{int(metrics['points'])} pts",
            f"{int(metrics.get('frames', 0))} frames",
            f"{metrics['fps']:.1f} FPS",
        ]
        x = 24
        for chunk in summary:
            cv2.putText(panel, chunk, (x, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (45, 45, 45), 2)
            x += 120
        return panel


def _optional_float(value) -> float | None:
    if value is None:
        return None
    return float(value)
