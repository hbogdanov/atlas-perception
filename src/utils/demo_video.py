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
    ) -> None:
        frame = self.compose_frame(rgb, depth_map, trajectory, pose, metrics, runtime, self.frame_size)
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
    ) -> np.ndarray:
        width, height = frame_size
        cell_w = width // 2
        cell_h = height // 2

        canvas = np.full((height, width, 3), 245, dtype=np.uint8)
        depth_vis = colorize_depth(depth_map)
        trajectory_vis = trajectory.render_plot(size=min(cell_w, cell_h) - 40)
        status_vis = DemoVideoRecorder._build_status_panel(cell_w, cell_h, pose, metrics, runtime)

        tiles = [
            DemoVideoRecorder._fit_tile(rgb, cell_w, cell_h, "Simulator Camera Feed"),
            DemoVideoRecorder._fit_tile(depth_vis, cell_w, cell_h, "Depth Output"),
            DemoVideoRecorder._fit_tile(trajectory_vis, cell_w, cell_h, "Trajectory"),
            DemoVideoRecorder._fit_tile(status_vis, cell_w, cell_h, "Atlas Outputs"),
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
        panel = np.full((height, width, 3), 250, dtype=np.uint8)
        tx, ty, tz = pose.matrix[0, 3], pose.matrix[1, 3], pose.matrix[2, 3]
        lines = [
            f"input: {runtime['input_mode']}",
            f"slam: {runtime['slam_mode']}",
            f"frame_id: {runtime['frame_id']}",
            f"pose xyz: ({tx:.2f}, {ty:.2f}, {tz:.2f})",
            f"tracking_ok: {pose.tracking_ok}",
            f"depth_ms: {metrics['depth_ms']:.2f}",
            f"mapping_ms: {metrics['mapping_ms']:.2f}",
            f"fps: {metrics['fps']:.2f}",
            f"points: {int(metrics['points'])}",
            f"depth topic: {runtime['depth_topic']}",
            f"pose topic: {runtime['pose_topic']}",
            f"path topic: {runtime['path_topic']}",
            f"cloud topic: {runtime['pointcloud_topic']}",
        ]
        y = 42
        for line in lines:
            cv2.putText(panel, line, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (30, 30, 30), 1)
            y += 34
        return panel
