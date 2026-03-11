from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from src.slam.odometry import PoseEstimate


@dataclass
class Trajectory:
    poses: list[PoseEstimate] = field(default_factory=list)

    def append(self, pose: PoseEstimate) -> None:
        self.poses.append(pose)

    def export(self, path: Path) -> None:
        stacked = np.stack([pose.matrix for pose in self.poses]) if self.poses else np.empty((0, 4, 4))
        np.save(path, stacked)

    def export_json(self, path: Path) -> None:
        payload = [
            {
                "timestamp": pose.timestamp,
                "tracking_ok": pose.tracking_ok,
                "T_world_camera": pose.matrix.tolist(),
            }
            for pose in self.poses
        ]
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def export_csv(self, path: Path) -> None:
        header = "timestamp,tracking_ok,tx,ty,tz\n"
        rows = [
            f"{pose.timestamp},{int(pose.tracking_ok)},{pose.matrix[0,3]},{pose.matrix[1,3]},{pose.matrix[2,3]}"
            for pose in self.poses
        ]
        path.write_text(header + "\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")

    def render_plot(self, size: int = 800) -> np.ndarray:
        canvas = np.full((size, size, 3), 255, dtype=np.uint8)
        if not self.poses:
            cv2.putText(canvas, "No trajectory", (size // 3, size // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
            return canvas

        points = np.array([[pose.matrix[0, 3], pose.matrix[1, 3]] for pose in self.poses], dtype=np.float32)
        mins = points.min(axis=0)
        maxs = points.max(axis=0)
        spans = np.maximum(maxs - mins, 1e-6)
        margin = 78.0
        plot_size = size - 2 * margin
        scale = min(plot_size / spans[0], plot_size / spans[1])
        x_offset = (plot_size - spans[0] * scale) * 0.5
        y_offset = (plot_size - spans[1] * scale) * 0.5

        def project(point: np.ndarray) -> tuple[int, int]:
            x = int((point[0] - mins[0]) * scale + margin + x_offset)
            y = int(size - ((point[1] - mins[1]) * scale + margin + y_offset))
            return x, y

        plot_left = int(margin)
        plot_top = int(margin)
        plot_right = int(size - margin)
        plot_bottom = int(size - margin)
        cv2.rectangle(canvas, (plot_left, plot_top), (plot_right, plot_bottom), (220, 220, 220), 2)
        for fraction in (0.25, 0.5, 0.75):
            x = int(plot_left + (plot_right - plot_left) * fraction)
            y = int(plot_top + (plot_bottom - plot_top) * fraction)
            cv2.line(canvas, (x, plot_top), (x, plot_bottom), (238, 238, 238), 1)
            cv2.line(canvas, (plot_left, y), (plot_right, y), (238, 238, 238), 1)

        projected = [project(point) for point in points]
        for start, end in zip(projected[:-1], projected[1:], strict=False):
            cv2.line(canvas, start, end, (40, 120, 220), 2)
        cv2.circle(canvas, projected[0], 6, (0, 180, 0), -1)
        cv2.circle(canvas, projected[-1], 6, (0, 0, 220), -1)
        cv2.putText(
            canvas, "start", (projected[0][0] + 8, projected[0][1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 120, 0), 1
        )
        cv2.putText(
            canvas, "end", (projected[-1][0] + 8, projected[-1][1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 160), 1
        )
        current_pose = self.poses[-1].matrix
        cv2.putText(canvas, "Camera Pose In World XY", (20, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2)
        cv2.putText(
            canvas,
            f"X forward: {current_pose[0, 3]:.2f} m",
            (20, size - 44),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (35, 35, 35),
            1,
        )
        cv2.putText(
            canvas,
            f"Y lateral: {current_pose[1, 3]:.2f} m",
            (20, size - 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (35, 35, 35),
            1,
        )
        cv2.putText(canvas, "Y (m)", (12, plot_top - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (55, 55, 55), 1)
        cv2.putText(
            canvas,
            "X (m)",
            (plot_right - 52, size - 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (55, 55, 55),
            1,
        )
        return canvas

    def export_plot(self, path: Path, size: int = 800) -> None:
        canvas = self.render_plot(size=size)
        cv2.imwrite(str(path), canvas)
