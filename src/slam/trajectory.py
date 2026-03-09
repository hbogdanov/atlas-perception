from __future__ import annotations

from dataclasses import dataclass, field
import json
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
        margin = 40.0
        scale = min((size - 2 * margin) / spans[0], (size - 2 * margin) / spans[1])

        def project(point: np.ndarray) -> tuple[int, int]:
            x = int((point[0] - mins[0]) * scale + margin)
            y = int(size - ((point[1] - mins[1]) * scale + margin))
            return x, y

        projected = [project(point) for point in points]
        for start, end in zip(projected[:-1], projected[1:], strict=False):
            cv2.line(canvas, start, end, (40, 120, 220), 2)
        cv2.circle(canvas, projected[0], 6, (0, 180, 0), -1)
        cv2.circle(canvas, projected[-1], 6, (0, 0, 220), -1)
        cv2.putText(canvas, "start", (projected[0][0] + 8, projected[0][1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 120, 0), 1)
        cv2.putText(canvas, "end", (projected[-1][0] + 8, projected[-1][1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 160), 1)
        cv2.putText(canvas, "XY trajectory", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2)
        return canvas

    def export_plot(self, path: Path, size: int = 800) -> None:
        canvas = self.render_plot(size=size)
        cv2.imwrite(str(path), canvas)
