from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path

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
