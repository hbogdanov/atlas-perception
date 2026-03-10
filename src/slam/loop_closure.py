from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.slam.odometry import PoseEstimate


@dataclass
class LoopClosureConstraint:
    source_index: int
    target_index: int
    distance: float
    timestamp: float


class LoopClosureDetector:
    def __init__(self, config: dict | None = None) -> None:
        config = config or {}
        self.enabled = bool(config.get("enabled", True))
        self.min_node_gap = int(config.get("min_node_gap", 15))
        self.distance_threshold = float(config.get("distance_threshold", 0.15))

    def detect(self, poses: list[PoseEstimate]) -> LoopClosureConstraint | None:
        if not self.enabled or len(poses) <= self.min_node_gap:
            return None
        current_index = len(poses) - 1
        current_pose = poses[current_index]
        current_position = current_pose.matrix[:3, 3]
        best_constraint: LoopClosureConstraint | None = None
        for target_index, target_pose in enumerate(poses[: current_index - self.min_node_gap + 1]):
            distance = float(np.linalg.norm(current_position - target_pose.matrix[:3, 3]))
            if distance > self.distance_threshold:
                continue
            if best_constraint is None or distance < best_constraint.distance:
                best_constraint = LoopClosureConstraint(
                    source_index=current_index,
                    target_index=target_index,
                    distance=distance,
                    timestamp=float(current_pose.timestamp),
                )
        return best_constraint
