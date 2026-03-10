from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.slam.loop_closure import LoopClosureConstraint, LoopClosureDetector
from src.slam.odometry import PoseEstimate


@dataclass
class PoseGraphNode:
    index: int
    timestamp: float
    pose_matrix: np.ndarray


@dataclass
class PoseGraphEdge:
    source_index: int
    target_index: int
    transform: np.ndarray
    edge_type: str
    error: float = 0.0


@dataclass
class PoseGraph:
    nodes: list[PoseGraphNode] = field(default_factory=list)
    edges: list[PoseGraphEdge] = field(default_factory=list)
    loop_closures: list[LoopClosureConstraint] = field(default_factory=list)
    loop_closure_detector: LoopClosureDetector | None = None

    def append(self, pose: PoseEstimate) -> None:
        node = PoseGraphNode(index=len(self.nodes), timestamp=float(pose.timestamp), pose_matrix=pose.matrix.copy())
        self.nodes.append(node)
        if len(self.nodes) > 1:
            previous = self.nodes[-2]
            relative = np.linalg.inv(previous.pose_matrix) @ node.pose_matrix
            error = float(np.linalg.norm(relative[:3, 3]))
            self.edges.append(
                PoseGraphEdge(
                    source_index=previous.index,
                    target_index=node.index,
                    transform=relative.astype(np.float32),
                    edge_type="odometry",
                    error=error,
                )
            )
        if self.loop_closure_detector is None:
            return
        closure = self.loop_closure_detector.detect(
            [PoseEstimate(T_world_camera=current.pose_matrix, timestamp=current.timestamp) for current in self.nodes]
        )
        if closure is None:
            return
        target = self.nodes[closure.target_index]
        relative = np.linalg.inv(target.pose_matrix) @ node.pose_matrix
        self.loop_closures.append(closure)
        self.edges.append(
            PoseGraphEdge(
                source_index=closure.target_index,
                target_index=closure.source_index,
                transform=relative.astype(np.float32),
                edge_type="loop_closure",
                error=closure.distance,
            )
        )

    def export_json(self, path: Path) -> None:
        payload = {
            "nodes": [
                {"index": node.index, "timestamp": node.timestamp, "pose_matrix": node.pose_matrix.tolist()}
                for node in self.nodes
            ],
            "edges": [
                {
                    "source_index": edge.source_index,
                    "target_index": edge.target_index,
                    "edge_type": edge.edge_type,
                    "error": edge.error,
                    "transform": edge.transform.tolist(),
                }
                for edge in self.edges
            ],
            "loop_closures": [
                {
                    "source_index": closure.source_index,
                    "target_index": closure.target_index,
                    "distance": closure.distance,
                    "timestamp": closure.timestamp,
                }
                for closure in self.loop_closures
            ],
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def export_csv(self, path: Path) -> None:
        rows = ["edge_type,source_index,target_index,error"]
        rows.extend(f"{edge.edge_type},{edge.source_index},{edge.target_index},{edge.error}" for edge in self.edges)
        path.write_text("\n".join(rows) + "\n", encoding="utf-8")
