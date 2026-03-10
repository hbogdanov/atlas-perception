from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.ros2.transforms import quaternion_to_rotation_matrix


@dataclass(frozen=True)
class TrajectoryPose:
    timestamp: float
    matrix: np.ndarray


@dataclass(frozen=True)
class TrajectoryMetrics:
    ate_rmse: float
    rpe_trans_rmse: float
    rpe_rot_deg_rmse: float
    matched_poses: int
    matched_pairs: int

    def to_dict(self) -> dict[str, float | int]:
        return {
            "ate_rmse": self.ate_rmse,
            "rpe_trans_rmse": self.rpe_trans_rmse,
            "rpe_rot_deg_rmse": self.rpe_rot_deg_rmse,
            "matched_poses": self.matched_poses,
            "matched_pairs": self.matched_pairs,
        }


def load_atlas_trajectory_json(path: str | Path) -> list[TrajectoryPose]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    poses = [
        TrajectoryPose(
            timestamp=float(entry["timestamp"]), matrix=np.asarray(entry["T_world_camera"], dtype=np.float32)
        )
        for entry in payload
        if bool(entry.get("tracking_ok", True))
    ]
    return sorted(poses, key=lambda pose: pose.timestamp)


def load_tum_groundtruth(path: str | Path) -> list[TrajectoryPose]:
    poses: list[TrajectoryPose] = []
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 8:
            raise ValueError(f"Expected TUM trajectory line with 8 columns, got {len(parts)}: {line!r}")
        timestamp, tx, ty, tz, qx, qy, qz, qw = map(float, parts)
        matrix = np.eye(4, dtype=np.float32)
        matrix[:3, :3] = quaternion_to_rotation_matrix(np.array([qx, qy, qz, qw], dtype=np.float32))
        matrix[:3, 3] = np.array([tx, ty, tz], dtype=np.float32)
        poses.append(TrajectoryPose(timestamp=timestamp, matrix=matrix))
    return poses


def associate_trajectories(
    estimated: list[TrajectoryPose],
    groundtruth: list[TrajectoryPose],
    max_timestamp_diff: float = 0.02,
) -> list[tuple[TrajectoryPose, TrajectoryPose]]:
    gt_timestamps = np.array([pose.timestamp for pose in groundtruth], dtype=np.float64)
    used_gt: set[int] = set()
    matches: list[tuple[TrajectoryPose, TrajectoryPose]] = []
    for est_pose in estimated:
        if gt_timestamps.size == 0:
            break
        index = int(np.argmin(np.abs(gt_timestamps - est_pose.timestamp)))
        if index in used_gt:
            continue
        if abs(gt_timestamps[index] - est_pose.timestamp) > max_timestamp_diff:
            continue
        used_gt.add(index)
        matches.append((est_pose, groundtruth[index]))
    matches.sort(key=lambda pair: pair[0].timestamp)
    return matches


def compute_trajectory_metrics(
    estimated: list[TrajectoryPose],
    groundtruth: list[TrajectoryPose],
    max_timestamp_diff: float = 0.02,
) -> TrajectoryMetrics:
    matches = associate_trajectories(estimated, groundtruth, max_timestamp_diff=max_timestamp_diff)
    if len(matches) < 2:
        raise ValueError("Need at least two matched poses to compute ATE and RPE.")

    est_points = np.stack([est.matrix[:3, 3] for est, _ in matches], axis=0)
    gt_points = np.stack([gt.matrix[:3, 3] for _, gt in matches], axis=0)
    aligned_est_points = _apply_rigid_alignment(est_points, gt_points)
    ate_rmse = float(np.sqrt(np.mean(np.sum((aligned_est_points - gt_points) ** 2, axis=1))))

    rpe_trans_errors: list[float] = []
    rpe_rot_errors_deg: list[float] = []
    for (est_a, gt_a), (est_b, gt_b) in zip(matches[:-1], matches[1:], strict=False):
        rel_est = np.linalg.inv(est_a.matrix) @ est_b.matrix
        rel_gt = np.linalg.inv(gt_a.matrix) @ gt_b.matrix
        delta = np.linalg.inv(rel_gt) @ rel_est
        rpe_trans_errors.append(float(np.linalg.norm(delta[:3, 3])))
        rpe_rot_errors_deg.append(float(np.degrees(_rotation_angle(delta[:3, :3]))))

    return TrajectoryMetrics(
        ate_rmse=ate_rmse,
        rpe_trans_rmse=float(np.sqrt(np.mean(np.square(rpe_trans_errors)))),
        rpe_rot_deg_rmse=float(np.sqrt(np.mean(np.square(rpe_rot_errors_deg)))),
        matched_poses=len(matches),
        matched_pairs=len(rpe_trans_errors),
    )


def _apply_rigid_alignment(points: np.ndarray, target: np.ndarray) -> np.ndarray:
    rotation, translation = _umeyama_rigid_transform(points, target)
    return (rotation @ points.T).T + translation


def _umeyama_rigid_transform(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    src_mean = np.mean(source, axis=0)
    tgt_mean = np.mean(target, axis=0)
    src_centered = source - src_mean
    tgt_centered = target - tgt_mean
    covariance = src_centered.T @ tgt_centered / max(source.shape[0], 1)
    u, _, vt = np.linalg.svd(covariance)
    correction = np.eye(3, dtype=np.float32)
    if np.linalg.det(u @ vt) < 0:
        correction[2, 2] = -1.0
    rotation = vt.T @ correction @ u.T
    translation = tgt_mean - rotation @ src_mean
    return rotation.astype(np.float32), translation.astype(np.float32)


def _rotation_angle(rotation: np.ndarray) -> float:
    trace = np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.arccos(trace))
