from pathlib import Path

import numpy as np
import pytest

from src.slam.metrics import (
    TrajectoryPose,
    associate_trajectories,
    compute_trajectory_metrics,
    load_atlas_trajectory_json,
    load_tum_groundtruth,
)


def make_pose(timestamp: float, tx: float, yaw_deg: float = 0.0) -> TrajectoryPose:
    angle = np.deg2rad(yaw_deg)
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, :3] = rotation
    matrix[0, 3] = tx
    return TrajectoryPose(timestamp=timestamp, matrix=matrix)


def test_associate_trajectories_matches_nearest_timestamps():
    estimated = [make_pose(0.0, 0.0), make_pose(1.01, 1.0)]
    groundtruth = [make_pose(0.01, 0.0), make_pose(1.0, 1.0)]
    matches = associate_trajectories(estimated, groundtruth, max_timestamp_diff=0.02)
    assert len(matches) == 2


def test_compute_trajectory_metrics_zero_for_aligned_motion_with_global_offset():
    estimated = [make_pose(0.0, 1.0), make_pose(1.0, 2.0), make_pose(2.0, 3.0)]
    groundtruth = [make_pose(0.0, 0.0), make_pose(1.0, 1.0), make_pose(2.0, 2.0)]
    metrics = compute_trajectory_metrics(estimated, groundtruth)
    assert metrics.ate_rmse == pytest.approx(0.0, abs=1e-6)
    assert metrics.rpe_trans_rmse == pytest.approx(0.0, abs=1e-6)
    assert metrics.rpe_rot_deg_rmse == pytest.approx(0.0, abs=1e-6)


def test_compute_trajectory_metrics_captures_relative_rotation_error():
    estimated = [make_pose(0.0, 0.0, 0.0), make_pose(1.0, 1.0, 20.0), make_pose(2.0, 2.0, 40.0)]
    groundtruth = [make_pose(0.0, 0.0, 0.0), make_pose(1.0, 1.0, 10.0), make_pose(2.0, 2.0, 20.0)]
    metrics = compute_trajectory_metrics(estimated, groundtruth)
    assert metrics.rpe_rot_deg_rmse > 0.0


def test_load_atlas_trajectory_json_filters_tracking_failures(tmp_path: Path):
    path = tmp_path / "trajectory.json"
    path.write_text(
        '[{"timestamp": 0.0, "tracking_ok": true, "T_world_camera": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]},'
        ' {"timestamp": 1.0, "tracking_ok": false, "T_world_camera": [[1,0,0,1],[0,1,0,0],[0,0,1,0],[0,0,0,1]]}]',
        encoding="utf-8",
    )
    poses = load_atlas_trajectory_json(path)
    assert len(poses) == 1
    assert poses[0].timestamp == pytest.approx(0.0)


def test_load_tum_groundtruth_parses_pose_file(tmp_path: Path):
    path = tmp_path / "groundtruth.txt"
    path.write_text("0.0 1.0 2.0 3.0 0.0 0.0 0.0 1.0\n", encoding="utf-8")
    poses = load_tum_groundtruth(path)
    assert len(poses) == 1
    assert np.allclose(poses[0].matrix[:3, 3], np.array([1.0, 2.0, 3.0], dtype=np.float32))
