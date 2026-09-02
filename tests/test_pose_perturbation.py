import numpy as np

from src.evaluation.pose_perturbation import PosePerturber
from src.slam.odometry import PoseEstimate


def make_pose(x: float, timestamp: float) -> PoseEstimate:
    transform = np.eye(4, dtype=np.float32)
    transform[0, 3] = x
    return PoseEstimate(transform, timestamp)


def test_pose_perturber_applies_configured_latency_without_noise():
    perturber = PosePerturber({"enabled": True, "latency_frames": 1, "seed": 3})

    first = perturber.perturb(make_pose(1.0, 1.0))
    second = perturber.perturb(make_pose(2.0, 2.0))

    assert first is not None and first.matrix[0, 3] == 1.0
    assert second is not None and second.matrix[0, 3] == 1.0
    assert second.timestamp == 2.0


def test_pose_perturber_can_drop_all_mapping_poses():
    perturber = PosePerturber({"enabled": True, "dropout_probability": 1.0, "seed": 3})

    assert perturber.perturb(make_pose(1.0, 1.0)) is None
