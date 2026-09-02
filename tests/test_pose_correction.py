import numpy as np
import pytest

from src.slam.odometry import PoseEstimate
from src.vision.landmark_pose import VisualPoseMeasurement
from src.vision.pose_correction import VisualPoseCorrector


def _measurement(x: float, timestamp: float = 1.0, covariance_scale: float = 0.01) -> VisualPoseMeasurement:
    transform = np.eye(4, dtype=np.float32)
    transform[0, 3] = x
    return VisualPoseMeasurement(
        timestamp=timestamp,
        T_world_camera=transform,
        covariance=np.eye(6, dtype=np.float32) * covariance_scale,
        reprojection_rmse=0.5,
        inlier_count=8,
        landmark_ids=("aruco:7",),
    )


def test_visual_pose_corrector_applies_a_quality_gated_measurement():
    corrector = VisualPoseCorrector({"apply_to_mapping": True, "blend_weight": 1.0})
    predicted = PoseEstimate(np.eye(4, dtype=np.float32), timestamp=1.0)

    correction = corrector.correct(predicted, _measurement(0.4), timestamp=1.0)

    assert correction.applied is True
    assert correction.reason == "accepted"
    assert correction.pose.matrix[0, 3] == pytest.approx(0.4)


def test_visual_pose_corrector_rejects_stale_or_implausible_measurements():
    corrector = VisualPoseCorrector(
        {"apply_to_mapping": True, "max_timestamp_delta_sec": 0.01, "max_translation_innovation_m": 0.5}
    )
    predicted = PoseEstimate(np.eye(4, dtype=np.float32), timestamp=1.0)

    stale = corrector.correct(predicted, _measurement(0.1, timestamp=1.1), timestamp=1.0)
    implausible = corrector.correct(predicted, _measurement(1.0), timestamp=1.0)

    assert stale.applied is False
    assert stale.reason == "timestamp_mismatch"
    assert implausible.applied is False
    assert implausible.reason == "translation_innovation"
