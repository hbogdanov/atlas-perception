import numpy as np

from src.mapping.confidence import compute_depth_confidence, compute_multiview_confidence
from src.mapping.pointcloud import PointCloudBuilder


def test_depth_confidence_downweights_discontinuities_and_invalid_depth():
    depth = np.array([[1.0, 1.0, 4.0], [1.0, 0.0, 4.0]], dtype=np.float32)
    confidence = compute_depth_confidence(depth)

    assert confidence[1, 1] == 0.0
    assert confidence[0, 1] < confidence[0, 0]


def test_confidence_fusion_rejects_low_confidence_samples():
    builder = PointCloudBuilder(
        {"fx": 1.0, "fy": 1.0, "cx": 0.0, "cy": 0.0},
        {
            "stride": 1,
            "max_points": 100,
            "confidence_fusion": {"enabled": True, "edge_scale": 0.1, "min_confidence": 0.8},
        },
    )
    depth = np.array([[1.0, 1.0, 5.0]], dtype=np.float32)
    rgb = np.zeros((1, 3, 3), dtype=np.uint8)
    pose = type("Pose", (), {"matrix": np.eye(4, dtype=np.float32)})()

    cloud = builder.integrate(depth, rgb, pose)

    assert cloud.confidence is not None
    assert builder.diagnostics()["rejected_points"] > 0


def test_multiview_confidence_penalizes_disagreeing_covisible_depth():
    intrinsics = {"fx": 1.0, "fy": 1.0, "cx": 0.0, "cy": 0.0}
    pose = np.eye(4, dtype=np.float32)
    previous = np.ones((3, 3), dtype=np.float32)
    current = previous.copy()
    current[1, 1] = 2.0

    confidence, overlap = compute_multiview_confidence(current, pose, previous, pose, intrinsics)

    assert overlap == 9
    assert confidence[0, 0] == 1.0
    assert confidence[1, 1] < 0.01
