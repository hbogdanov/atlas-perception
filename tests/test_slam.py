import pytest

from src.slam.wrapper import SlamWrapper


def test_disabled_mode_returns_identity_pose():
    slam = SlamWrapper({"mode": "disabled"})
    pose_a = slam.update(None, None, 1.0)
    pose_b = slam.update(None, None, 2.0)
    assert pose_a.matrix[0, 3] == 0.0
    assert pose_b.matrix[0, 3] == 0.0


def test_dummy_mode_generates_synthetic_motion():
    slam = SlamWrapper({"mode": "dummy"})
    pose_a = slam.update(None, None, 1.0)
    pose_b = slam.update(None, None, 2.0)
    assert pose_a.matrix[0, 3] == 0.0
    assert pose_b.matrix[0, 3] == pytest.approx(0.05)


def test_unknown_backend_mode_fails_explicitly():
    with pytest.raises(ValueError):
        SlamWrapper({"mode": "visual_odometry"})


def test_rtabmap_mode_without_ros_runtime_fails_cleanly():
    slam = SlamWrapper({"mode": "rtabmap"})
    with pytest.raises(RuntimeError):
        slam.update(None, None, 1.0)
