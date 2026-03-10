import numpy as np
from pathlib import Path
import pytest

from src.slam.wrapper import RtabmapBackend, SlamWrapper


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


def test_rtabmap_pose_callback_stores_full_rotation_and_translation():
    backend = RtabmapBackend({"mode": "rtabmap", "pose_topic": "/rtabmap/localization_pose"})

    pose_msg = type(
        "PoseStamped",
        (),
        {
            "header": type(
                "Header",
                (),
                {"stamp": type("Stamp", (), {"sec": 1, "nanosec": 500_000_000})()},
            )(),
            "pose": type(
                "Pose",
                (),
                {
                    "position": type("Position", (), {"x": 1.0, "y": 2.0, "z": 3.0})(),
                    "orientation": type(
                        "Orientation",
                        (),
                        {"x": 0.0, "y": 0.0, "z": np.sqrt(0.5), "w": np.sqrt(0.5)},
                    )(),
                },
            )(),
        },
    )()

    backend._pose_callback(pose_msg)

    pose = backend.get_pose()
    assert pose is not None
    assert pose.timestamp == pytest.approx(1.5)
    assert pose.matrix[0, 3] == pytest.approx(1.0)
    assert pose.matrix[1, 3] == pytest.approx(2.0)
    assert pose.matrix[2, 3] == pytest.approx(3.0)
    assert np.allclose(
        pose.matrix[:3, :3],
        np.array(
            [
                [0.0, -1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        ),
        atol=1e-6,
    )


def test_trajectory_export_writes_plot(tmp_path: Path):
    slam = SlamWrapper({"mode": "dummy"})
    slam.update(None, None, 1.0)
    slam.update(None, None, 2.0)
    out = tmp_path / "trajectory.npy"
    slam.export_trajectory(out)
    assert out.exists()
    assert out.with_suffix(".json").exists()
    assert out.with_suffix(".csv").exists()
    assert (tmp_path / "trajectory_plot.png").exists()
    assert (tmp_path / "pose_graph.json").exists()
    assert (tmp_path / "pose_graph_edges.csv").exists()


def test_pose_graph_tracks_odometry_edges():
    slam = SlamWrapper({"mode": "dummy", "pose_graph": {"enabled": True, "loop_closure": {"enabled": False}}})
    slam.update(None, None, 1.0)
    slam.update(None, None, 2.0)
    slam.update(None, None, 3.0)
    assert len(slam.pose_graph.nodes) == 3
    assert len(slam.pose_graph.edges) == 2
    assert all(edge.edge_type == "odometry" for edge in slam.pose_graph.edges)


def test_pose_graph_adds_simple_loop_closure():
    slam = SlamWrapper(
        {
            "mode": "disabled",
            "pose_graph": {
                "enabled": True,
                "loop_closure": {"enabled": True, "min_node_gap": 2, "distance_threshold": 0.01},
            },
        }
    )
    slam.update(None, None, 1.0)
    slam.update(None, None, 2.0)
    slam.update(None, None, 3.0)
    assert len(slam.pose_graph.loop_closures) == 1
    assert slam.pose_graph.edges[-1].edge_type == "loop_closure"
