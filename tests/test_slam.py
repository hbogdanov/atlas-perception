import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.slam.loop_closure import AppearanceLoopClosureDetector, LoopClosureConstraint
from src.slam.odometry import PoseEstimate
from src.slam.pose_graph import PoseGraph
from src.slam.trajectory import _select_projection_axes
from src.slam.wrapper import RtabmapBackend, SlamWrapper
from src.vision.landmark_pose import VisualPoseMeasurement


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
    pose_c = slam.update(None, None, 3.0)
    assert pose_a.matrix[0, 3] == 0.0
    assert pose_b.matrix[0, 3] == pytest.approx(1.4 * (1.0 - np.cos(0.035)))
    assert pose_b.matrix[1, 3] == pytest.approx(1.4 * np.sin(0.035))
    assert pose_c.matrix[1, 3] > pose_b.matrix[1, 3]
    assert pose_b.matrix[2, 3] == pytest.approx(np.sin(0.05) * 0.02)
    assert pose_b.matrix[0, 0] < 1.0
    assert pose_b.matrix[1, 0] > 0.0


def test_dummy_mode_accepts_custom_motion_profile():
    slam = SlamWrapper(
        {
            "mode": "dummy",
            "path_radius_x": 2.0,
            "path_radius_y": 0.5,
            "path_frequency": 0.2,
            "vertical_amplitude": 0.0,
            "heading_lookahead": 0.2,
        }
    )
    slam.update(None, None, 1.0)
    pose = slam.update(None, None, 2.0)
    assert pose.matrix[0, 3] == pytest.approx(2.0 * (1.0 - np.cos(0.2)))
    assert pose.matrix[1, 3] == pytest.approx(np.sin(0.2) * 0.5)
    assert pose.matrix[2, 3] == pytest.approx(0.0)
    assert pose.matrix[0, 0] < 1.0


def test_unknown_backend_mode_fails_explicitly():
    with pytest.raises(ValueError):
        SlamWrapper({"mode": "visual_odometry"})


def test_rgbd_vo_tracks_a_textured_translation_with_metric_depth():
    import cv2

    rng = np.random.default_rng(7)
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    for x, y in rng.integers((20, 20), (300, 220), size=(180, 2)):
        cv2.circle(image, (int(x), int(y)), 2, (255, 255, 255), -1)
    shifted = cv2.warpAffine(image, np.float32([[1, 0, -8], [0, 1, 0]]), (320, 240))
    slam = SlamWrapper(
        {
            "mode": "rgbd_vo",
            "camera": {"fx": 250.0, "fy": 250.0, "cx": 160.0, "cy": 120.0},
            "min_correspondences": 12,
            "min_inliers": 8,
        }
    )

    first = slam.update(image, np.full((240, 320), 2.0, dtype=np.float32), 0.0)
    second = slam.update(shifted, np.full((240, 320), 2.0, dtype=np.float32), 1.0)

    assert first.tracking_ok is False
    assert second.tracking_ok is True
    assert abs(second.matrix[0, 3]) > 0.02


def test_groundtruth_mode_uses_pose_hint():
    slam = SlamWrapper({"mode": "groundtruth"})
    pose_matrix = np.eye(4, dtype=np.float32)
    pose_matrix[0, 3] = 1.25
    pose = slam.update(None, None, 1.0, pose_hint=pose_matrix)
    assert pose.matrix[0, 3] == pytest.approx(1.25)
    assert pose.tracking_ok is True


def test_slam_wrapper_uses_accepted_visual_pose_for_trajectory_and_mapping_pose():
    measurement_transform = np.eye(4, dtype=np.float32)
    measurement_transform[0, 3] = 0.25
    measurement = VisualPoseMeasurement(
        timestamp=1.0,
        T_world_camera=measurement_transform,
        covariance=np.eye(6, dtype=np.float32) * 0.01,
        reprojection_rmse=0.5,
        inlier_count=8,
        landmark_ids=("aruco:7",),
    )
    slam = SlamWrapper({"mode": "disabled"}, {"pose_correction": {"apply_to_mapping": True}})

    pose = slam.update(None, None, 1.0, visual_measurement=measurement)

    assert pose.matrix[0, 3] == pytest.approx(0.25)
    assert slam.trajectory.poses[-1].matrix[0, 3] == pytest.approx(0.25)


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


def test_rtabmap_pose_callback_accepts_covariance_pose_message():
    backend = RtabmapBackend({"mode": "rtabmap"})
    pose = type(
        "Pose",
        (),
        {
            "position": type("Position", (), {"x": 1.0, "y": 2.0, "z": 3.0})(),
            "orientation": type("Orientation", (), {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0})(),
        },
    )()
    message = type(
        "PoseWithCovarianceStamped",
        (),
        {
            "header": type("Header", (), {"stamp": type("Stamp", (), {"sec": 4, "nanosec": 0})()})(),
            "pose": type("PoseWithCovariance", (), {"pose": pose})(),
        },
    )()

    backend._pose_callback(message)

    assert backend.get_pose() is not None
    assert backend.get_pose().matrix[2, 3] == pytest.approx(3.0)


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


def test_trajectory_plot_auto_selects_the_plane_with_the_largest_motion():
    poses = [PoseEstimate(np.eye(4, dtype=np.float32), 0.0), PoseEstimate(np.eye(4, dtype=np.float32), 1.0)]
    poses[1].matrix[0, 3] = 3.0
    poses[1].matrix[2, 3] = 4.0

    assert _select_projection_axes(poses, "auto") == (0, 2, "X", "Z")


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


def test_pose_graph_loop_constraint_globally_corrects_translation_drift():
    graph = PoseGraph()
    poses = []
    for index, x in enumerate((0.0, 1.0, 2.0)):
        matrix = np.eye(4, dtype=np.float32)
        matrix[0, 3] = x
        poses.append(PoseEstimate(matrix, float(index)))
        graph.append(poses[-1])
    loop_transform = np.eye(4, dtype=np.float32)
    graph.append(
        PoseEstimate(np.array([[1, 0, 0, 3], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32), 3.0),
        LoopClosureConstraint(source_index=3, target_index=0, distance=0.0, timestamp=3.0, transform=loop_transform),
    )

    assert graph.nodes[-1].pose_matrix[0, 3] < 3.0
    assert graph.nodes[-1].pose_matrix[0, 3] > 0.0
    assert graph.edges[-1].edge_type == "loop_closure"


def test_landmark_measurement_relocalizes_lost_rgbd_tracker():
    measurement_matrix = np.eye(4, dtype=np.float32)
    measurement_matrix[0, 3] = 4.0
    measurement = VisualPoseMeasurement(
        timestamp=0.0,
        T_world_camera=measurement_matrix,
        covariance=np.eye(6, dtype=np.float32) * 0.01,
        reprojection_rmse=0.5,
        inlier_count=8,
        landmark_ids=("aruco:7",),
    )
    slam = SlamWrapper(
        {"mode": "rgbd_vo", "camera": {"fx": 250.0, "fy": 250.0, "cx": 160.0, "cy": 120.0}},
        {"pose_correction": {"apply_to_mapping": True}},
    )

    pose = slam.update(
        np.zeros((240, 320, 3), dtype=np.uint8),
        np.ones((240, 320), dtype=np.float32),
        0.0,
        visual_measurement=measurement,
    )

    assert pose.tracking_ok is True
    assert pose.matrix[0, 3] == pytest.approx(4.0)
    assert slam.last_visual_correction.reason == "relocalized"
    persisted = slam.update(np.zeros((240, 320, 3), dtype=np.uint8), np.ones((240, 320), dtype=np.float32), 1.0)
    assert persisted.matrix[0, 3] == pytest.approx(4.0)


def test_rgbd_vo_accepts_live_intrinsics_updates():
    slam = SlamWrapper({"mode": "rgbd_vo", "camera": {"fx": 250.0, "fy": 250.0, "cx": 160.0, "cy": 120.0}})

    slam.update_camera_intrinsics({"fx": 300.0, "fy": 301.0, "cx": 161.0, "cy": 121.0})

    assert slam.backend._camera_matrix[0, 0] == pytest.approx(300.0)
    assert slam.backend._loop_detector._camera_matrix[1, 1] == pytest.approx(301.0)


def test_pose_graph_disabled_does_not_collect_nodes():
    slam = SlamWrapper({"mode": "dummy", "pose_graph": {"enabled": False}})

    slam.update(None, None, 1.0)
    slam.update(None, None, 2.0)

    assert slam.pose_graph.nodes == []
    assert slam.pose_graph.edges == []


def test_appearance_loop_detector_verifies_rgbd_revisit_and_exports_diagnostics(tmp_path: Path):
    rng = np.random.default_rng(17)
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    for x, y in rng.integers((20, 20), (300, 220), size=(180, 2)):
        cv2.circle(image, (int(x), int(y)), 2, (255, 255, 255), -1)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = cv2.ORB_create(nfeatures=1200).detectAndCompute(gray, None)
    detector = AppearanceLoopClosureDetector(
        {"enabled": True, "min_node_gap": 2, "min_matches": 8, "min_inliers": 6},
        np.array([[250.0, 0.0, 160.0], [0.0, 250.0, 120.0], [0.0, 0.0, 1.0]], dtype=np.float32),
    )
    depth = np.full((240, 320), 2.0, dtype=np.float32)

    assert detector.detect(keypoints, descriptors, depth, 0.0) is None
    assert detector.detect(keypoints, descriptors, depth, 1.0) is None
    closure = detector.detect(keypoints, descriptors, depth, 2.0)

    assert closure is not None
    assert closure.source_index == 2
    assert closure.target_index == 0
    assert closure.transform is not None
    assert closure.inlier_count >= 6
    graph = PoseGraph()
    for timestamp in (0.0, 1.0, 2.0):
        graph.append(PoseEstimate(np.eye(4, dtype=np.float32), timestamp), closure if timestamp == 2.0 else None)
    output = tmp_path / "pose_graph.json"
    graph.export_json(output)
    exported = json.loads(output.read_text(encoding="utf-8"))["loop_closures"][0]
    assert exported["inlier_count"] == closure.inlier_count
    assert exported["transform"] is not None
