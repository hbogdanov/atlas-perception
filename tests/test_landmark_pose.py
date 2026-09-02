import cv2
import numpy as np

from src.vision.landmark_pose import LandmarkObservation, solve_landmark_pose


def test_landmark_pnp_recovers_camera_pose_from_projected_corners():
    intrinsics = {"fx": 400.0, "fy": 400.0, "cx": 320.0, "cy": 240.0}
    world_points = np.array([[-0.5, -0.5, 0.0], [0.5, -0.5, 0.0], [0.5, 0.5, 0.0], [-0.5, 0.5, 0.0]], dtype=np.float32)
    camera_matrix = np.array([[400.0, 0.0, 320.0], [0.0, 400.0, 240.0], [0.0, 0.0, 1.0]])
    rotation_vector = np.array([[0.0], [0.0], [0.05]], dtype=np.float64)
    translation_vector = np.array([[0.1], [-0.05], [2.0]], dtype=np.float64)
    image_points, _ = cv2.projectPoints(world_points, rotation_vector, translation_vector, camera_matrix, None)
    observation = LandmarkObservation("tag-1", world_points, image_points.reshape(-1, 2))

    measurement = solve_landmark_pose([observation], intrinsics, timestamp=42.0)

    assert measurement is not None
    assert measurement.timestamp == 42.0
    assert measurement.inlier_count == 4
    assert measurement.reprojection_rmse < 1e-3
    rotation, _ = cv2.Rodrigues(rotation_vector)
    expected = np.eye(4, dtype=np.float32)
    expected[:3, :3] = rotation.astype(np.float32)
    expected[:3, 3] = translation_vector.reshape(3)
    assert np.allclose(measurement.T_world_camera, np.linalg.inv(expected), atol=1e-3)


def test_landmark_pnp_rejects_insufficient_correspondences():
    observation = LandmarkObservation(
        "tag-1",
        np.zeros((3, 3), dtype=np.float32),
        np.zeros((3, 2), dtype=np.float32),
    )

    assert solve_landmark_pose([observation], {"fx": 1, "fy": 1, "cx": 0, "cy": 0}, 0.0) is None
