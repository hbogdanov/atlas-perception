import cv2
import numpy as np

from src.vision.aruco_landmarks import ArucoLandmarkDetector, build_marker_observations
from src.vision.landmark_pose import solve_landmark_pose


def test_marker_observations_transform_known_marker_corners_into_world_frame():
    transform = np.eye(4, dtype=np.float32)
    transform[:3, 3] = [1.0, 2.0, 3.0]
    detections = [(7, np.array([[10, 10], [20, 10], [20, 20], [10, 20]], dtype=np.float32))]

    observations = build_marker_observations(detections, {7: transform}, marker_length_m=0.2)

    assert len(observations) == 1
    assert observations[0].landmark_id == "aruco:7"
    assert np.allclose(observations[0].world_points[0], [0.9, 2.1, 3.0])


def test_marker_observations_ignore_unknown_marker_ids():
    observations = build_marker_observations([(9, np.zeros((4, 2), dtype=np.float32))], {}, marker_length_m=0.2)

    assert observations == []


def test_aruco_detector_creates_known_landmark_observation():
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    marker = cv2.aruco.generateImageMarker(dictionary, 7, 160)
    canvas = np.full((240, 240), 255, dtype=np.uint8)
    canvas[40:200, 40:200] = marker
    image = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    detector = ArucoLandmarkDetector(
        {
            "dictionary": "DICT_4X4_50",
            "marker_length_m": 0.2,
            "landmarks": [{"id": 7, "T_world_marker": np.eye(4, dtype=np.float32).tolist()}],
        }
    )

    observations = detector.detect(image)

    assert len(observations) == 1
    assert observations[0].world_points.shape == (4, 3)


def test_detected_aruco_corners_produce_a_visual_pose_measurement():
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    marker = cv2.aruco.generateImageMarker(dictionary, 7, 160)
    canvas = np.full((240, 240), 255, dtype=np.uint8)
    canvas[40:200, 40:200] = marker
    detector = ArucoLandmarkDetector(
        {
            "dictionary": "DICT_4X4_50",
            "marker_length_m": 0.2,
            "landmarks": [{"id": 7, "T_world_marker": np.eye(4, dtype=np.float32).tolist()}],
        }
    )

    measurement = solve_landmark_pose(
        detector.detect(cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)), {"fx": 200, "fy": 200, "cx": 120, "cy": 120}, 1.0
    )

    assert measurement is not None
    assert measurement.inlier_count == 4
    assert measurement.reprojection_rmse < 3.0
