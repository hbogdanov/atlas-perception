from __future__ import annotations

import cv2
import numpy as np

from src.vision.landmark_pose import LandmarkObservation


class ArucoLandmarkDetector:
    """Convert known ArUco markers into world-frame PnP correspondences."""

    def __init__(self, config: dict) -> None:
        if not hasattr(cv2, "aruco"):
            raise RuntimeError("ArUco landmark localization requires an OpenCV build with cv2.aruco.")
        self.marker_length_m = float(config.get("marker_length_m", 0.16))
        dictionary_name = str(config.get("dictionary", "DICT_4X4_50"))
        dictionary_id = getattr(cv2.aruco, dictionary_name, None)
        if dictionary_id is None:
            raise ValueError(f"Unsupported ArUco dictionary: {dictionary_name}")
        self._dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
        self._detector = cv2.aruco.ArucoDetector(self._dictionary) if hasattr(cv2.aruco, "ArucoDetector") else None
        self._world_markers = _parse_world_markers(config.get("landmarks", []))

    def detect(self, image: np.ndarray) -> list[LandmarkObservation]:
        if self._detector is not None:
            corners, ids, _ = self._detector.detectMarkers(image)
        else:  # pragma: no cover - compatibility with older OpenCV builds
            corners, ids, _ = cv2.aruco.detectMarkers(image, self._dictionary)
        if ids is None:
            return []
        detections = [
            (int(marker_id), marker_corners.reshape(4, 2)) for marker_id, marker_corners in zip(ids.ravel(), corners)
        ]
        return build_marker_observations(detections, self._world_markers, self.marker_length_m)


def build_marker_observations(
    detections: list[tuple[int, np.ndarray]], world_markers: dict[int, np.ndarray], marker_length_m: float
) -> list[LandmarkObservation]:
    half_size = float(marker_length_m) * 0.5
    marker_corners = np.array(
        [
            [-half_size, half_size, 0.0],
            [half_size, half_size, 0.0],
            [half_size, -half_size, 0.0],
            [-half_size, -half_size, 0.0],
        ],
        dtype=np.float32,
    )
    observations: list[LandmarkObservation] = []
    for marker_id, image_corners in detections:
        T_world_marker = world_markers.get(marker_id)
        if T_world_marker is None:
            continue
        homogeneous = np.column_stack([marker_corners, np.ones(4, dtype=np.float32)])
        world_points = (homogeneous @ T_world_marker.T)[:, :3]
        observations.append(
            LandmarkObservation(
                landmark_id=f"aruco:{marker_id}",
                world_points=world_points.astype(np.float32),
                image_points=np.asarray(image_corners, dtype=np.float32),
            )
        )
    return observations


def _parse_world_markers(entries: list[dict]) -> dict[int, np.ndarray]:
    markers: dict[int, np.ndarray] = {}
    for entry in entries:
        marker_id = int(entry["id"])
        transform = np.asarray(entry["T_world_marker"], dtype=np.float32)
        if transform.shape != (4, 4):
            raise ValueError(f"Landmark {marker_id} T_world_marker must have shape (4, 4).")
        markers[marker_id] = transform
    return markers
