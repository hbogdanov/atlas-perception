from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.vision.aruco_landmarks import ArucoLandmarkDetector
from src.vision.landmark_pose import solve_landmark_pose


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Atlas ArUco detection and PnP on a labeled synthetic sequence."
    )
    parser.add_argument("--frames", type=int, default=20)
    parser.add_argument("--output-dir", default="data/outputs/visual_localization_synthetic")
    return parser.parse_args()


def run_evaluation(frames: int, output_dir: Path) -> dict:
    intrinsics = {"fx": 450.0, "fy": 450.0, "cx": 320.0, "cy": 240.0}
    detector = ArucoLandmarkDetector(
        {
            "dictionary": "DICT_4X4_50",
            "marker_length_m": 0.2,
            "landmarks": [{"id": 7, "T_world_marker": np.eye(4).tolist()}],
        }
    )
    rows = []
    for index in range(frames):
        T_camera_world = _camera_pose(index, frames)
        image = _render_marker(T_camera_world, intrinsics)
        measurement = solve_landmark_pose(detector.detect(image), intrinsics, float(index))
        if measurement is None:
            rows.append({"frame": index, "detected": False, "translation_error_m": None, "rotation_error_deg": None})
            continue
        expected = np.linalg.inv(T_camera_world)
        rows.append(
            {
                "frame": index,
                "detected": True,
                "translation_error_m": float(np.linalg.norm(measurement.T_world_camera[:3, 3] - expected[:3, 3])),
                "rotation_error_deg": _rotation_error_deg(measurement.T_world_camera[:3, :3], expected[:3, :3]),
                "reprojection_rmse_px": measurement.reprojection_rmse,
                "inliers": measurement.inlier_count,
            }
        )
    valid = [row for row in rows if row["detected"]]
    summary = {
        "evaluation": "synthetic_rendered_aruco_with_exact_ground_truth",
        "frames": frames,
        "detections": len(valid),
        "detection_rate": len(valid) / max(frames, 1),
        "translation_rmse_m": (
            float(np.sqrt(np.mean([row["translation_error_m"] ** 2 for row in valid]))) if valid else None
        ),
        "rotation_rmse_deg": (
            float(np.sqrt(np.mean([row["rotation_error_deg"] ** 2 for row in valid]))) if valid else None
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with (output_dir / "per_frame.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)
    return summary


def _camera_pose(index: int, frames: int) -> np.ndarray:
    phase = (index / max(frames - 1, 1) - 0.5) * 0.35
    perturbation, _ = cv2.Rodrigues(np.array([0.05, phase, 0.0], dtype=np.float64))
    front_facing_marker = np.diag([1.0, -1.0, -1.0])
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = perturbation @ front_facing_marker
    transform[:3, 3] = [0.12 * np.sin(phase * 4), 0.04, 1.4]
    return transform


def _render_marker(T_camera_world: np.ndarray, intrinsics: dict) -> np.ndarray:
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    marker = cv2.aruco.generateImageMarker(dictionary, 7, 180)
    marker_corners = np.array([[-0.1, 0.1, 0.0], [0.1, 0.1, 0.0], [0.1, -0.1, 0.0], [-0.1, -0.1, 0.0]])
    rvec, _ = cv2.Rodrigues(T_camera_world[:3, :3])
    image_corners, _ = cv2.projectPoints(
        marker_corners,
        rvec,
        T_camera_world[:3, 3],
        np.array([[intrinsics["fx"], 0, intrinsics["cx"]], [0, intrinsics["fy"], intrinsics["cy"]], [0, 0, 1]]),
        None,
    )
    source = np.array([[0, 0], [179, 0], [179, 179], [0, 179]], dtype=np.float32)
    homography = cv2.getPerspectiveTransform(source, image_corners.reshape(4, 2).astype(np.float32))
    canvas = np.full((480, 640), 255, dtype=np.uint8)
    warped = cv2.warpPerspective(marker, homography, (640, 480), borderValue=255)
    mask = cv2.warpPerspective(np.full(marker.shape, 255, dtype=np.uint8), homography, (640, 480))
    canvas[mask > 0] = warped[mask > 0]
    return cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)


def _rotation_error_deg(estimated: np.ndarray, expected: np.ndarray) -> float:
    relative = estimated.T @ expected
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def main() -> None:
    args = parse_args()
    summary = run_evaluation(args.frames, Path(args.output_dir))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
