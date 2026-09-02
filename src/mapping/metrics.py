from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MapMetrics:
    chamfer_distance: float
    estimated_to_reference_mean: float
    reference_to_estimated_mean: float
    completeness: float
    estimated_points: int
    reference_points: int

    def to_dict(self) -> dict[str, float | int]:
        return {
            "chamfer_distance": self.chamfer_distance,
            "estimated_to_reference_mean": self.estimated_to_reference_mean,
            "reference_to_estimated_mean": self.reference_to_estimated_mean,
            "completeness": self.completeness,
            "estimated_points": self.estimated_points,
            "reference_points": self.reference_points,
        }


def compute_map_metrics(
    estimated: np.ndarray, reference: np.ndarray, completeness_threshold: float = 0.1
) -> MapMetrics:
    estimated_points = _validate_points(estimated, "estimated")
    reference_points = _validate_points(reference, "reference")
    estimated_distances = _nearest_distances(estimated_points, reference_points)
    reference_distances = _nearest_distances(reference_points, estimated_points)
    return MapMetrics(
        chamfer_distance=float(np.mean(estimated_distances) + np.mean(reference_distances)),
        estimated_to_reference_mean=float(np.mean(estimated_distances)),
        reference_to_estimated_mean=float(np.mean(reference_distances)),
        completeness=float(np.mean(reference_distances <= completeness_threshold)),
        estimated_points=int(estimated_points.shape[0]),
        reference_points=int(reference_points.shape[0]),
    )


def _nearest_distances(source: np.ndarray, target: np.ndarray, block_size: int = 2048) -> np.ndarray:
    distances = np.empty(source.shape[0], dtype=np.float32)
    for start in range(0, source.shape[0], block_size):
        block = source[start : start + block_size]
        squared = np.sum((block[:, None, :] - target[None, :, :]) ** 2, axis=2)
        distances[start : start + block.shape[0]] = np.sqrt(np.min(squared, axis=1))
    return distances


def _validate_points(points: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(points, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 3 or values.shape[0] == 0:
        raise ValueError(f"{name} map must contain at least one point with shape (N, 3).")
    return values
