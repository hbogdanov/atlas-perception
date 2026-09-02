import numpy as np
import pytest

from src.mapping.metrics import compute_map_metrics
from tools.evaluate_map import downsample_points


def test_map_metrics_are_zero_for_identical_point_clouds():
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)

    metrics = compute_map_metrics(points, points, completeness_threshold=0.01)

    assert metrics.chamfer_distance == pytest.approx(0.0)
    assert metrics.completeness == pytest.approx(1.0)


def test_map_metrics_measure_missing_reference_coverage():
    estimated = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    reference = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32)

    metrics = compute_map_metrics(estimated, reference, completeness_threshold=0.2)

    assert metrics.completeness == pytest.approx(0.5)
    assert metrics.chamfer_distance > 0.0


def test_map_evaluation_downsampling_is_deterministic():
    points = np.arange(30, dtype=np.float32).reshape(10, 3)

    sampled = downsample_points(points, max_points=4)

    assert np.array_equal(sampled, points[[0, 3, 6, 9]])
