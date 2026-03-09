import numpy as np
import pytest

from src.depth.estimator import DepthEstimator


class FakeBackend:
    def __init__(self, output: np.ndarray) -> None:
        self.output = output.astype(np.float32)

    def predict(self, image: np.ndarray) -> np.ndarray:
        del image
        return self.output.copy()


def build_estimator(output_mode: str, depth_output: np.ndarray) -> DepthEstimator:
    estimator = DepthEstimator.__new__(DepthEstimator)
    estimator.config = {"model": "fake", "output_mode": output_mode}
    estimator.model_name = "fake"
    estimator.output_mode = output_mode
    estimator.backend = FakeBackend(depth_output)
    return estimator


def test_relative_normalized_output_scales_between_zero_and_one():
    estimator = build_estimator("relative_normalized", np.array([[2.0, 4.0]], dtype=np.float32))
    depth = estimator.predict(np.zeros((1, 2, 3), dtype=np.uint8))
    assert depth.min() == pytest.approx(0.0)
    assert depth.max() == pytest.approx(1.0)


def test_raw_output_preserves_backend_values():
    estimator = build_estimator("raw", np.array([[2.5, 7.5]], dtype=np.float32))
    depth = estimator.predict(np.zeros((1, 2, 3), dtype=np.uint8))
    assert np.array_equal(depth, np.array([[2.5, 7.5]], dtype=np.float32))


def test_unknown_output_mode_fails_explicitly():
    estimator = build_estimator("metricish", np.array([[1.0]], dtype=np.float32))
    with pytest.raises(ValueError):
        estimator.predict(np.zeros((1, 1, 3), dtype=np.uint8))
