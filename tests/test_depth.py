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
    estimator.postprocess_config = {}
    estimator._previous_depth = None
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


def test_bilateral_postprocess_changes_depth_when_enabled():
    estimator = build_estimator(
        "raw",
        np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
    )
    estimator.postprocess_config = {
        "enabled": True,
        "bilateral_filter": True,
        "bilateral_diameter": 3,
        "bilateral_sigma_color": 1.0,
        "bilateral_sigma_space": 1.0,
    }
    depth = estimator.predict(np.zeros((3, 3, 3), dtype=np.uint8))
    assert depth[1, 1] < 1.0


def test_guided_refine_preserves_rgb_discontinuity_better_than_plain_blur():
    estimator = build_estimator(
        "raw",
        np.array(
            [
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 1.0, 1.0],
            ],
            dtype=np.float32,
        ),
    )
    estimator.postprocess_config = {
        "enabled": True,
        "guided_refine": True,
        "guided_radius": 1,
        "guided_eps": 1e-6,
    }
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    image[:, :2] = 0
    image[:, 2:] = 255
    depth = estimator.predict(image)
    assert depth[1, 1] < 0.25
    assert depth[1, 2] > 0.75


def test_temporal_fusion_blends_with_previous_frame():
    estimator = build_estimator("raw", np.ones((2, 2), dtype=np.float32))
    estimator.postprocess_config = {
        "enabled": True,
        "temporal_fusion": True,
        "temporal_alpha": 0.25,
    }

    first = estimator.predict(np.zeros((2, 2, 3), dtype=np.uint8))
    estimator.backend = FakeBackend(np.zeros((2, 2), dtype=np.float32))
    second = estimator.predict(np.zeros((2, 2, 3), dtype=np.uint8))

    assert np.array_equal(first, np.ones((2, 2), dtype=np.float32))
    assert np.allclose(second, np.full((2, 2), 0.75, dtype=np.float32))
