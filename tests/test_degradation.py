import numpy as np
import pytest

from src.depth.degradation import degrade_image


@pytest.mark.parametrize("kind", ["brightness", "gaussian_noise", "motion_blur", "resolution", "occlusion"])
def test_degradations_preserve_image_shape_and_type(kind: str):
    image = np.full((20, 30, 3), 180, dtype=np.uint8)

    degraded = degrade_image(image, kind, severity=0.7, seed=17)

    assert degraded.shape == image.shape
    assert degraded.dtype == np.uint8


def test_gaussian_noise_is_deterministic_for_a_fixed_seed():
    image = np.full((12, 12, 3), 128, dtype=np.uint8)

    first = degrade_image(image, "gaussian_noise", severity=0.5, seed=9)
    second = degrade_image(image, "gaussian_noise", severity=0.5, seed=9)

    assert np.array_equal(first, second)


def test_unknown_degradation_is_rejected():
    with pytest.raises(ValueError):
        degrade_image(np.zeros((2, 2, 3), dtype=np.uint8), "rainbow", severity=0.5)
