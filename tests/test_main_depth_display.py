import numpy as np

from src.main import _display_depth_map


def test_display_depth_normalizes_relative_model_output_before_colorization():
    display = _display_depth_map(np.array([[5.0, 15.0, 25.0]], dtype=np.float32), {"depth": {}})

    assert display.min() >= 0.0
    assert display.max() <= 1.0
    assert not np.allclose(display[0, 0], display[0, 2])
