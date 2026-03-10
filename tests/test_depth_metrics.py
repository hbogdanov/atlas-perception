from pathlib import Path

import cv2
import numpy as np
import pytest

from src.depth.metrics import align_depth_scale, compute_depth_metrics, decode_tum_depth_png
from tools.evaluate_depth import list_tum_pairs


def test_align_depth_scale_matches_target_median():
    prediction = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    target = prediction * 2.5
    aligned = align_depth_scale(prediction, target)
    assert np.allclose(aligned, target)


def test_compute_depth_metrics_returns_expected_values():
    prediction = np.array([1.0, 2.0, 4.0], dtype=np.float32)
    target = np.array([1.0, 1.0, 2.0], dtype=np.float32)
    metrics = compute_depth_metrics(prediction, target)
    assert metrics.abs_rel == pytest.approx(2.0 / 3.0)
    assert metrics.rmse == pytest.approx(np.sqrt(5.0 / 3.0))
    assert metrics.delta1 == pytest.approx(1.0 / 3.0)
    assert metrics.valid_pixels == 3


def test_decode_tum_depth_png_converts_to_meters():
    encoded = np.array([[0, 5000, 10000]], dtype=np.uint16)
    decoded = decode_tum_depth_png(encoded)
    assert np.allclose(decoded, np.array([[0.0, 1.0, 2.0]], dtype=np.float32))


def test_decode_tum_depth_png_rejects_non_uint16():
    with pytest.raises(ValueError):
        decode_tum_depth_png(np.array([[1.0]], dtype=np.float32))


def test_list_tum_pairs_matches_rgb_and_depth_stems(tmp_path: Path):
    rgb_dir = tmp_path / "rgb"
    depth_dir = tmp_path / "depth"
    rgb_dir.mkdir()
    depth_dir.mkdir()

    rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    depth = np.zeros((2, 2), dtype=np.uint16)
    cv2.imwrite(str(rgb_dir / "1.0.png"), rgb)
    cv2.imwrite(str(depth_dir / "1.0.png"), depth)
    cv2.imwrite(str(rgb_dir / "2.0.png"), rgb)

    pairs = list_tum_pairs(tmp_path)
    assert pairs == [(rgb_dir / "1.0.png", depth_dir / "1.0.png")]
