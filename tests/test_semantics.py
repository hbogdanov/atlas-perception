import numpy as np
import pytest

from src.semantics.models import SemanticPrediction, get_registered_semantic_backends, get_semantic_backend_class
from src.semantics.segmenter import SemanticSegmenter


def test_disabled_segmenter_returns_none():
    segmenter = SemanticSegmenter({"enabled": False})
    assert segmenter.predict(np.zeros((2, 2, 3), dtype=np.uint8)) is None


def test_registered_semantic_backends_include_yolov8_and_disabled():
    backends = get_registered_semantic_backends()
    assert "disabled" in backends
    assert "yolov8_seg" in backends


def test_unknown_semantic_backend_fails_explicitly():
    with pytest.raises(ValueError):
        get_semantic_backend_class("sam2")


def test_semantic_prediction_overlay_and_sampling():
    prediction = SemanticPrediction(labels=np.array([[0, 1], [-1, 1]], dtype=np.int32), class_names={0: "a", 1: "b"})
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    overlay = prediction.overlay(image)
    labels, colors = prediction.sample(stride=1)
    assert overlay.shape == image.shape
    assert labels.tolist() == [0, 1, -1, 1]
    assert colors.shape == (4, 3)
