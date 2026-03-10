from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch

from src.depth.models import get_depth_backend_class


@dataclass
class DepthPrediction:
    depth: np.ndarray
    confidence: np.ndarray | None = None


class DepthEstimator:
    """Load and run a pretrained monocular depth backend."""

    def __init__(self, config: dict) -> None:
        self.config = config
        self.model_name = str(config.get("depth_model", config.get("model", "midas")))
        self.device = torch.device(config.get("device", "cpu"))
        self.output_mode = str(config.get("output_mode", "relative_normalized")).lower()
        self.postprocess_config = dict(config.get("postprocess", {}))
        self._previous_depth: np.ndarray | None = None
        try:
            self.backend = self._load_backend()
        except Exception as exc:
            raise RuntimeError(
                f"Failed to initialize depth backend '{self.model_name}'. "
                "Check torch installation, first-run model downloads, and local weight paths."
            ) from exc

    def predict(self, image: np.ndarray) -> np.ndarray:
        depth = self.backend.predict(image)
        depth = self._postprocess_depth(depth, image)
        return self._format_output(depth)

    def _load_backend(self):
        backend_cls = get_depth_backend_class(self.model_name)
        return backend_cls(self.config, self.device)

    def _format_output(self, depth: np.ndarray) -> np.ndarray:
        if self.output_mode == "relative_normalized":
            return cv2.normalize(depth, None, 0.0, 1.0, cv2.NORM_MINMAX)
        if self.output_mode == "raw":
            return depth.astype(np.float32)
        raise ValueError(
            f"Unsupported depth output_mode: {self.output_mode}. " "Expected one of: relative_normalized, raw."
        )

    def _postprocess_depth(self, depth: np.ndarray, image: np.ndarray) -> np.ndarray:
        postprocess = self.postprocess_config
        if not postprocess.get("enabled", False):
            return depth.astype(np.float32)

        refined = depth.astype(np.float32)

        if postprocess.get("bilateral_filter", False):
            diameter = int(postprocess.get("bilateral_diameter", 7))
            sigma_color = float(postprocess.get("bilateral_sigma_color", 0.1))
            sigma_space = float(postprocess.get("bilateral_sigma_space", 7.0))
            refined = cv2.bilateralFilter(refined, diameter, sigma_color, sigma_space)

        if postprocess.get("guided_refine", False):
            radius = int(postprocess.get("guided_radius", 8))
            eps = float(postprocess.get("guided_eps", 1e-3))
            refined = _guided_filter_depth(image, refined, radius, eps)

        if postprocess.get("temporal_fusion", False):
            alpha = float(postprocess.get("temporal_alpha", 0.7))
            if self._previous_depth is not None and self._previous_depth.shape == refined.shape:
                refined = alpha * refined + (1.0 - alpha) * self._previous_depth
            self._previous_depth = refined.copy()

        return refined.astype(np.float32)


def _guided_filter_depth(image: np.ndarray, depth: np.ndarray, radius: int, eps: float) -> np.ndarray:
    """Refine depth with a grayscale guided filter to preserve RGB edges."""
    guide = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    src = depth.astype(np.float32)
    kernel = (2 * radius + 1, 2 * radius + 1)

    mean_i = cv2.boxFilter(guide, cv2.CV_32F, kernel, normalize=True)
    mean_p = cv2.boxFilter(src, cv2.CV_32F, kernel, normalize=True)
    mean_ip = cv2.boxFilter(guide * src, cv2.CV_32F, kernel, normalize=True)
    cov_ip = mean_ip - mean_i * mean_p

    mean_ii = cv2.boxFilter(guide * guide, cv2.CV_32F, kernel, normalize=True)
    var_i = mean_ii - mean_i * mean_i

    a = cov_ip / (var_i + eps)
    b = mean_p - a * mean_i
    mean_a = cv2.boxFilter(a, cv2.CV_32F, kernel, normalize=True)
    mean_b = cv2.boxFilter(b, cv2.CV_32F, kernel, normalize=True)
    return mean_a * guide + mean_b
