from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch


@dataclass
class DepthPrediction:
    depth: np.ndarray
    confidence: np.ndarray | None = None


class DepthEstimator:
    """Load and run a pretrained monocular depth backend."""

    def __init__(self, config: dict) -> None:
        self.config = config
        self.model_name = config["model"]
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
        model_name = self.model_name.lower()
        if model_name == "midas":
            return MidasBackend(self.config, self.device)
        if model_name == "depth_anything":
            return DepthAnythingBackend(self.config, self.device)
        raise ValueError(f"Unsupported depth backend: {self.model_name}")

    def _format_output(self, depth: np.ndarray) -> np.ndarray:
        if self.output_mode == "relative_normalized":
            return cv2.normalize(depth, None, 0.0, 1.0, cv2.NORM_MINMAX)
        if self.output_mode == "raw":
            return depth.astype(np.float32)
        raise ValueError(
            f"Unsupported depth output_mode: {self.output_mode}. "
            "Expected one of: relative_normalized, raw."
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


class _TorchHubDepthBackend:
    repo: str
    model_arg: str
    transform_arg: str

    def __init__(self, config: dict, device: torch.device) -> None:
        self.config = config
        self.device = device
        weights_dir = Path(config.get("weights_dir", "models"))
        weights_dir.mkdir(parents=True, exist_ok=True)
        torch.hub.set_dir(str(weights_dir.resolve()))

        prefer_torch_hub = bool(config.get("prefer_torch_hub", True))
        if not prefer_torch_hub:
            raise RuntimeError("Only torch.hub loading is currently implemented for this backend.")

        self.model = torch.hub.load(self.repo, self.model_arg, pretrained=True)
        self.model.to(self.device)
        self.model.eval()
        transforms = torch.hub.load(self.repo, "transforms")
        self.transform = getattr(transforms, self.transform_arg)
        self._load_local_weights_if_configured()

    def predict(self, image: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_batch = self.transform(rgb).to(self.device)
        with torch.no_grad():
            prediction = self.model(input_batch)
            if prediction.ndim == 3:
                prediction = prediction.unsqueeze(1)
            prediction = torch.nn.functional.interpolate(
                prediction,
                size=rgb.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        return prediction.detach().cpu().numpy().astype(np.float32)

    def _load_local_weights_if_configured(self) -> None:
        weights_path = self.config.get("local_weights_path")
        if not weights_path:
            return
        checkpoint = torch.load(weights_path, map_location=self.device)
        state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        self.model.load_state_dict(state_dict, strict=False)


class MidasBackend(_TorchHubDepthBackend):
    repo = "intel-isl/MiDaS"

    def __init__(self, config: dict, device: torch.device) -> None:
        self.model_arg = config.get("midas_variant", "MiDaS_small")
        self.transform_arg = "small_transform" if self.model_arg == "MiDaS_small" else "dpt_transform"
        super().__init__(config, device)


class DepthAnythingBackend(_TorchHubDepthBackend):
    repo = "LiheYoung/Depth-Anything"

    def __init__(self, config: dict, device: torch.device) -> None:
        encoder = config.get("depth_anything_encoder", "vitl")
        self.model_arg = f"depth_anything_{encoder}14"
        self.transform_arg = "dpt_transform"
        super().__init__(config, device)
