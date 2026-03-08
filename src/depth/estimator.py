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
        self.backend = self._load_backend()

    def predict(self, image: np.ndarray) -> np.ndarray:
        depth = self.backend.predict(image)
        if self.config.get("normalize", True):
            depth = cv2.normalize(depth, None, 0.0, 1.0, cv2.NORM_MINMAX)
        min_depth = float(self.config.get("min_depth", 0.1))
        max_depth = float(self.config.get("max_depth", 20.0))
        depth = np.clip(depth, min_depth, max_depth)
        return depth

    def _load_backend(self):
        model_name = self.model_name.lower()
        if model_name == "midas":
            return MidasBackend(self.config, self.device)
        if model_name == "depth_anything":
            return DepthAnythingBackend(self.config, self.device)
        raise ValueError(f"Unsupported depth backend: {self.model_name}")


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
