from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch


class DepthBackend:
    def __init__(self, config: dict, device: torch.device) -> None:
        self.config = config
        self.device = device

    def predict(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError


_DEPTH_BACKEND_REGISTRY: dict[str, type[DepthBackend]] = {}


def register_depth_backend(name: str):
    normalized = str(name).lower()

    def decorator(cls: type[DepthBackend]) -> type[DepthBackend]:
        _DEPTH_BACKEND_REGISTRY[normalized] = cls
        return cls

    return decorator


def get_depth_backend_class(name: str) -> type[DepthBackend]:
    normalized = str(name).lower()
    try:
        return _DEPTH_BACKEND_REGISTRY[normalized]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported depth backend: {name}. Available backends: {', '.join(sorted(_DEPTH_BACKEND_REGISTRY))}"
        ) from exc


def get_registered_depth_backends() -> dict[str, str]:
    return {name: cls.__name__ for name, cls in sorted(_DEPTH_BACKEND_REGISTRY.items())}


class TorchHubDepthBackend(DepthBackend):
    repo: str
    model_arg: str
    transform_arg: str

    def __init__(self, config: dict, device: torch.device) -> None:
        super().__init__(config, device)
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


@register_depth_backend("midas")
class MidasBackend(TorchHubDepthBackend):
    repo = "intel-isl/MiDaS"

    def __init__(self, config: dict, device: torch.device) -> None:
        self.model_arg = config.get("midas_variant", "MiDaS_small")
        self.transform_arg = "small_transform" if self.model_arg == "MiDaS_small" else "dpt_transform"
        super().__init__(config, device)


@register_depth_backend("depth_anything")
class DepthAnythingBackend(TorchHubDepthBackend):
    repo = "LiheYoung/Depth-Anything"

    def __init__(self, config: dict, device: torch.device) -> None:
        encoder = config.get("depth_anything_encoder", "vitl")
        self.model_arg = f"depth_anything_{encoder}14"
        self.transform_arg = "dpt_transform"
        super().__init__(config, device)
