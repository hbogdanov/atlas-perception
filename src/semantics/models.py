from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]


def _ensure_yolo_config_dir() -> Path:
    config_dir = Path(os.environ.get("YOLO_CONFIG_DIR", REPO_ROOT / ".cache" / "ultralytics"))
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(config_dir)
    return config_dir


def default_palette() -> dict[int, tuple[int, int, int]]:
    return {
        0: (220, 20, 60),
        1: (0, 128, 255),
        2: (255, 170, 0),
        3: (80, 200, 120),
        4: (180, 80, 255),
        5: (255, 215, 0),
    }


class SemanticBackend(ABC):
    def __init__(self, config: dict) -> None:
        self.config = config

    @abstractmethod
    def predict(self, image: np.ndarray) -> "SemanticPrediction":
        raise NotImplementedError


class DisabledSemanticBackend(SemanticBackend):
    def predict(self, image: np.ndarray) -> "SemanticPrediction":
        height, width = image.shape[:2]
        return SemanticPrediction(
            labels=np.full((height, width), -1, dtype=np.int32),
            class_names={},
            palette=default_palette(),
            scores=None,
        )


class YoloV8SegBackend(SemanticBackend):
    def __init__(self, config: dict) -> None:
        super().__init__(config)
        _ensure_yolo_config_dir()
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "YOLOv8 segmentation requires the 'ultralytics' package. Install it to enable semantics."
            ) from exc
        model_name = str(config.get("model", "yolov8n-seg.pt"))
        try:
            self.model = YOLO(model_name)
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                f"Failed to load YOLOv8 segmentation weights from {model_name!r}. "
                "Ensure the weights file exists locally."
            ) from exc
        self.device = str(config.get("device", "cpu"))
        self.confidence = float(config.get("confidence", 0.25))
        self.iou = float(config.get("iou", 0.7))

    def predict(self, image: np.ndarray) -> "SemanticPrediction":
        results = self.model.predict(image, device=self.device, conf=self.confidence, iou=self.iou, verbose=False)
        result = results[0]
        height, width = image.shape[:2]
        labels = np.full((height, width), -1, dtype=np.int32)
        scores = np.zeros((height, width), dtype=np.float32)
        names = {int(index): str(name) for index, name in result.names.items()}
        if result.masks is not None and result.boxes is not None:
            masks = result.masks.data.detach().cpu().numpy()
            classes = result.boxes.cls.detach().cpu().numpy().astype(np.int32)
            confidences = result.boxes.conf.detach().cpu().numpy().astype(np.float32)
            for mask, class_id, confidence in zip(masks, classes, confidences, strict=False):
                resized = cv2.resize(mask.astype(np.float32), (width, height), interpolation=cv2.INTER_NEAREST) > 0.5
                overwrite = resized & (confidence >= scores)
                labels[overwrite] = int(class_id)
                scores[overwrite] = float(confidence)
        return SemanticPrediction(labels=labels, class_names=names, palette=default_palette(), scores=scores)


class SemanticPrediction:
    def __init__(
        self,
        labels: np.ndarray,
        class_names: dict[int, str] | None = None,
        palette: dict[int, tuple[int, int, int]] | None = None,
        scores: np.ndarray | None = None,
    ) -> None:
        self.labels = labels.astype(np.int32)
        self.class_names = class_names or {}
        self.palette = palette or default_palette()
        self.scores = None if scores is None else scores.astype(np.float32)

    def colorize(self) -> np.ndarray:
        colored = np.zeros((*self.labels.shape, 3), dtype=np.uint8)
        for class_id in np.unique(self.labels):
            if class_id < 0:
                continue
            color = self.palette.get(int(class_id), self.palette[int(class_id) % len(self.palette)])
            colored[self.labels == class_id] = np.array(color, dtype=np.uint8)
        return colored

    def overlay(self, image: np.ndarray, alpha: float = 0.45) -> np.ndarray:
        semantic = self.colorize()
        blended = image.copy()
        mask = self.labels >= 0
        if np.any(mask):
            image_pixels = image[mask].astype(np.float32)
            semantic_pixels = semantic[mask].astype(np.float32)
            mixed = image_pixels * (1.0 - alpha) + semantic_pixels * alpha
            blended[mask] = np.clip(mixed, 0, 255).astype(np.uint8)
        return blended

    def sample(self, stride: int) -> tuple[np.ndarray, np.ndarray]:
        sampled_labels = self.labels[::stride, ::stride].reshape(-1)
        semantic_colors = self.colorize()[::stride, ::stride].reshape(-1, 3).astype(np.float32) / 255.0
        return sampled_labels.astype(np.int32), semantic_colors


_SEMANTIC_BACKENDS: dict[str, type[SemanticBackend]] = {
    "disabled": DisabledSemanticBackend,
    "yolov8_seg": YoloV8SegBackend,
}


def get_semantic_backend_class(name: str) -> type[SemanticBackend]:
    backend = _SEMANTIC_BACKENDS.get(name.lower())
    if backend is None:
        raise ValueError(f"Unknown semantic backend {name!r}. Available backends: {sorted(_SEMANTIC_BACKENDS)}")
    return backend


def get_registered_semantic_backends() -> list[str]:
    return sorted(_SEMANTIC_BACKENDS)
