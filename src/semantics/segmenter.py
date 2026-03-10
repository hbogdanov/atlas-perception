from __future__ import annotations

import numpy as np

from src.semantics.models import SemanticPrediction, get_semantic_backend_class


class SemanticSegmenter:
    def __init__(self, config: dict | None = None) -> None:
        self.config = config or {}
        self.enabled = bool(self.config.get("enabled", False))
        backend_name = str(self.config.get("backend", "disabled" if not self.enabled else "yolov8_seg")).lower()
        if not self.enabled:
            backend_name = "disabled"
        self.backend_name = backend_name
        self.backend = get_semantic_backend_class(backend_name)(self.config)

    def predict(self, image: np.ndarray) -> SemanticPrediction | None:
        if not self.enabled:
            return None
        return self.backend.predict(image)
