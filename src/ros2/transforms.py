from __future__ import annotations

import numpy as np


def make_transform(translation: tuple[float, float, float]) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, 3] = np.array(translation, dtype=np.float32)
    return matrix

