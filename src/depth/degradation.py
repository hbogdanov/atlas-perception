from __future__ import annotations

import cv2
import numpy as np


def degrade_image(image: np.ndarray, kind: str, severity: float, seed: int = 0) -> np.ndarray:
    """Apply one deterministic, bounded visual degradation for robustness evaluation."""
    normalized_kind = str(kind).lower()
    strength = float(np.clip(severity, 0.0, 1.0))
    source = np.asarray(image, dtype=np.uint8)
    if normalized_kind == "brightness":
        gain = 1.0 - 0.65 * strength
        return cv2.convertScaleAbs(source, alpha=gain, beta=-35.0 * strength)
    if normalized_kind == "gaussian_noise":
        rng = np.random.default_rng(seed)
        noise = rng.normal(0.0, 40.0 * strength, source.shape)
        return np.clip(source.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if normalized_kind == "motion_blur":
        kernel_size = max(1, int(round(1 + 14 * strength)))
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        kernel[kernel_size // 2, :] = 1.0 / kernel_size
        return cv2.filter2D(source, -1, kernel)
    if normalized_kind == "resolution":
        scale = max(0.2, 1.0 - 0.7 * strength)
        height, width = source.shape[:2]
        reduced = cv2.resize(
            source, (max(1, int(width * scale)), max(1, int(height * scale))), interpolation=cv2.INTER_AREA
        )
        return cv2.resize(reduced, (width, height), interpolation=cv2.INTER_LINEAR)
    if normalized_kind == "occlusion":
        output = source.copy()
        height, width = output.shape[:2]
        side = max(1, int(min(height, width) * 0.45 * strength))
        x = (width - side) // 2
        y = (height - side) // 2
        output[y : y + side, x : x + side] = 0
        return output
    raise ValueError(f"Unsupported degradation kind: {kind}")
