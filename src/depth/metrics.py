from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DepthMetrics:
    abs_rel: float
    rmse: float
    rmse_log: float
    delta1: float
    delta2: float
    delta3: float
    valid_pixels: int
    valid_fraction: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "abs_rel": self.abs_rel,
            "rmse": self.rmse,
            "rmse_log": self.rmse_log,
            "delta1": self.delta1,
            "delta2": self.delta2,
            "delta3": self.delta3,
            "valid_pixels": self.valid_pixels,
            "valid_fraction": self.valid_fraction,
        }


def align_depth_scale(
    prediction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray | None = None,
    min_depth: float = 1e-6,
) -> np.ndarray:
    pred = np.asarray(prediction, dtype=np.float32)
    gt = np.asarray(target, dtype=np.float32)
    valid = _build_valid_mask(pred, gt, mask, min_depth)
    if not np.any(valid):
        raise ValueError("No valid pixels available for depth scale alignment.")

    pred_median = float(np.median(pred[valid]))
    gt_median = float(np.median(gt[valid]))
    if abs(pred_median) < min_depth:
        raise ValueError("Prediction median is too small for stable depth alignment.")
    scale = gt_median / pred_median
    return pred * scale


def align_depth_scale_shift(
    prediction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray | None = None,
    min_depth: float = 1e-6,
) -> np.ndarray:
    pred = np.asarray(prediction, dtype=np.float32)
    gt = np.asarray(target, dtype=np.float32)
    valid = _build_valid_mask(pred, gt, mask, min_depth)
    if np.count_nonzero(valid) < 2:
        raise ValueError("At least two valid pixels are required for scale-and-shift alignment.")
    design = np.column_stack((pred[valid], np.ones(np.count_nonzero(valid), dtype=np.float32)))
    scale, shift = np.linalg.lstsq(design, gt[valid], rcond=None)[0]
    return pred * float(scale) + float(shift)


def compute_depth_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray | None = None,
    min_depth: float = 1e-6,
) -> DepthMetrics:
    pred = np.asarray(prediction, dtype=np.float32)
    gt = np.asarray(target, dtype=np.float32)
    valid = _build_valid_mask(pred, gt, mask, min_depth)
    if not np.any(valid):
        raise ValueError("No valid pixels available for metric computation.")

    pred_valid = pred[valid]
    gt_valid = gt[valid]
    abs_rel = float(np.mean(np.abs(pred_valid - gt_valid) / gt_valid))
    rmse = float(np.sqrt(np.mean((pred_valid - gt_valid) ** 2)))
    rmse_log = float(np.sqrt(np.mean((np.log(pred_valid) - np.log(gt_valid)) ** 2)))
    ratio = np.maximum(pred_valid / gt_valid, gt_valid / pred_valid)
    delta1 = float(np.mean(ratio < 1.25))
    delta2 = float(np.mean(ratio < 1.25**2))
    delta3 = float(np.mean(ratio < 1.25**3))
    return DepthMetrics(
        abs_rel=abs_rel,
        rmse=rmse,
        rmse_log=rmse_log,
        delta1=delta1,
        delta2=delta2,
        delta3=delta3,
        valid_pixels=int(pred_valid.size),
        valid_fraction=float(pred_valid.size / pred.size),
    )


def decode_tum_depth_png(depth_png: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth_png)
    if depth.dtype != np.uint16:
        raise ValueError(f"TUM depth PNG must be uint16, got {depth.dtype}.")
    return depth.astype(np.float32) / 5000.0


def _build_valid_mask(
    prediction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray | None,
    min_depth: float,
) -> np.ndarray:
    valid = np.isfinite(prediction) & np.isfinite(target) & (prediction > min_depth) & (target > min_depth)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)
    return valid
