from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.depth.estimator import DepthEstimator
from src.depth.metrics import align_depth_scale, compute_depth_metrics, decode_tum_depth_png
from src.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quantitatively evaluate monocular depth on a TUM RGB-D sequence.")
    parser.add_argument(
        "--dataset-root", required=True, help="Path to the TUM sequence root containing rgb/ and depth/."
    )
    parser.add_argument("--config", default="configs/default.yaml", help="Base config path.")
    parser.add_argument("--override-config", default=None, help="Optional override config path.")
    parser.add_argument("--limit", type=int, default=0, help="Maximum number of RGB/depth pairs to evaluate.")
    parser.add_argument(
        "--output-json", default="data/outputs/depth_eval/tum_depth_eval.json", help="Summary JSON path."
    )
    parser.add_argument("--output-csv", default="data/outputs/depth_eval/tum_depth_eval.csv", help="Per-run CSV path.")
    return parser.parse_args()


def list_tum_pairs(dataset_root: Path) -> list[tuple[Path, Path]]:
    rgb_dir = dataset_root / "rgb"
    depth_dir = dataset_root / "depth"
    if not rgb_dir.is_dir() or not depth_dir.is_dir():
        raise RuntimeError(f"Expected TUM dataset with rgb/ and depth/ under {dataset_root}.")

    depth_by_stem = {path.stem: path for path in depth_dir.glob("*.png")}
    pairs: list[tuple[Path, Path]] = []
    for rgb_path in sorted(rgb_dir.glob("*.png")):
        depth_path = depth_by_stem.get(rgb_path.stem)
        if depth_path is not None:
            pairs.append((rgb_path, depth_path))
    if not pairs:
        raise RuntimeError(f"No matching TUM RGB/depth PNG pairs found under {dataset_root}.")
    return pairs


def evaluate_tum_depth(
    dataset_root: Path,
    estimator: DepthEstimator,
    limit: int = 0,
) -> dict[str, object]:
    pairs = list_tum_pairs(dataset_root)
    if limit > 0:
        pairs = pairs[:limit]

    frame_metrics = []
    inference_times_ms: list[float] = []
    for rgb_path, depth_path in pairs:
        rgb = cv2.imread(str(rgb_path))
        raw_depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if rgb is None:
            raise RuntimeError(f"Failed to read RGB frame: {rgb_path}")
        if raw_depth is None:
            raise RuntimeError(f"Failed to read depth frame: {depth_path}")

        gt_depth = decode_tum_depth_png(raw_depth)
        start = perf_counter()
        pred_depth = estimator.predict(rgb)
        inference_times_ms.append((perf_counter() - start) * 1000.0)

        valid = gt_depth > 0.0
        aligned_depth = align_depth_scale(pred_depth, gt_depth, mask=valid)
        metrics = compute_depth_metrics(aligned_depth, gt_depth, mask=valid)
        frame_metrics.append(metrics)

    abs_rel = float(np.mean([metric.abs_rel for metric in frame_metrics]))
    rmse = float(np.mean([metric.rmse for metric in frame_metrics]))
    delta1 = float(np.mean([metric.delta1 for metric in frame_metrics]))
    avg_inference_ms = float(np.mean(inference_times_ms))
    fps = float(1000.0 / avg_inference_ms) if avg_inference_ms > 0.0 else 0.0

    return {
        "dataset": "tum_rgbd",
        "dataset_root": str(dataset_root),
        "frames_evaluated": len(frame_metrics),
        "abs_rel": abs_rel,
        "rmse": rmse,
        "delta1": delta1,
        "avg_inference_ms": avg_inference_ms,
        "fps": fps,
        "model": estimator.model_name,
        "output_mode": estimator.output_mode,
        "postprocess_enabled": bool(estimator.postprocess_config.get("enabled", False)),
    }


def write_outputs(summary: dict[str, object], output_json: Path, output_csv: Path) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "model",
                "dataset",
                "frames_evaluated",
                "abs_rel",
                "rmse",
                "delta1",
                "fps",
                "avg_inference_ms",
                "output_mode",
                "postprocess_enabled",
                "dataset_root",
            ],
        )
        writer.writeheader()
        writer.writerow(summary)


def main() -> None:
    args = parse_args()
    config = load_config(args.config, args.override_config)
    estimator = DepthEstimator(config["depth"])
    summary = evaluate_tum_depth(Path(args.dataset_root), estimator, limit=args.limit)
    write_outputs(summary, Path(args.output_json), Path(args.output_csv))

    print(
        (
            "Model={model} AbsRel={abs_rel:.4f} RMSE={rmse:.4f} "
            "delta1={delta1:.4f} FPS={fps:.2f} Frames={frames_evaluated}"
        ).format(**summary)
    )
    print(f"JSON: {args.output_json}")
    print(f"CSV: {args.output_csv}")


if __name__ == "__main__":
    main()
