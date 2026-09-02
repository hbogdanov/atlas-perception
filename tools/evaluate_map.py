from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.mapping.metrics import compute_map_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an Atlas point cloud against a reference point cloud.")
    parser.add_argument("--estimated-ply", required=True)
    parser.add_argument("--reference-ply", required=True)
    parser.add_argument("--completeness-threshold", type=float, default=0.1)
    parser.add_argument(
        "--max-points", type=int, default=10000, help="Deterministic cap per cloud for bounded evaluation."
    )
    parser.add_argument("--output-json", default="data/outputs/map_eval/map_metrics.json")
    parser.add_argument("--output-csv", default="data/outputs/map_eval/map_metrics.csv")
    return parser.parse_args()


def read_ascii_ply(path: Path) -> np.ndarray:
    lines = path.read_text(encoding="ascii").splitlines()
    try:
        header_end = lines.index("end_header")
    except ValueError as exc:
        raise ValueError(f"Expected ASCII PLY header in {path}.") from exc
    rows = [line.split()[:3] for line in lines[header_end + 1 :] if line.strip()]
    return np.asarray(rows, dtype=np.float32)


def downsample_points(points: np.ndarray, max_points: int) -> np.ndarray:
    if max_points <= 0 or points.shape[0] <= max_points:
        return points
    indices = np.linspace(0, points.shape[0] - 1, num=max_points, dtype=np.int64)
    return points[indices]


def main() -> None:
    args = parse_args()
    metrics = compute_map_metrics(
        downsample_points(read_ascii_ply(Path(args.estimated_ply)), args.max_points),
        downsample_points(read_ascii_ply(Path(args.reference_ply)), args.max_points),
        completeness_threshold=args.completeness_threshold,
    ).to_dict()
    metrics["estimated_ply"] = args.estimated_ply
    metrics["reference_ply"] = args.reference_ply
    metrics["completeness_threshold"] = args.completeness_threshold
    output_json = Path(args.output_json)
    output_csv = Path(args.output_csv)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics))
        writer.writeheader()
        writer.writerow(metrics)
    print(f"Chamfer={metrics['chamfer_distance']:.4f} completeness={metrics['completeness']:.4f}")


if __name__ == "__main__":
    main()
