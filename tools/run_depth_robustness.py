from __future__ import annotations

import argparse
import csv
import json
import sys
from copy import deepcopy
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.depth.degradation import degrade_image
from src.utils.config import load_config
from src.utils.run_metadata import build_run_manifest, write_json
from tools.evaluate_depth import evaluate_tum_depth


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run deterministic monocular-depth robustness experiments on TUM RGB-D."
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--override-config", default=None)
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--alignment", choices=("raw", "median_scale", "scale_shift"), default="median_scale")
    parser.add_argument("--degradations", default="brightness,gaussian_noise,motion_blur,resolution,occlusion")
    parser.add_argument("--severities", default="0.33,0.66,1.0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-json", default="data/outputs/robustness/depth_robustness.json")
    parser.add_argument("--output-csv", default="data/outputs/robustness/depth_robustness.csv")
    return parser.parse_args()


def parse_csv_values(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> None:
    args = parse_args()
    from src.depth.estimator import DepthEstimator

    config = load_config(args.config, args.override_config)
    estimator = DepthEstimator(config["depth"])
    rows: list[dict[str, object]] = []
    baseline = evaluate_tum_depth(Path(args.dataset_root), estimator, limit=args.limit, alignment=args.alignment)
    rows.append({"degradation": "clean", "severity": 0.0, **baseline})
    for degradation in parse_csv_values(args.degradations):
        for severity_text in parse_csv_values(args.severities):
            severity = float(severity_text)
            summary = evaluate_tum_depth(
                Path(args.dataset_root),
                estimator,
                limit=args.limit,
                alignment=args.alignment,
                image_transform=lambda image, frame_index, kind=degradation, amount=severity: degrade_image(
                    image, kind, amount, seed=args.seed + frame_index
                ),
            )
            rows.append({"degradation": degradation, "severity": severity, **summary})
            print(f"{degradation} severity={severity:.2f} AbsRel={summary['abs_rel']:.4f} RMSE={summary['rmse']:.4f}")

    output_json = Path(args.output_json)
    output_csv = Path(args.output_csv)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    manifest_config = deepcopy(config)
    manifest_config["input"] = {"mode": "rgbd_dataset", "source": args.dataset_root}
    write_json(
        output_json.with_name("manifest.json"),
        {
            **build_run_manifest(manifest_config, REPO_ROOT),
            "experiment": {
                "kind": "depth_robustness",
                "dataset_root": args.dataset_root,
                "limit": args.limit,
                "alignment": args.alignment,
                "seed": args.seed,
                "degradations": parse_csv_values(args.degradations),
                "severities": [float(value) for value in parse_csv_values(args.severities)],
                "baseline": "clean",
            },
        },
    )
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["degradation", "severity"])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
