from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.slam.metrics import compute_trajectory_metrics, load_atlas_trajectory_json, load_tum_groundtruth


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an Atlas trajectory against TUM-format ground truth.")
    parser.add_argument("--estimated-json", required=True, help="Path to Atlas trajectory.json export.")
    parser.add_argument("--groundtruth-tum", required=True, help="Path to TUM-format groundtruth text file.")
    parser.add_argument("--max-timestamp-diff", type=float, default=0.02, help="Maximum timestamp association delta in seconds.")
    parser.add_argument(
        "--output-json",
        default="data/outputs/trajectory_eval/trajectory_eval.json",
        help="Summary JSON path.",
    )
    parser.add_argument(
        "--output-csv",
        default="data/outputs/trajectory_eval/trajectory_eval.csv",
        help="Summary CSV path.",
    )
    return parser.parse_args()


def write_outputs(summary: dict[str, object], output_json: Path, output_csv: Path) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "ate_rmse",
                "rpe_trans_rmse",
                "rpe_rot_deg_rmse",
                "matched_poses",
                "matched_pairs",
                "estimated_json",
                "groundtruth_tum",
                "max_timestamp_diff",
            ],
        )
        writer.writeheader()
        writer.writerow(summary)


def main() -> None:
    args = parse_args()
    estimated = load_atlas_trajectory_json(args.estimated_json)
    groundtruth = load_tum_groundtruth(args.groundtruth_tum)
    metrics = compute_trajectory_metrics(estimated, groundtruth, max_timestamp_diff=args.max_timestamp_diff)
    summary = {
        **metrics.to_dict(),
        "estimated_json": args.estimated_json,
        "groundtruth_tum": args.groundtruth_tum,
        "max_timestamp_diff": args.max_timestamp_diff,
    }
    write_outputs(summary, Path(args.output_json), Path(args.output_csv))
    print(
        "ATE_RMSE={ate_rmse:.4f} RPE_trans_RMSE={rpe_trans_rmse:.4f} "
        "RPE_rot_deg_RMSE={rpe_rot_deg_rmse:.4f} MatchedPoses={matched_poses}".format(**summary)
    )
    print(f"JSON: {args.output_json}")
    print(f"CSV: {args.output_csv}")


if __name__ == "__main__":
    main()
