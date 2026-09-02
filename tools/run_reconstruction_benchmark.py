from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

SCENARIOS = {
    "gt_depth_gt_pose": "configs/benchmarks/tum_gt_depth_gt_pose.yaml",
    "gt_depth_gt_pose_multiview": "configs/benchmarks/tum_gt_depth_gt_pose_multiview.yaml",
    "estimated_depth_gt_pose": "configs/benchmarks/tum_estimated_depth_gt_pose.yaml",
    "gt_depth_perturbed_pose": "configs/benchmarks/tum_gt_depth_perturbed_pose.yaml",
    "estimated_depth_perturbed_pose": "configs/benchmarks/tum_estimated_depth_perturbed_pose.yaml",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Atlas TUM reconstruction ablations with recorded manifests.")
    parser.add_argument("--scenario", choices=sorted(SCENARIOS), default=None)
    parser.add_argument("--all", action="store_true", help="Run every named reconstruction scenario.")
    parser.add_argument("--max-frames", type=int, default=120)
    return parser.parse_args()


def build_command(scenario: str, max_frames: int) -> list[str]:
    return [
        sys.executable,
        "-m",
        "src.main",
        "--config",
        "configs/default.yaml",
        "--override-config",
        SCENARIOS[scenario],
        "--max-frames",
        str(max_frames),
    ]


def main() -> None:
    args = parse_args()
    if not args.all and args.scenario is None:
        raise SystemExit("Choose --scenario or pass --all.")
    scenarios = list(SCENARIOS) if args.all else [args.scenario]
    for scenario in scenarios:
        command = build_command(scenario, args.max_frames)
        print(f"Running {scenario}: {' '.join(command)}")
        subprocess.run(command, check=True, cwd=REPO_ROOT)


if __name__ == "__main__":
    main()
