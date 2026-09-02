from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import matplotlib
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.mapping.metrics import compute_map_metrics
from tools.compile_flagship_study import downsample, read_ascii_ply


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run controlled Atlas calibration or pose-latency sensitivity sweeps.")
    parser.add_argument("--kind", choices=("calibration", "latency"), required=True)
    parser.add_argument("--values", default=None, help="Comma-separated focal-scale deltas or integer latency frames.")
    parser.add_argument("--max-frames", type=int, default=60)
    parser.add_argument("--output-dir", default="data/outputs/sensitivity")
    return parser.parse_args()


def parse_values(kind: str, raw: str | None) -> list[float]:
    default = "0,0.01,0.03,0.05" if kind == "calibration" else "0,1,2,4"
    values = [float(value) for value in (raw or default).split(",")]
    if kind == "latency" and any(value < 0 or not value.is_integer() for value in values):
        raise ValueError("Latency values must be non-negative integer frame counts.")
    return values


def build_override(kind: str, value: float, output_dir: Path) -> dict:
    evaluation = {
        "calibration_perturbation": {"enabled": False},
        "pose_perturbation": {"enabled": False},
    }
    if kind == "calibration":
        evaluation["calibration_perturbation"] = {
            "enabled": value != 0.0,
            "fx_scale": 1.0 + value,
            "fy_scale": 1.0 + value,
            "cx_offset_px": 0.0,
            "cy_offset_px": 0.0,
        }
    else:
        evaluation["pose_perturbation"] = {
            "enabled": value != 0.0,
            "translation_std_m": 0.0,
            "rotation_std_deg": 0.0,
            "dropout_probability": 0.0,
            "latency_frames": int(value),
            "seed": 0,
        }
    return {
        "input": {"mode": "rgbd_dataset", "source": "data/samples/tum_freiburg1_xyz"},
        "depth": {"source_mode": "input", "output_mode": "raw", "postprocess": {"enabled": False}},
        "slam": {"mode": "groundtruth"},
        "semantics": {"enabled": False},
        "ros2": {"enabled": False},
        "evaluation": evaluation,
        "output": {
            "output_dir": str(output_dir),
            "save_rgb_snapshot": False,
            "save_depth_snapshot": False,
            "save_pointcloud": True,
            "save_trajectory": True,
        },
    }


def run_study(kind: str, values: list[float], max_frames: int, output_dir: Path) -> list[dict]:
    configs_dir = output_dir / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    run_dirs: list[tuple[float, Path]] = []
    for value in values:
        label = f"{kind}_{value:+.3f}".replace("+", "plus_").replace("-", "minus_").replace(".", "p")
        run_dir = output_dir / "runs" / label
        config_path = configs_dir / f"{label}.yaml"
        config_path.write_text(yaml.safe_dump(build_override(kind, value, run_dir), sort_keys=False), encoding="utf-8")
        subprocess.run(
            [
                sys.executable,
                "-m",
                "src.main",
                "--config",
                "configs/default.yaml",
                "--override-config",
                str(config_path),
                "--max-frames",
                str(max_frames),
            ],
            cwd=REPO_ROOT,
            check=True,
        )
        run_dirs.append((value, run_dir))
    reference_dir = next(directory for value, directory in run_dirs if value == 0.0)
    reference = downsample(read_ascii_ply(reference_dir / "frame_cloud.ply"), 10000)
    rows = []
    for value, run_dir in run_dirs:
        runtime = json.loads((run_dir / "runtime_metrics.json").read_text(encoding="utf-8"))
        metrics = compute_map_metrics(
            downsample(read_ascii_ply(run_dir / "frame_cloud.ply"), 10000), reference
        ).to_dict()
        rows.append({"kind": kind, "value": value, **metrics, "avg_fps": runtime["avg_fps"], "frames": max_frames})
    _write_results(rows, output_dir, kind)
    return rows


def _write_results(rows: list[dict], output_dir: Path, kind: str) -> None:
    (output_dir / f"{kind}_sensitivity.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    with (output_dir / f"{kind}_sensitivity.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.plot([row["value"] for row in rows], [row["chamfer_distance"] for row in rows], marker="o", color="#d56a3a")
    axis.set_xlabel("Focal-scale error" if kind == "calibration" else "Pose latency (frames)")
    axis.set_ylabel("Sampled Chamfer to unperturbed map (m)")
    axis.set_title(f"Atlas {kind.title()} Sensitivity")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / f"{kind}_sensitivity.png", dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    rows = run_study(args.kind, parse_values(args.kind, args.values), args.max_frames, Path(args.output_dir))
    print(f"Completed {args.kind} sensitivity study with {len(rows)} conditions.")


if __name__ == "__main__":
    main()
