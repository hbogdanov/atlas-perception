from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.mapping.metrics import compute_map_metrics

DEFAULT_SCENARIOS = (
    "tum_gt_depth_gt_pose",
    "tum_gt_depth_gt_pose_multiview",
    "tum_gt_depth_perturbed_pose",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compile Atlas reconstruction artifacts into a technical-study report."
    )
    parser.add_argument("--run-root", default="data/outputs/benchmarks")
    parser.add_argument("--output-dir", default="data/outputs/flagship_study")
    parser.add_argument("--reference-scenario", default="tum_gt_depth_gt_pose")
    parser.add_argument("--scenarios", nargs="+", default=list(DEFAULT_SCENARIOS))
    parser.add_argument("--max-points", type=int, default=10000)
    return parser.parse_args()


def read_ascii_ply(path: Path) -> np.ndarray:
    lines = path.read_text(encoding="ascii").splitlines()
    try:
        header_end = lines.index("end_header")
    except ValueError as exc:
        raise ValueError(f"Expected an ASCII PLY file: {path}") from exc
    return np.asarray([line.split()[:3] for line in lines[header_end + 1 :] if line.strip()], dtype=np.float32)


def downsample(points: np.ndarray, max_points: int) -> np.ndarray:
    if max_points <= 0 or len(points) <= max_points:
        return points
    return points[np.linspace(0, len(points) - 1, max_points, dtype=np.int64)]


def compile_study(
    run_root: Path, output_dir: Path, scenarios: list[str], reference_scenario: str, max_points: int
) -> list[dict]:
    reference_path = run_root / reference_scenario / "frame_cloud.ply"
    reference_points = downsample(read_ascii_ply(reference_path), max_points)
    rows = []
    for scenario in scenarios:
        run_dir = run_root / scenario
        cloud_path = run_dir / "frame_cloud.ply"
        metrics_path = run_dir / "runtime_metrics.json"
        if not cloud_path.exists() or not metrics_path.exists():
            raise FileNotFoundError(f"Missing complete artifacts for scenario '{scenario}' in {run_dir}.")
        runtime = json.loads(metrics_path.read_text(encoding="utf-8"))
        map_metrics = compute_map_metrics(
            downsample(read_ascii_ply(cloud_path), max_points), reference_points
        ).to_dict()
        manifest_path = run_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
        rows.append(
            {
                "scenario": scenario,
                "pose_source": manifest.get("slam", {}).get("mode", "unknown"),
                "depth_source": manifest.get("depth", {}).get("source_mode", "unknown"),
                "chamfer_distance_m": map_metrics["chamfer_distance"],
                "completeness_at_0_1m": map_metrics["completeness"],
                "avg_fps": runtime.get("avg_fps"),
                "avg_mapping_ms": runtime.get("avg_mapping_ms"),
                "point_count": runtime.get("point_count"),
                "visual_pose_measurements": runtime.get("visual_pose_measurements", 0),
                "provenance": _provenance_label(manifest),
            }
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_tables(rows, output_dir)
    _write_figures(rows, output_dir)
    _write_report(rows, output_dir, reference_scenario)
    return rows


def _provenance_label(manifest: dict) -> str:
    pose_mode = manifest.get("slam", {}).get("mode", "unknown")
    depth_mode = manifest.get("depth", {}).get("source_mode", "unknown")
    if pose_mode == "groundtruth" and depth_mode == "input":
        return "dataset ground-truth pose and depth"
    if pose_mode == "rtabmap":
        return "external RTAB-Map pose"
    return f"pose={pose_mode}; depth={depth_mode}"


def _write_tables(rows: list[dict], output_dir: Path) -> None:
    (output_dir / "headline_results.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    with (output_dir / "headline_results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_figures(rows: list[dict], output_dir: Path) -> None:
    labels = [_display_label(row["scenario"]) for row in rows]
    chamfer = [row["chamfer_distance_m"] for row in rows]
    fps = [row["avg_fps"] for row in rows]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].bar(labels, chamfer, color="#d56a3a")
    axes[0].set_ylabel("Sampled Chamfer distance to GT-pose baseline (m)")
    axes[0].tick_params(axis="x", rotation=20)
    axes[1].bar(labels, fps, color="#2f7d73")
    axes[1].set_ylabel("End-to-end FPS")
    axes[1].tick_params(axis="x", rotation=20)
    figure.suptitle("Atlas Reconstruction Study")
    figure.tight_layout()
    figure.savefig(output_dir / "map_quality_runtime.png", dpi=180)
    plt.close(figure)


def _display_label(scenario: str) -> str:
    labels = {
        "tum_gt_depth_gt_pose": "GT depth +\nGT pose",
        "tum_gt_depth_gt_pose_multiview": "GT depth + GT pose\n+multi-view fusion",
        "tum_gt_depth_perturbed_pose": "GT depth +\nperturbed pose",
        "tum_estimated_depth_gt_pose": "Estimated depth +\nGT pose",
        "tum_estimated_depth_perturbed_pose": "Estimated depth +\nperturbed pose",
    }
    return labels.get(scenario, scenario.replace("_", " "))


def _write_report(rows: list[dict], output_dir: Path, reference_scenario: str) -> None:
    lines = [
        "# Atlas Perception: Reconstruction Study",
        "",
        "## Scope",
        "",
        "This report evaluates map sensitivity under fixed dataset, intrinsics, frame count, and point-cloud settings. "
        f"All map distances are sampled comparisons to `{reference_scenario}`, not distances to a dense surface "
        "ground truth.",
        "",
        "## Results",
        "",
        "| Scenario | Pose source | Depth source | Chamfer (m) | Completeness @ 0.1 m | FPS | Provenance |",
        "| --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['scenario']} | {row['pose_source']} | {row['depth_source']} | "
            f"{row['chamfer_distance_m']:.4f} | {row['completeness_at_0_1m']:.3f} | "
            f"{row['avg_fps']:.2f} | {row['provenance']} |"
        )
    lines.extend(
        [
            "",
            "## Failure Taxonomy",
            "",
            "- Reflective or texture-poor surfaces: unreliable monocular or RGB-D depth support.",
            "- Motion blur and sparse views: fewer stable landmark corners and weaker multi-view agreement.",
            "- Calibration or timing error: coherent pose-to-depth misalignment that produces duplicated surfaces.",
            "- Occlusions and limited baseline: incomplete geometry and conservative confidence rejection.",
            "- Delayed or bad pose updates: map tearing; Atlas gates visual corrections by timestamp, innovation, and "
            "covariance.",
            "",
            "## Interpretation Boundary",
            "",
            "Ground-truth-pose rows are a reconstruction ceiling, not an autonomous SLAM claim. Visual landmark poses "
            "are owned PnP localization measurements; they can correct a supplied or externally tracked pose but do "
            "not provide continuous visual odometry, loop closure optimization, or relocalization in uninstrumented "
            "scenes.",
        ]
    )
    (output_dir / "technical_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = compile_study(
        Path(args.run_root), Path(args.output_dir), list(args.scenarios), args.reference_scenario, args.max_points
    )
    print(f"Compiled {len(rows)} scenarios into {args.output_dir}")


if __name__ == "__main__":
    main()
