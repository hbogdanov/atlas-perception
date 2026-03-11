from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_DATASETS = [
    "data/raw/tum/fr1_desk",
    "data/raw/tum/fr1_room",
    "data/raw/tum/fr3_long_office_household",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate temporary README showcase assets for TUM RGB-D datasets.")
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=DEFAULT_DATASETS,
        help="Top-level dataset directories to process.",
    )
    parser.add_argument("--demo-frames", type=int, default=180, help="Frames to run through Atlas for each showcase.")
    return parser.parse_args()


def resolve_dataset_root(path: Path) -> Path:
    nested = path / path.name
    if nested.exists():
        return nested
    return path


def dataset_slug(path: Path) -> str:
    name = path.name
    tum_aliases = {
        "fr1_desk": "freiburg1_desk",
        "fr1_room": "freiburg1_room",
        "fr3_long_office_household": "freiburg3_long_office_household",
        "fr1_desk2": "freiburg1_desk2",
        "fr2_desk": "freiburg2_desk",
    }
    if name in tum_aliases:
        return tum_aliases[name]
    if name.startswith("rgbd_dataset_freiburg"):
        return name.replace("rgbd_dataset_freiburg", "freiburg")
    return name


def write_override_config(path: Path, dataset_root: Path, output_dir: Path, demo_video_path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "input:",
                "  mode: rgbd_dataset",
                f"  source: {dataset_root.as_posix()}",
                "",
                "semantics:",
                "  enabled: true",
                "  backend: yolov8_seg",
                "  model: models/checkpoints/yolov8n-seg.pt",
                "  device: cpu",
                "  confidence: 0.2",
                "  iou: 0.6",
                "",
                "depth:",
                "  source_mode: input",
                "  output_mode: raw",
                "  viz_min_depth: 0.4",
                "  viz_max_depth: 5.5",
                "  viz_fill_invalid: true",
                "  viz_smooth_ksize: 5",
                "  postprocess:",
                "    enabled: false",
                "",
                "slam:",
                "  mode: groundtruth",
                "",
                "mapping:",
                "  representation: pointcloud",
                "  stride: 2",
                "  max_points: 150000",
                "",
                "ros2:",
                "  enabled: false",
                "",
                "output:",
                "  save_rgb_snapshot: true",
                "  save_depth_snapshot: true",
                "  save_demo_video: true",
                "  save_pointcloud: true",
                "  save_trajectory: true",
                f"  output_dir: {output_dir.as_posix()}",
                f"  demo_video_path: {demo_video_path.as_posix()}",
                "  demo_map_projection: xz",
                "  demo_map_bounds: null",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path


def generate_showcase(dataset_name: str, demo_frames: int) -> dict[str, str]:
    from tools.export_demo_gif import export_demo_gif

    dataset_path = Path(dataset_name)
    top_level = dataset_path if dataset_path.is_absolute() else REPO_ROOT / dataset_path
    dataset_root = resolve_dataset_root(top_level)
    slug = dataset_slug(top_level)
    output_dir = REPO_ROOT / "data" / "outputs" / "dataset_showcase" / slug / "eval"
    demo_video = REPO_ROOT / "demo" / "videos" / f"{slug}_demo.mp4"
    demo_gif = REPO_ROOT / "demo" / "gifs" / f"{slug}_demo.gif"
    trajectory_plot = output_dir / "trajectory_plot.png"
    trajectory_screenshot = REPO_ROOT / "demo" / "screenshots" / f"{slug}_trajectory_plot.png"
    override_config = REPO_ROOT / "configs" / "generated" / f"{slug}_showcase.yaml"

    write_override_config(override_config, dataset_root, output_dir, demo_video)

    command = [
        sys.executable,
        "-m",
        "src.main",
        "--config",
        "configs/default.yaml",
        "--override-config",
        str(override_config),
        "--max-frames",
        str(int(demo_frames)),
    ]
    subprocess.run(command, check=True, cwd=REPO_ROOT)

    if trajectory_plot.exists():
        trajectory_screenshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(trajectory_plot, trajectory_screenshot)

    export_demo_gif(
        demo_video,
        demo_gif,
        fps=6.0,
        max_frames=120,
        width=960,
        duration_ms=100,
    )

    return {
        "dataset_name": dataset_name,
        "slug": slug,
        "gif_path": demo_gif.relative_to(REPO_ROOT).as_posix(),
    }


def main() -> None:
    args = parse_args()
    generated = []
    for dataset_name in args.datasets:
        generated.append(generate_showcase(dataset_name, demo_frames=args.demo_frames))
    for item in generated:
        print(f"{item['dataset_name']}: {item['gif_path']}")


if __name__ == "__main__":
    main()
