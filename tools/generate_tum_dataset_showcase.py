from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.export_demo_gif import export_demo_gif


DEFAULT_DATASETS = [
    "rgbd_dataset_freiburg1_desk",
    "rgbd_dataset_freiburg1_desk2",
    "rgbd_dataset_freiburg1_room",
    "rgbd_dataset_freiburg2_desk",
    "rgbd_dataset_freiburg3_long_office_household",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate temporary README showcase assets for TUM RGB-D datasets.")
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=DEFAULT_DATASETS,
        help="Top-level dataset directories to process.",
    )
    parser.add_argument("--source-frames", type=int, default=120, help="RGB frames to pack into the source MP4.")
    parser.add_argument("--demo-frames", type=int, default=180, help="Frames to run through Atlas for each showcase.")
    return parser.parse_args()


def resolve_dataset_root(path: Path) -> Path:
    nested = path / path.name
    if nested.exists():
        return nested
    return path


def build_rgb_demo_video(dataset_root: Path, out_path: Path, max_frames: int) -> Path:
    rgb_dir = dataset_root / "rgb"
    frames = sorted(rgb_dir.glob("*.png"))[: max(1, int(max_frames))]
    if not frames:
        raise RuntimeError(f"No RGB frames found under {rgb_dir}")

    first = cv2.imread(str(frames[0]))
    if first is None:
        raise RuntimeError(f"Failed to read RGB frame {frames[0]}")
    height, width = first.shape[:2]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), 15.0, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to create video writer at {out_path}")
    try:
        for frame_path in frames:
            frame = cv2.imread(str(frame_path))
            if frame is None:
                raise RuntimeError(f"Failed to read RGB frame {frame_path}")
            writer.write(frame)
    finally:
        writer.release()
    return out_path


def write_override_config(path: Path, source_video: Path, output_dir: Path, demo_video_path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "input:",
                "  mode: video",
                f"  source: {source_video.as_posix()}",
                "  loop: true",
                "",
                "semantics:",
                "  enabled: true",
                "  backend: yolov8_seg",
                "  model: models/checkpoints/yolov8n-seg.pt",
                "  device: cpu",
                "  confidence: 0.2",
                "  iou: 0.6",
                "",
                "slam:",
                "  mode: dummy",
                "  path_radius_x: 1.6",
                "  path_radius_y: 0.9",
                "  path_frequency: 0.035",
                "  vertical_amplitude: 0.015",
                "  vertical_frequency: 0.06",
                "  heading_lookahead: 0.15",
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
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path


def generate_showcase(dataset_name: str, source_frames: int, demo_frames: int) -> dict[str, str]:
    top_level = REPO_ROOT / dataset_name
    dataset_root = resolve_dataset_root(top_level)
    slug = dataset_name.replace("rgbd_dataset_", "")
    source_video = REPO_ROOT / "data" / "outputs" / "dataset_showcase" / slug / f"{slug}_rgb_demo.mp4"
    output_dir = REPO_ROOT / "data" / "outputs" / "dataset_showcase" / slug / "eval"
    demo_video = REPO_ROOT / "demo" / "videos" / f"{slug}_demo.mp4"
    demo_gif = REPO_ROOT / "demo" / "gifs" / f"{slug}_demo.gif"
    trajectory_plot = output_dir / "trajectory_plot.png"
    trajectory_screenshot = REPO_ROOT / "demo" / "screenshots" / f"{slug}_trajectory_plot.png"
    override_config = REPO_ROOT / "configs" / "generated" / f"{slug}_showcase.yaml"

    build_rgb_demo_video(dataset_root, source_video, max_frames=source_frames)
    write_override_config(override_config, source_video, output_dir, demo_video)

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
        generated.append(generate_showcase(dataset_name, source_frames=args.source_frames, demo_frames=args.demo_frames))
    for item in generated:
        print(f"{item['dataset_name']}: {item['gif_path']}")


if __name__ == "__main__":
    main()
