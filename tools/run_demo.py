from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.export_demo_gif import export_demo_gif

DEMO_PRESETS = {
    "tum": {
        "config": "configs/default.yaml",
        "override_config": "configs/tum_demo.yaml",
        "max_frames": 240,
        "video_path": "demo/videos/tum_demo.mp4",
        "gif_path": "demo/gifs/tum_demo.gif",
        "trajectory_plot": "data/outputs/tum_demo_eval/trajectory_plot.png",
        "trajectory_screenshot": "demo/screenshots/tum_trajectory_plot.png",
        "gif_fps": 6.0,
        "gif_max_frames": 160,
        "gif_width": 960,
        "gif_duration_ms": 100,
    }
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a one-command Atlas demo preset.")
    parser.add_argument("--dataset", choices=sorted(DEMO_PRESETS), default="tum", help="Named demo preset to run.")
    parser.add_argument("--max-frames", type=int, default=None, help="Override the preset frame count.")
    parser.add_argument("--skip-gif", action="store_true", help="Skip GIF generation after the demo run.")
    return parser.parse_args()


def build_demo_command(dataset: str, max_frames: int | None = None) -> list[str]:
    preset = DEMO_PRESETS[dataset]
    return [
        sys.executable,
        "-m",
        "src.main",
        "--config",
        preset["config"],
        "--override-config",
        preset["override_config"],
        "--max-frames",
        str(int(preset["max_frames"] if max_frames is None else max_frames)),
    ]


def run_demo(dataset: str, max_frames: int | None = None, skip_gif: bool = False) -> Path:
    command = build_demo_command(dataset, max_frames=max_frames)
    subprocess.run(command, check=True, cwd=REPO_ROOT)

    preset = DEMO_PRESETS[dataset]
    video_path = REPO_ROOT / preset["video_path"]
    trajectory_plot = REPO_ROOT / preset["trajectory_plot"]
    trajectory_screenshot = REPO_ROOT / preset["trajectory_screenshot"]
    if trajectory_plot.exists():
        trajectory_screenshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(trajectory_plot, trajectory_screenshot)
    if not skip_gif:
        export_demo_gif(
            video_path,
            REPO_ROOT / preset["gif_path"],
            fps=float(preset.get("gif_fps", 8.0)),
            max_frames=int(preset.get("gif_max_frames", 80)),
            width=int(preset.get("gif_width", 960)),
            duration_ms=int(preset["gif_duration_ms"]) if "gif_duration_ms" in preset else None,
        )
    return video_path


def main() -> None:
    args = parse_args()
    video_path = run_demo(args.dataset, max_frames=args.max_frames, skip_gif=args.skip_gif)
    print(f"Demo video: {video_path}")
    if not args.skip_gif:
        print(f"Demo GIF: {REPO_ROOT / DEMO_PRESETS[args.dataset]['gif_path']}")


if __name__ == "__main__":
    main()
