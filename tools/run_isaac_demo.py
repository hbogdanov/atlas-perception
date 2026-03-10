from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.config import deep_merge_dicts, load_config, validate_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Atlas Perception against an Isaac Sim ROS2 camera feed.")
    parser.add_argument("--config", default="configs/default.yaml", help="Base Atlas config.")
    parser.add_argument("--override-config", default="configs/isaac_demo.yaml", help="Isaac override config.")
    parser.add_argument("--camera-topic", default="/isaac/camera/color", help="Isaac Sim RGB topic.")
    parser.add_argument(
        "--camera-info-topic",
        default="/isaac/camera/camera_info",
        help="Isaac Sim CameraInfo topic.",
    )
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N frames. Use 0 to run until stopped.")
    parser.add_argument(
        "--slam-mode",
        choices=("dummy", "rtabmap", "disabled"),
        default=None,
        help="Override the configured SLAM backend.",
    )
    return parser.parse_args()


def build_runtime_config(args: argparse.Namespace) -> dict:
    config = load_config(args.config, args.override_config)
    override = {
        "input": {
            "mode": "ros2",
            "source": str(args.camera_topic),
            "camera_info_topic": str(args.camera_info_topic),
        },
        "sim": {
            "platform": "isaac",
            "camera_topic": str(args.camera_topic),
            "camera_info_topic": str(args.camera_info_topic),
        },
    }
    if args.slam_mode is not None:
        override["slam"] = {"mode": str(args.slam_mode)}
    return validate_config(deep_merge_dicts(config, override))


def build_command(args: argparse.Namespace) -> list[str]:
    config = build_runtime_config(args)
    override_path = REPO_ROOT / ".tmp" / "isaac_runtime.yaml"
    override_path.parent.mkdir(parents=True, exist_ok=True)
    import yaml

    with override_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

    return [
        sys.executable,
        "-m",
        "src.main",
        "--config",
        str(REPO_ROOT / "configs" / "default.yaml"),
        "--override-config",
        str(override_path),
        "--max-frames",
        str(int(args.max_frames)),
    ]


def main() -> None:
    args = parse_args()
    command = build_command(args)
    subprocess.run(command, check=True, cwd=REPO_ROOT)


if __name__ == "__main__":
    main()
