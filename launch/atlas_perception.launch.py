"""ROS2 launch description for the Atlas Perception pipeline."""

from __future__ import annotations

from pathlib import Path

try:
    from launch.actions import DeclareLaunchArgument, ExecuteProcess
    from launch.substitutions import LaunchConfiguration

    from launch import LaunchDescription
except ImportError:  # pragma: no cover
    LaunchDescription = None
    DeclareLaunchArgument = None
    ExecuteProcess = None
    LaunchConfiguration = None


def generate_launch_description():
    base_config = str(Path("configs/default.yaml"))
    if LaunchDescription is None:
        return {
            "command": ["python", "-m", "src.main", "--config", base_config],
            "config": base_config,
        }

    config_arg = DeclareLaunchArgument("config", default_value=base_config)
    max_frames_arg = DeclareLaunchArgument("max_frames", default_value="0")

    command = ExecuteProcess(
        cmd=[
            "python",
            "-m",
            "src.main",
            "--config",
            LaunchConfiguration("config"),
            "--max-frames",
            LaunchConfiguration("max_frames"),
        ],
        output="screen",
    )
    return LaunchDescription([config_arg, max_frames_arg, command])
