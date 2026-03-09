"""ROS2 launch description for simulator-backed Atlas Perception demos."""

from __future__ import annotations

from pathlib import Path

import yaml

try:
    from launch import LaunchDescription
    from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
    from launch.substitutions import LaunchConfiguration
except ImportError:  # pragma: no cover
    LaunchDescription = None
    DeclareLaunchArgument = None
    ExecuteProcess = None
    OpaqueFunction = None
    LaunchConfiguration = None

from src.sim.factory import create_sim_bridge


def _load_bridge(config_path: str):
    with Path(config_path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    return create_sim_bridge(config.get("sim"))


def _launch_setup(context):
    base_config = str(Path("configs/default.yaml"))
    override_config = LaunchConfiguration("sim_config").perform(context)
    bridge = _load_bridge(override_config)
    launch_arguments = bridge.launch_arguments() if bridge is not None else {}

    process = ExecuteProcess(
        cmd=[
            "python",
            "-m",
            "src.main",
            "--config",
            base_config,
            "--override-config",
            override_config,
            "--max-frames",
            LaunchConfiguration("max_frames").perform(context),
        ],
        output="screen",
        additional_env={f"ATLAS_{key.upper()}": value for key, value in launch_arguments.items()},
    )
    return [process]


def generate_launch_description():
    default_sim_config = str(Path("configs/isaac_demo.yaml"))
    if LaunchDescription is None:
        bridge = _load_bridge(default_sim_config)
        return {
            "command": [
                "python",
                "-m",
                "src.main",
                "--config",
                "configs/default.yaml",
                "--override-config",
                default_sim_config,
            ],
            "sim": bridge.launch_arguments() if bridge is not None else {},
        }

    sim_config_arg = DeclareLaunchArgument("sim_config", default_value=default_sim_config)
    max_frames_arg = DeclareLaunchArgument("max_frames", default_value="0")
    return LaunchDescription([sim_config_arg, max_frames_arg, OpaqueFunction(function=_launch_setup)])
