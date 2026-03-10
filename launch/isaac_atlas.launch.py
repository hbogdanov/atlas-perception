"""ROS2 launch description for Atlas Perception on Isaac Sim RGB topics."""

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
    override_config = str(Path("configs/isaac_demo.yaml"))
    if LaunchDescription is None:
        return {
            "command": [
                "python",
                "tools/run_isaac_demo.py",
                "--config",
                base_config,
                "--override-config",
                override_config,
                "--camera-topic",
                "/isaac/camera/color",
                "--camera-info-topic",
                "/isaac/camera/camera_info",
            ],
            "config": override_config,
        }

    camera_topic_arg = DeclareLaunchArgument("camera_topic", default_value="/isaac/camera/color")
    camera_info_topic_arg = DeclareLaunchArgument("camera_info_topic", default_value="/isaac/camera/camera_info")
    slam_mode_arg = DeclareLaunchArgument("slam_mode", default_value="dummy")
    max_frames_arg = DeclareLaunchArgument("max_frames", default_value="0")

    command = ExecuteProcess(
        cmd=[
            "python",
            "tools/run_isaac_demo.py",
            "--config",
            base_config,
            "--override-config",
            override_config,
            "--camera-topic",
            LaunchConfiguration("camera_topic"),
            "--camera-info-topic",
            LaunchConfiguration("camera_info_topic"),
            "--slam-mode",
            LaunchConfiguration("slam_mode"),
            "--max-frames",
            LaunchConfiguration("max_frames"),
        ],
        output="screen",
    )
    return LaunchDescription([camera_topic_arg, camera_info_topic_arg, slam_mode_arg, max_frames_arg, command])
