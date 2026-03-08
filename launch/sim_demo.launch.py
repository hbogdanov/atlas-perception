"""Placeholder ROS2 launch description for simulator demos."""


def generate_launch_description():
    return {
        "node": "atlas_perception",
        "config": "configs/isaac_demo.yaml",
    }

