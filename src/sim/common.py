from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass


@dataclass
class SimulatorSpec:
    platform: str
    namespace: str
    camera_topic: str
    camera_info_topic: str
    clock_topic: str
    launch_package: str
    launch_file: str


class SimulatorBridge:
    platform = "generic"

    def __init__(self, config: dict) -> None:
        self.spec = SimulatorSpec(
            platform=str(config["platform"]),
            namespace=str(config.get("namespace", "")),
            camera_topic=str(config["camera_topic"]),
            camera_info_topic=str(config.get("camera_info_topic", "")),
            clock_topic=str(config.get("clock_topic", "/clock")),
            launch_package=str(config.get("launch_package", "")),
            launch_file=str(config.get("launch_file", "")),
        )

    def apply(self, config: dict) -> dict:
        merged = deepcopy(config)
        merged.setdefault("input", {})
        merged.setdefault("ros2", {})
        merged["input"]["mode"] = "ros2"
        merged["input"]["source"] = self.spec.camera_topic
        merged["input"]["camera_info_topic"] = self.spec.camera_info_topic
        merged["ros2"]["enabled"] = True
        return merged

    def topic_remaps(self) -> dict[str, str]:
        return {
            "image": self.spec.camera_topic,
            "camera_info": self.spec.camera_info_topic,
            "clock": self.spec.clock_topic,
        }

    def launch_arguments(self) -> dict[str, str]:
        return {
            "sim_platform": self.spec.platform,
            "sim_namespace": self.spec.namespace,
            "camera_topic": self.spec.camera_topic,
            "camera_info_topic": self.spec.camera_info_topic,
            "clock_topic": self.spec.clock_topic,
        }
