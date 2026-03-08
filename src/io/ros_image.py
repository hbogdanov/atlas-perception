from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RosImageMessage:
    topic: str
    image: np.ndarray
    timestamp: float


class RosImageSubscriber:
    """Thin placeholder for a ROS2 image subscription interface."""

    def __init__(self, topic: str) -> None:
        self.topic = topic

    def latest(self) -> RosImageMessage | None:
        return None

