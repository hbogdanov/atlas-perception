from __future__ import annotations

from time import time
from typing import Generator, Iterable

import cv2
import numpy as np

from src.io.types import FramePacket
from src.io.video import VideoFrameSource


class CameraFrameSource:
    def __init__(self, source: int, width: int, height: int) -> None:
        self._capture = cv2.VideoCapture(source)
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def frames(self) -> Generator[FramePacket, None, None]:
        while True:
            ok, frame = self._capture.read()
            if not ok:
                break
            yield FramePacket(image=frame, timestamp=time())

    def close(self) -> None:
        self._capture.release()


class StaticFrameSource:
    def __init__(self, frames: Iterable[np.ndarray]) -> None:
        self._frames = frames

    def frames(self) -> Generator[FramePacket, None, None]:
        for frame in self._frames:
            yield FramePacket(image=frame, timestamp=time())

    def close(self) -> None:
        return None


def create_frame_source(config: dict):
    mode = config["mode"]
    if mode == "webcam":
        return CameraFrameSource(
            source=int(config["source"]),
            width=int(config["width"]),
            height=int(config["height"]),
        )
    if mode == "video":
        return VideoFrameSource(str(config["source"]))
    if mode == "ros2":
        # Placeholder until a live ROS2 subscriber is connected.
        blank = np.zeros((config["height"], config["width"], 3), dtype=np.uint8)
        return StaticFrameSource([blank])
    raise ValueError(f"Unsupported input mode: {mode}")
