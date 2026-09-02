from __future__ import annotations

import cv2

from src.io.base import FrameSource
from src.io.types import FramePacket


class VideoFrameSource(FrameSource):
    def __init__(self, path: str, loop: bool = False) -> None:
        self._loop = bool(loop)
        self._capture = cv2.VideoCapture(path)
        if not self._capture.isOpened():
            raise RuntimeError(f"Unable to open video source at {path}.")

    def frames(self):
        while True:
            ok, frame = self._capture.read()
            if not ok:
                if self._loop:
                    self._capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break
            timestamp = float(self._capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
            if timestamp <= 0.0:
                fps = float(self._capture.get(cv2.CAP_PROP_FPS))
                frame_index = float(self._capture.get(cv2.CAP_PROP_POS_FRAMES))
                timestamp = frame_index / fps if fps > 0.0 else frame_index
            yield FramePacket(image=frame, timestamp=timestamp)

    def close(self) -> None:
        self._capture.release()
