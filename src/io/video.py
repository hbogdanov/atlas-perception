from __future__ import annotations

from time import time

import cv2

from src.io.base import FrameSource
from src.io.types import FramePacket


class VideoFrameSource(FrameSource):
    def __init__(self, path: str) -> None:
        self._capture = cv2.VideoCapture(path)
        if not self._capture.isOpened():
            raise RuntimeError(f"Unable to open video source at {path}.")

    def frames(self):
        while True:
            ok, frame = self._capture.read()
            if not ok:
                break
            yield FramePacket(image=frame, timestamp=time())

    def close(self) -> None:
        self._capture.release()
