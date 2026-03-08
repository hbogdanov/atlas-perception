from __future__ import annotations

from time import time

import cv2

from src.io.types import FramePacket


class VideoFrameSource:
    def __init__(self, path: str) -> None:
        self._capture = cv2.VideoCapture(path)

    def frames(self):
        while True:
            ok, frame = self._capture.read()
            if not ok:
                break
            yield FramePacket(image=frame, timestamp=time())

    def close(self) -> None:
        self._capture.release()

