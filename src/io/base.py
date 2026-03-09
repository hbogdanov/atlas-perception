from __future__ import annotations

from typing import Generator, Protocol, runtime_checkable

from src.io.types import FramePacket


@runtime_checkable
class FrameSource(Protocol):
    def frames(self) -> Generator[FramePacket, None, None]:
        ...

    def close(self) -> None:
        ...
