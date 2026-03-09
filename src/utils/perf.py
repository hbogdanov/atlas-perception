from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter


@dataclass
class TimerResult:
    seconds: float

    @property
    def milliseconds(self) -> float:
        return self.seconds * 1000.0


class Timer:
    def __enter__(self):
        self._start = perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.result = TimerResult(seconds=perf_counter() - self._start)
