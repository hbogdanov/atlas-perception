from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FramePacket:
    image: np.ndarray
    timestamp: float
