from __future__ import annotations

from src.sim.gazebo_bridge import GazeboBridge
from src.sim.isaac_bridge import IsaacBridge


def create_sim_bridge(config: dict | None):
    if not config:
        return None
    platform = str(config.get("platform", "")).lower()
    if platform == "isaac":
        return IsaacBridge(config)
    if platform == "gazebo":
        return GazeboBridge(config)
    raise ValueError(f"Unsupported simulator platform: {platform}")
