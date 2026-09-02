from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, TypeVar

import cv2
import numpy as np

from src.depth.metrics import decode_tum_depth_png
from src.io.base import FrameSource
from src.io.types import FramePacket
from src.ros2.transforms import quaternion_to_rotation_matrix

T = TypeVar("T")


@dataclass(frozen=True)
class AssociationReport:
    source_count: int
    target_count: int
    matched_pairs: int
    unmatched_source: int
    unmatched_target: int
    mean_timestamp_error: float | None
    max_timestamp_error: float | None


def _read_tum_index(path: Path) -> list[tuple[float, str]]:
    entries: list[tuple[float, str]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        entries.append((float(parts[0]), parts[1]))
    return entries


def _read_tum_groundtruth(path: Path) -> list[tuple[float, np.ndarray]]:
    poses: list[tuple[float, np.ndarray]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 8:
            continue
        timestamp = float(parts[0])
        tx, ty, tz = (float(value) for value in parts[1:4])
        quaternion = np.array([float(value) for value in parts[4:8]], dtype=np.float32)
        transform = np.eye(4, dtype=np.float32)
        transform[:3, :3] = quaternion_to_rotation_matrix(quaternion)
        transform[0, 3] = tx
        transform[1, 3] = ty
        transform[2, 3] = tz
        poses.append((timestamp, transform))
    return poses


def _associate_nearest(
    source: list[tuple[float, str]],
    target: list[tuple[float, str]] | list[tuple[float, np.ndarray]],
    tolerance: float,
) -> list[tuple[float, str, str | np.ndarray | None]]:
    target_timestamps = [entry[0] for entry in target]
    associated: list[tuple[float, str, str | np.ndarray | None]] = []
    for source_timestamp, source_value in source:
        index = bisect_left(target_timestamps, source_timestamp)
        candidates: list[int] = []
        if index < len(target_timestamps):
            candidates.append(index)
        if index > 0:
            candidates.append(index - 1)
        best_index = None
        best_delta = None
        for candidate in candidates:
            delta = abs(target_timestamps[candidate] - source_timestamp)
            if delta <= tolerance and (best_delta is None or delta < best_delta):
                best_index = candidate
                best_delta = delta
        associated.append(
            (
                source_timestamp,
                source_value,
                None if best_index is None else target[best_index][1],
            )
        )
    return associated


def associate_tum_entries(
    source: list[tuple[float, T]], target: list[tuple[float, T]], tolerance: float
) -> tuple[list[tuple[float, T, float, T]], AssociationReport]:
    """Associate ordered TUM streams once, preserving the timestamp of each observation."""
    target_timestamps = [entry[0] for entry in target]
    used_targets: set[int] = set()
    pairs: list[tuple[float, T, float, T]] = []
    errors: list[float] = []
    for source_timestamp, source_value in source:
        index = bisect_left(target_timestamps, source_timestamp)
        candidates = (index - 1, index)
        best_index = min(
            (candidate for candidate in candidates if 0 <= candidate < len(target) and candidate not in used_targets),
            key=lambda candidate: abs(target_timestamps[candidate] - source_timestamp),
            default=None,
        )
        if best_index is None:
            continue
        target_timestamp, target_value = target[best_index]
        error = abs(target_timestamp - source_timestamp)
        if error > tolerance:
            continue
        used_targets.add(best_index)
        pairs.append((source_timestamp, source_value, target_timestamp, target_value))
        errors.append(error)
    report = AssociationReport(
        source_count=len(source),
        target_count=len(target),
        matched_pairs=len(pairs),
        unmatched_source=len(source) - len(pairs),
        unmatched_target=len(target) - len(used_targets),
        mean_timestamp_error=float(np.mean(errors)) if errors else None,
        max_timestamp_error=float(np.max(errors)) if errors else None,
    )
    return pairs, report


class TumRgbdFrameSource(FrameSource):
    def __init__(self, dataset_root: str | Path, tolerance: float = 0.03) -> None:
        self.dataset_root = Path(dataset_root)
        nested_root = self.dataset_root / self.dataset_root.name
        if nested_root.exists():
            self.dataset_root = nested_root
        self.tolerance = float(tolerance)
        if not self.dataset_root.exists():
            raise RuntimeError(f"TUM dataset root does not exist: {self.dataset_root}")
        self._rgb_entries = _read_tum_index(self.dataset_root / "rgb.txt")
        self._depth_entries = _read_tum_index(self.dataset_root / "depth.txt")
        self._pose_entries = _read_tum_groundtruth(self.dataset_root / "groundtruth.txt")
        if not self._rgb_entries or not self._depth_entries:
            raise RuntimeError(f"Expected rgb.txt and depth.txt entries under {self.dataset_root}")
        self._depth_pairs, self.depth_association = associate_tum_entries(
            self._rgb_entries, self._depth_entries, self.tolerance
        )
        self.association_rows = [
            {
                "rgb_timestamp": rgb_timestamp,
                "rgb_path": rgb_path,
                "depth_timestamp": depth_timestamp,
                "depth_path": depth_path,
                "timestamp_error": abs(depth_timestamp - rgb_timestamp),
            }
            for rgb_timestamp, rgb_path, depth_timestamp, depth_path in self._depth_pairs
        ]
        self._pose_pairs = _associate_nearest(self._rgb_entries, self._pose_entries, self.tolerance)

    def frames(self) -> Generator[FramePacket, None, None]:
        pose_by_timestamp = {timestamp: pose for timestamp, _, pose in self._pose_pairs}
        for timestamp, rgb_rel_path, _, depth_rel_path in self._depth_pairs:
            rgb_path = self.dataset_root / rgb_rel_path
            depth_path = self.dataset_root / str(depth_rel_path)
            rgb = cv2.imread(str(rgb_path))
            raw_depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
            if rgb is None:
                raise RuntimeError(f"Failed to read TUM RGB frame {rgb_path}")
            if raw_depth is None:
                raise RuntimeError(f"Failed to read TUM depth frame {depth_path}")
            depth_map = decode_tum_depth_png(raw_depth)
            pose_matrix = pose_by_timestamp.get(timestamp)
            yield FramePacket(
                image=rgb,
                timestamp=float(timestamp),
                depth_map=depth_map,
                pose_matrix=None if pose_matrix is None else np.asarray(pose_matrix, dtype=np.float32).copy(),
            )

    def get_camera_intrinsics(self) -> dict | None:
        return None

    def close(self) -> None:
        return None
