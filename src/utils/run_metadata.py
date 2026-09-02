from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from csv import DictWriter
from pathlib import Path
from typing import Any


def config_hash(config: dict) -> str:
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def git_commit(repo_root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_associations_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_run_manifest(config: dict, repo_root: Path, association: Any = None) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "config_hash": config_hash(config),
        "git_commit": git_commit(repo_root),
        "input": dict(config.get("input", {})),
        "depth": dict(config.get("depth", {})),
        "slam": dict(config.get("slam", {})),
        "mapping": dict(config.get("mapping", {})),
        "camera": dict(config.get("camera", {})),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
    }
    if association is not None:
        manifest["associations"] = {
            "rgb_frames": association.source_count,
            "depth_frames": association.target_count,
            "matched_pairs": association.matched_pairs,
            "unmatched_rgb": association.unmatched_source,
            "unmatched_depth": association.unmatched_target,
            "mean_timestamp_error": association.mean_timestamp_error,
            "max_timestamp_error": association.max_timestamp_error,
        }
    return manifest
