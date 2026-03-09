from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import yaml


def _read_yaml(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config at {path} must deserialize to a dictionary.")
    return data


def deep_merge_dicts(base: dict, override: dict) -> dict:
    merged = deepcopy(base)
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = deep_merge_dicts(current, value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_config(path: str | Path, override_path: str | Path | None = None) -> dict:
    config = _read_yaml(path)
    if override_path is None:
        return config
    override = _read_yaml(override_path)
    return deep_merge_dicts(config, override)
