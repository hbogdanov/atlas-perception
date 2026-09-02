from pathlib import Path

import cv2
import numpy as np
import pytest

from tools import run_tum_artifact


def test_tum_artifact_requires_depth_for_input_mode(tmp_path: Path, monkeypatch):
    rgb_path = tmp_path / "rgb.png"
    cv2.imwrite(str(rgb_path), np.zeros((4, 4, 3), dtype=np.uint8))
    args = type(
        "Args",
        (),
        {
            "rgb": str(rgb_path),
            "depth": None,
            "config": "configs/default.yaml",
            "override_config": "configs/tum_demo.yaml",
            "out_dir": str(tmp_path / "out"),
        },
    )()
    monkeypatch.setattr(run_tum_artifact, "parse_args", lambda: args)

    with pytest.raises(RuntimeError, match="--depth is required"):
        run_tum_artifact.main()
