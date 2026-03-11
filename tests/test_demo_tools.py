from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from tools.export_demo_gif import export_demo_gif
from tools.run_demo import build_demo_command
from tools.run_webcam_mapping import _badge_style_for_mode, _draw_mode_badge, build_runtime_config


def test_build_demo_command_for_tum_uses_expected_configs():
    command = build_demo_command("tum")
    assert command[1:4] == ["-m", "src.main", "--config"]
    assert "configs/default.yaml" in command
    assert "configs/tum_demo.yaml" in command
    assert command[-1] == "240"


def test_export_demo_gif_creates_output(tmp_path: Path):
    video_path = tmp_path / "demo.mp4"
    gif_path = tmp_path / "demo.gif"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 48))
    assert writer.isOpened()
    try:
        for index in range(6):
            frame = np.full((48, 64, 3), index * 30, dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()

    out = export_demo_gif(video_path, gif_path, fps=5.0, max_frames=4, width=64, duration_ms=100)
    assert out == gif_path
    assert gif_path.exists()
    assert gif_path.stat().st_size > 0
    with Image.open(gif_path) as image:
        assert image.info.get("duration") == 100


def test_build_webcam_runtime_config_applies_live_overrides():
    args = type(
        "Args",
        (),
        {
            "config": "configs/default.yaml",
            "override_config": "configs/webcam_mapping.yaml",
            "camera_index": 2,
            "width": 800,
            "height": 600,
            "max_frames": 0,
            "slam_mode": "dummy",
            "representation": "tsdf",
            "show_cloud": False,
            "save_artifacts": True,
        },
    )()
    config = build_runtime_config(args)
    assert config["input"]["mode"] == "webcam"
    assert config["input"]["source"] == 2
    assert config["input"]["width"] == 800
    assert config["input"]["height"] == 600
    assert config["slam"]["mode"] == "dummy"
    assert config["mapping"]["representation"] == "tsdf"
    assert config["output"]["save_pointcloud"] is True
    assert config["output"]["save_trajectory"] is True

def test_mode_badge_renders_for_each_slam_mode():
    assert _badge_style_for_mode("disabled")[0] == "SLAM: DISABLED"
    assert _badge_style_for_mode("dummy")[0] == "SLAM: DUMMY"
    assert _badge_style_for_mode("rtabmap")[0] == "SLAM: RTABMAP"

    canvas = np.full((120, 320, 3), 255, dtype=np.uint8)
    badged = _draw_mode_badge(canvas.copy(), "dummy")
    assert np.count_nonzero(badged != canvas) > 0
