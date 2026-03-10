from pathlib import Path

import cv2
import numpy as np

from tools.export_demo_gif import export_demo_gif
from tools.run_demo import build_demo_command


def test_build_demo_command_for_tum_uses_expected_configs():
    command = build_demo_command("tum")
    assert command[1:4] == ["-m", "src.main", "--config"]
    assert "configs/default.yaml" in command
    assert "configs/tum_demo.yaml" in command
    assert command[-1] == "120"


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

    out = export_demo_gif(video_path, gif_path, fps=5.0, max_frames=4, width=64)
    assert out == gif_path
    assert gif_path.exists()
    assert gif_path.stat().st_size > 0
