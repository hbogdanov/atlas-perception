from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a compact GIF from an Atlas demo video.")
    parser.add_argument("--video", required=True, help="Input demo video path.")
    parser.add_argument("--gif", required=True, help="Output GIF path.")
    parser.add_argument("--fps", type=float, default=8.0, help="Target GIF FPS.")
    parser.add_argument("--max-frames", type=int, default=80, help="Maximum frames to include in the GIF.")
    parser.add_argument("--width", type=int, default=960, help="Resize GIF frames to this width.")
    return parser.parse_args()


def export_demo_gif(
    video_path: str | Path,
    gif_path: str | Path,
    fps: float = 8.0,
    max_frames: int = 80,
    width: int = 960,
) -> Path:
    video_path = Path(video_path)
    gif_path = Path(gif_path)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video for GIF export: {video_path}")

    frames: list[Image.Image] = []
    try:
        source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        source_fps = source_fps if source_fps > 0.0 else max(float(fps), 1.0)
        stride = max(1, int(round(source_fps / max(float(fps), 1.0))))

        frame_index = 0
        while len(frames) < max_frames:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % stride != 0:
                frame_index += 1
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if width > 0 and rgb.shape[1] != width:
                height = max(1, int(rgb.shape[0] * (width / rgb.shape[1])))
                rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
            frames.append(Image.fromarray(rgb))
            frame_index += 1
    finally:
        capture.release()

    if not frames:
        raise RuntimeError(f"No frames were decoded from video: {video_path}")

    gif_path.parent.mkdir(parents=True, exist_ok=True)
    duration_ms = int(round(1000.0 / max(float(fps), 1.0)))
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )
    for frame in frames:
        frame.close()
    return gif_path


def main() -> None:
    args = parse_args()
    out = export_demo_gif(args.video, args.gif, fps=args.fps, max_frames=args.max_frames, width=args.width)
    print(f"GIF: {out}")


if __name__ == "__main__":
    main()
