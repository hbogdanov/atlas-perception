from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.depth.estimator import DepthEstimator
from src.depth.visualize import colorize_depth
from src.mapping.pointcloud import PointCloudBuilder
from src.slam.wrapper import SlamWrapper
from src.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Atlas demo artifacts from a single TUM RGB frame.")
    parser.add_argument("--rgb", required=True, help="Path to a TUM RGB frame PNG.")
    parser.add_argument("--config", default="configs/default.yaml", help="Base config path.")
    parser.add_argument("--override-config", default=None, help="Optional override config path.")
    parser.add_argument("--out-dir", default="data/outputs/tum_demo", help="Output directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config, args.override_config)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rgb_path = Path(args.rgb)
    image = cv2.imread(str(rgb_path))
    if image is None:
        raise RuntimeError(f"Failed to read image: {rgb_path}")

    depth_estimator = DepthEstimator(config["depth"])
    slam = SlamWrapper({"mode": "disabled"})
    mapper = PointCloudBuilder(config["camera"], config["mapping"])

    depth_map = depth_estimator.predict(image)
    pose = slam.update(image, depth_map, 0.0)
    mapper.integrate(depth_map, image, pose)

    rgb_out = out_dir / "rgb_frame.png"
    depth_out = out_dir / "depth_map.png"
    cloud_out = out_dir / "frame_cloud.ply"

    shutil.copy2(rgb_path, rgb_out)
    cv2.imwrite(str(depth_out), colorize_depth(depth_map))
    mapper.export_ply(cloud_out)

    print(f"RGB: {rgb_out}")
    print(f"Depth: {depth_out}")
    print(f"Point cloud: {cloud_out}")
    print("Screenshot: open the PLY in Open3D or RViz and save a viewer screenshot manually.")


if __name__ == "__main__":
    main()
