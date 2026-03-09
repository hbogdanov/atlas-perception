from __future__ import annotations

import argparse
from pathlib import Path

from src.depth.estimator import DepthEstimator
from src.depth.visualize import colorize_depth
from src.io.camera import create_frame_source
from src.mapping.pointcloud import PointCloudBuilder
from src.ros2.nodes import AtlasRosBridge
from src.sim.factory import create_sim_bridge
from src.slam.wrapper import SlamWrapper
from src.utils.config import load_config
from src.utils.logger import get_logger


LOGGER = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Atlas Perception pipeline.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to the base YAML config.")
    parser.add_argument(
        "--override-config",
        default=None,
        help="Optional YAML config whose values recursively override the base config.",
    )
    parser.add_argument("--max-frames", type=int, default=10, help="Frames to process before exit.")
    return parser.parse_args()


def ensure_output_dir(path_str: str) -> Path:
    output_dir = Path(path_str)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def run() -> None:
    args = parse_args()
    config = load_config(args.config, args.override_config)
    sim_bridge = create_sim_bridge(config.get("sim"))
    if sim_bridge is not None:
        config = sim_bridge.apply(config)
    output_dir = ensure_output_dir(config["output"]["output_dir"])

    source = create_frame_source(config["input"])
    depth_estimator = DepthEstimator(config["depth"])
    slam = SlamWrapper(config["slam"])
    mapper = PointCloudBuilder(config["camera"], config["mapping"])
    ros_bridge = AtlasRosBridge(config["ros2"])

    LOGGER.info("Starting pipeline with input mode=%s", config["input"]["mode"])

    processed = 0
    for frame in source.frames():
        timestamp = frame.timestamp
        rgb = frame.image
        depth_map = depth_estimator.predict(rgb)
        pose = slam.update(rgb, depth_map, timestamp)
        point_cloud = mapper.integrate(depth_map, rgb, pose)

        ros_bridge.publish_depth(depth_map, timestamp)
        ros_bridge.publish_pose(pose, timestamp)
        ros_bridge.publish_pointcloud(point_cloud, timestamp)

        if config["output"].get("visualize", False):
            _ = colorize_depth(depth_map)

        processed += 1
        if args.max_frames > 0 and processed >= args.max_frames:
            break

    if config["output"].get("save_pointcloud", False):
        mapper.export_ply(output_dir / "frame_cloud.ply")
    if config["output"].get("save_trajectory", False):
        slam.export_trajectory(output_dir / "trajectory.npy")

    source.close()
    ros_bridge.shutdown()
    LOGGER.info("Processed %s frames", processed)


if __name__ == "__main__":
    run()
