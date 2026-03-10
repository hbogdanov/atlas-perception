from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import cv2

from src.depth.estimator import DepthEstimator
from src.depth.visualize import colorize_depth
from src.io.camera import create_frame_source
from src.mapping.pointcloud import PointCloudBuilder
from src.ros2.nodes import AtlasRosBridge
from src.semantics.segmenter import SemanticSegmenter
from src.sim.factory import create_sim_bridge
from src.slam.wrapper import SlamWrapper
from src.utils.config import load_config
from src.utils.demo_video import DemoVideoRecorder
from src.utils.logger import get_logger
from src.utils.perf import Timer


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


def save_demo_snapshots(output_dir: Path, rgb, depth_map, semantics, config: dict, saved: set[str]) -> None:
    if config["output"].get("save_rgb_snapshot", False) and "rgb" not in saved:
        cv2.imwrite(str(output_dir / "rgb_frame.png"), rgb)
        saved.add("rgb")
    if config["output"].get("save_depth_snapshot", False) and "depth" not in saved:
        cv2.imwrite(str(output_dir / "depth_map.png"), colorize_depth(depth_map))
        saved.add("depth")
    if config["output"].get("save_semantic_snapshot", False) and "semantic" not in saved and semantics is not None:
        cv2.imwrite(str(output_dir / "semantic_overlay.png"), semantics.overlay(rgb))
        saved.add("semantic")


def create_demo_video_recorder(config: dict, output_dir: Path) -> DemoVideoRecorder | None:
    if not config["output"].get("save_demo_video", False):
        return None
    path = Path(config["output"].get("demo_video_path", output_dir / "atlas_demo.mp4"))
    fps = float(config["output"].get("demo_video_fps", config["input"].get("fps", 15)))
    width = int(config["output"].get("demo_video_width", 1280))
    height = int(config["output"].get("demo_video_height", 720))
    return DemoVideoRecorder(path=path, fps=fps, width=width, height=height)


def run() -> None:
    args = parse_args()
    try:
        config = load_config(args.config, args.override_config)
    except Exception as exc:
        raise RuntimeError(f"Failed to load configuration: {exc}") from exc
    sim_bridge = create_sim_bridge(config.get("sim"))
    if sim_bridge is not None:
        config = sim_bridge.apply(config)
    output_dir = ensure_output_dir(config["output"]["output_dir"])

    demo_video: DemoVideoRecorder | None = None
    try:
        source = create_frame_source(config["input"])
        depth_estimator = DepthEstimator(config["depth"])
        semantic_segmenter = SemanticSegmenter(config.get("semantics"))
        slam = SlamWrapper(config["slam"])
        mapper = PointCloudBuilder(config["camera"], config["mapping"])
        ros_bridge = AtlasRosBridge(config["ros2"])
        demo_video = create_demo_video_recorder(config, output_dir)
    except Exception as exc:
        raise RuntimeError(f"Failed to initialize pipeline components: {exc}") from exc

    LOGGER.info("Starting pipeline with input mode=%s", config["input"]["mode"])

    processed = 0
    saved_snapshots: set[str] = set()
    start_time = perf_counter()
    depth_times_ms: list[float] = []
    semantic_times_ms: list[float] = []
    mapping_times_ms: list[float] = []
    latest_point_count = 0
    try:
        for frame in source.frames():
            timestamp = frame.timestamp
            rgb = frame.image
            mapper.update_camera_intrinsics(getattr(source, "get_camera_intrinsics", lambda: None)())
            with Timer() as depth_timer:
                depth_map = depth_estimator.predict(rgb)
            with Timer() as semantic_timer:
                semantic_prediction = semantic_segmenter.predict(rgb)
            pose = slam.update(rgb, depth_map, timestamp)
            with Timer() as mapping_timer:
                point_cloud = mapper.integrate(depth_map, rgb, pose, semantics=semantic_prediction)

            ros_bridge.publish_depth(depth_map, timestamp)
            ros_bridge.publish_pose(pose, timestamp)
            ros_bridge.publish_trajectory(slam.trajectory, timestamp)
            ros_bridge.publish_pointcloud(point_cloud, timestamp)

            save_demo_snapshots(output_dir, rgb, depth_map, semantic_prediction, config, saved_snapshots)
            if demo_video is not None:
                demo_video.write(
                    rgb=rgb,
                    depth_map=depth_map,
                    trajectory=slam.trajectory,
                    pose=pose,
                    metrics={
                        "depth_ms": depth_timer.result.milliseconds,
                        "semantic_ms": semantic_timer.result.milliseconds,
                        "mapping_ms": mapping_timer.result.milliseconds,
                        "fps": processed / max(perf_counter() - start_time, 1e-6),
                        "points": point_cloud.points.shape[0],
                    },
                    runtime={
                        "input_mode": str(config["input"]["mode"]),
                        "slam_mode": str(config["slam"]["mode"]),
                        "frame_id": str(config["ros2"].get("frame_id", "atlas_camera")),
                        "depth_topic": str(config["ros2"].get("depth_topic", "/atlas/depth")),
                        "pose_topic": str(config["ros2"].get("pose_topic", "/atlas/pose")),
                        "path_topic": str(config["ros2"].get("path_topic", "/atlas/path")),
                        "pointcloud_topic": str(config["ros2"].get("pointcloud_topic", "/atlas/pointcloud")),
                    },
                )

            if config["output"].get("visualize", False):
                _ = colorize_depth(depth_map)

            processed += 1
            depth_times_ms.append(depth_timer.result.milliseconds)
            semantic_times_ms.append(semantic_timer.result.milliseconds)
            mapping_times_ms.append(mapping_timer.result.milliseconds)
            latest_point_count = point_cloud.points.shape[0]
            elapsed = max(perf_counter() - start_time, 1e-6)
            LOGGER.info(
                "frame=%s depth_ms=%.2f semantic_ms=%.2f mapping_ms=%.2f fps=%.2f points=%s",
                processed,
                depth_timer.result.milliseconds,
                semantic_timer.result.milliseconds,
                mapping_timer.result.milliseconds,
                processed / elapsed,
                point_cloud.points.shape[0],
            )
            if args.max_frames > 0 and processed >= args.max_frames:
                break
    except Exception as exc:
        raise RuntimeError(f"Pipeline execution failed after {processed} frames: {exc}") from exc

    if config["output"].get("save_pointcloud", False):
        try:
            mapper.export_ply(output_dir / "frame_cloud.ply")
            if point_cloud.semantic_labels is not None:
                mapper.export_semantic_ply(output_dir / "semantic_cloud.ply")
            if str(config["mapping"].get("representation", "pointcloud")).lower() == "tsdf":
                mapper.export_mesh(output_dir / "tsdf_mesh.ply")
        except Exception as exc:
            raise RuntimeError(f"Failed to export point cloud: {exc}") from exc
    if config["output"].get("save_trajectory", False):
        slam.export_trajectory(output_dir / "trajectory.npy")

    source.close()
    slam.shutdown()
    ros_bridge.shutdown()
    if demo_video is not None:
        demo_video.close()
    total_elapsed = max(perf_counter() - start_time, 1e-6)
    if processed:
        LOGGER.info(
            "summary avg_depth_ms=%.2f avg_semantic_ms=%.2f avg_mapping_ms=%.2f avg_fps=%.2f points=%s",
            sum(depth_times_ms) / len(depth_times_ms),
            sum(semantic_times_ms) / len(semantic_times_ms),
            sum(mapping_times_ms) / len(mapping_times_ms),
            processed / total_elapsed,
            latest_point_count,
        )
    LOGGER.info("Processed %s frames", processed)


if __name__ == "__main__":
    run()
