from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from src.depth.input_depth import InputDepthProcessor
from src.depth.visualize import colorize_depth, normalize_depth_for_display
from src.evaluation.calibration_perturbation import CalibrationPerturber
from src.evaluation.pose_perturbation import PosePerturber
from src.io.camera import create_frame_source
from src.mapping.confidence import compute_depth_confidence
from src.mapping.pointcloud import PointCloudBuilder
from src.ros2.nodes import AtlasRosBridge
from src.semantics.segmenter import SemanticSegmenter
from src.sim.factory import create_sim_bridge
from src.slam.wrapper import SlamWrapper
from src.utils.config import load_config
from src.utils.demo_video import DemoVideoRecorder
from src.utils.logger import get_logger
from src.utils.perf import Timer
from src.utils.run_metadata import build_run_manifest, write_associations_csv, write_json
from src.vision.aruco_landmarks import ArucoLandmarkDetector
from src.vision.landmark_pose import solve_landmark_pose

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


def _display_depth_map(depth_map: np.ndarray, config: dict) -> np.ndarray:
    del config
    # Both metric input depth and relative model output need display-only normalization.
    return normalize_depth_for_display(depth_map)


def save_demo_snapshots(output_dir: Path, rgb, depth_map, semantics, config: dict, saved: set[str]) -> None:
    if config["output"].get("save_rgb_snapshot", False) and "rgb" not in saved:
        cv2.imwrite(str(output_dir / "rgb_frame.png"), rgb)
        saved.add("rgb")
    if config["output"].get("save_depth_snapshot", False) and "depth" not in saved:
        cv2.imwrite(str(output_dir / "depth_map.png"), colorize_depth(_display_depth_map(depth_map, config)))
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
        depth_source_mode = str(config["depth"].get("source_mode", "estimate")).lower()
        depth_estimator = None
        input_depth_processor = None
        if depth_source_mode == "input":
            input_depth_processor = InputDepthProcessor(config["depth"])
        else:
            # Keep metric RGB-D runs independent of Torch and pretrained weights.
            from src.depth.estimator import DepthEstimator

            depth_estimator = DepthEstimator(config["depth"])
        semantic_segmenter = SemanticSegmenter(config.get("semantics"))
        visual_config = config.get("visual_localization", {})
        slam = SlamWrapper(config["slam"], visual_config)
        pose_perturber = PosePerturber(config.get("evaluation", {}).get("pose_perturbation"))
        calibration_perturber = CalibrationPerturber(config.get("evaluation", {}).get("calibration_perturbation"))
        landmark_detector = ArucoLandmarkDetector(visual_config) if visual_config.get("enabled", False) else None
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
    latest_mapping_diagnostics: dict[str, float | int] = {}
    dropped_mapping_frames = 0
    visual_pose_measurements = 0
    try:
        for frame in source.frames():
            timestamp = frame.timestamp
            rgb = frame.image
            source_intrinsics = getattr(source, "get_camera_intrinsics", lambda: None)()
            mapper.update_camera_intrinsics(calibration_perturber.perturb(source_intrinsics or config["camera"]))
            with Timer() as depth_timer:
                if depth_source_mode == "input":
                    if frame.depth_map is None:
                        raise RuntimeError(
                            "depth.source_mode is 'input' but the active frame source " "does not provide depth."
                        )
                    if input_depth_processor is None:
                        raise RuntimeError("Input depth processor was not initialized.")
                    depth_map = input_depth_processor.prepare(frame.depth_map, rgb)
                else:
                    if depth_estimator is None:
                        raise RuntimeError("Depth estimator was not initialized.")
                    depth_map = depth_estimator.predict(rgb)
            with Timer() as semantic_timer:
                semantic_prediction = semantic_segmenter.predict(rgb)
            visual_measurement = None
            if landmark_detector is not None:
                visual_measurement = solve_landmark_pose(
                    landmark_detector.detect(rgb),
                    config["camera"],
                    timestamp,
                    max_reprojection_error=float(visual_config.get("max_reprojection_error", 3.0)),
                )
            pose = slam.update(
                rgb,
                depth_map,
                timestamp,
                pose_hint=frame.pose_matrix,
                visual_measurement=visual_measurement,
            )
            if visual_measurement is not None:
                ros_bridge.publish_visual_pose(visual_measurement)
                visual_pose_measurements += 1
            with Timer() as mapping_timer:
                mapping_pose = pose_perturber.perturb(pose)
                if mapping_pose is None:
                    point_cloud = mapper.data()
                    dropped_mapping_frames += 1
                    latest_mapping_diagnostics = {"mapping_skipped_for_pose_dropout": 1}
                else:
                    point_cloud = mapper.integrate(depth_map, rgb, mapping_pose, semantics=semantic_prediction)
                    latest_mapping_diagnostics = mapper.diagnostics()

            ros_bridge.publish_depth(depth_map, timestamp)
            ros_bridge.publish_pose(pose, timestamp)
            ros_bridge.publish_trajectory(slam.trajectory, timestamp)
            ros_bridge.publish_pointcloud(point_cloud, timestamp)

            save_demo_snapshots(output_dir, rgb, depth_map, semantic_prediction, config, saved_snapshots)
            if demo_video is not None:
                display_depth = _display_depth_map(depth_map, config)
                confidence_enabled = bool(config["mapping"].get("confidence_fusion", {}).get("enabled", False))
                confidence_map = compute_depth_confidence(depth_map) if confidence_enabled else None
                semantic_mode = "disabled"
                semantic_image = None
                semantic_summary = "disabled"
                if semantic_prediction is not None:
                    labeled_pixels = int((semantic_prediction.labels >= 0).sum())
                    semantic_mode = "detected" if labeled_pixels > 0 else "empty"
                    semantic_summary = f"{labeled_pixels} labeled px" if labeled_pixels > 0 else "0 recognized objects"
                    semantic_image = semantic_prediction.overlay(rgb)
                metrics = {
                    "depth_ms": depth_timer.result.milliseconds,
                    "semantic_ms": semantic_timer.result.milliseconds,
                    "mapping_ms": mapping_timer.result.milliseconds,
                    "fps": processed / max(perf_counter() - start_time, 1e-6),
                    "points": point_cloud.points.shape[0],
                    "frames": processed + 1,
                }
                demo_video.write(
                    rgb=DemoVideoRecorder.overlay_perception_diagnostics(
                        rgb,
                        confidence_map,
                        visual_measurement,
                        slam.last_visual_correction,
                    ),
                    depth_map=display_depth,
                    trajectory=slam.trajectory,
                    pose=pose,
                    metrics=metrics,
                    runtime={
                        "input_mode": str(config["input"]["mode"]),
                        "slam_mode": str(config["slam"]["mode"]),
                        "frame_id": str(config["ros2"].get("frame_id", "atlas_camera")),
                        "depth_topic": str(config["ros2"].get("depth_topic", "/atlas/depth")),
                        "pose_topic": str(config["ros2"].get("pose_topic", "/atlas/pose")),
                        "path_topic": str(config["ros2"].get("path_topic", "/atlas/path")),
                        "pointcloud_topic": str(config["ros2"].get("pointcloud_topic", "/atlas/pointcloud")),
                        "semantic_title": "Semantic Overlay",
                        "map_title": "Fused Point Cloud Map",
                        "semantic_mode": semantic_mode,
                        "semantic_summary": semantic_summary,
                        "confidence_mean": float(latest_mapping_diagnostics.get("mean_confidence", 1.0)),
                        "visual_pose_status": ("measurement" if visual_measurement is not None else "no measurement"),
                        "map_projection": str(config["output"].get("demo_map_projection", "auto")),
                        "map_bounds": config["output"].get("demo_map_bounds"),
                    },
                    semantic_image=semantic_image,
                    map_image=DemoVideoRecorder.render_topdown_map(
                        point_cloud,
                        pose,
                        metrics,
                        {
                            "slam_mode": str(config["slam"]["mode"]),
                            "map_projection": str(config["output"].get("demo_map_projection", "auto")),
                            "map_bounds": config["output"].get("demo_map_bounds"),
                        },
                    ),
                )

            if config["output"].get("visualize", False):
                _ = colorize_depth(_display_depth_map(depth_map, config))

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
        association = getattr(source, "depth_association", None)
        write_associations_csv(Path(output_dir / "associations.csv"), getattr(source, "association_rows", []))
        write_json(Path(output_dir / "config.json"), config)
        write_json(Path(output_dir / "manifest.json"), build_run_manifest(config, Path.cwd(), association))
        write_json(
            Path(output_dir / "runtime_metrics.json"),
            {
                "frames_processed": processed,
                "total_elapsed_seconds": total_elapsed,
                "avg_depth_ms": sum(depth_times_ms) / len(depth_times_ms),
                "avg_semantic_ms": sum(semantic_times_ms) / len(semantic_times_ms),
                "avg_mapping_ms": sum(mapping_times_ms) / len(mapping_times_ms),
                "avg_fps": processed / total_elapsed,
                "point_count": latest_point_count,
                "mapping_diagnostics": latest_mapping_diagnostics,
                "dropped_mapping_frames": dropped_mapping_frames,
                "visual_pose_measurements": visual_pose_measurements,
                "visual_pose_correction": (
                    None
                    if slam.last_visual_correction is None
                    else {
                        "applied": slam.last_visual_correction.applied,
                        "reason": slam.last_visual_correction.reason,
                        "translation_innovation_m": slam.last_visual_correction.translation_innovation_m,
                        "rotation_innovation_deg": slam.last_visual_correction.rotation_innovation_deg,
                    }
                ),
            },
        )
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
