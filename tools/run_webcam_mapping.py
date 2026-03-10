from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import perf_counter

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.depth.estimator import DepthEstimator
from src.io.camera import create_frame_source
from src.main import ensure_output_dir
from src.mapping.pointcloud import PointCloudBuilder
from src.ros2.nodes import AtlasRosBridge
from src.slam.wrapper import SlamWrapper
from src.utils.config import deep_merge_dicts, load_config, validate_config
from src.utils.demo_video import DemoVideoRecorder
from src.utils.logger import get_logger
from src.utils.perf import Timer

try:
    import open3d as o3d
except ImportError:  # pragma: no cover
    o3d = None


LOGGER = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Atlas Perception in live webcam mapping mode.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to the base YAML config.")
    parser.add_argument(
        "--override-config",
        default="configs/webcam_mapping.yaml",
        help="Optional YAML config whose values override the base config.",
    )
    parser.add_argument("--camera-index", type=int, default=0, help="OpenCV webcam device index.")
    parser.add_argument("--width", type=int, default=640, help="Capture width.")
    parser.add_argument("--height", type=int, default=480, help="Capture height.")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N frames. Use 0 to run until quit.")
    parser.add_argument(
        "--slam-mode",
        choices=("disabled", "dummy", "rtabmap"),
        default=None,
        help="Override the SLAM backend for the live run.",
    )
    parser.add_argument(
        "--representation",
        choices=("pointcloud", "tsdf"),
        default=None,
        help="Override the mapping representation for the live run.",
    )
    parser.add_argument(
        "--show-cloud",
        action="store_true",
        help="Open a live Open3D point-cloud window alongside the dashboard.",
    )
    parser.add_argument(
        "--save-artifacts",
        action="store_true",
        help="Export the accumulated point cloud and trajectory artifacts on exit.",
    )
    return parser.parse_args()


def build_runtime_config(args: argparse.Namespace) -> dict:
    config = load_config(args.config, args.override_config)
    override = {
        "input": {
            "mode": "webcam",
            "source": int(args.camera_index),
            "width": int(args.width),
            "height": int(args.height),
        },
        "ros2": {"enabled": False},
        "output": {
            "save_demo_video": False,
            "visualize": False,
            "output_dir": "data/outputs/webcam_mapping",
        },
    }
    if args.slam_mode is not None:
        override["slam"] = {"mode": args.slam_mode}
    if args.representation is not None:
        override.setdefault("mapping", {})
        override["mapping"]["representation"] = args.representation
    if args.save_artifacts:
        override["output"]["save_pointcloud"] = True
        override["output"]["save_trajectory"] = True
    else:
        override["output"]["save_pointcloud"] = False
        override["output"]["save_trajectory"] = False
    return validate_config(deep_merge_dicts(config, override))


class LivePointCloudViewer:
    def __init__(self) -> None:
        if o3d is None:
            raise RuntimeError("Open3D is required for the live point-cloud window.")
        self.visualizer = o3d.visualization.Visualizer()
        if not self.visualizer.create_window(window_name="Atlas Live Point Cloud", width=960, height=720):
            raise RuntimeError("Failed to create the Open3D live point-cloud window.")
        self.cloud = o3d.geometry.PointCloud()
        self._geometry_added = False
        render_option = self.visualizer.get_render_option()
        if render_option is not None:
            render_option.point_size = 2.0
            render_option.background_color = [0.04, 0.04, 0.04]

    def update(self, mapper: PointCloudBuilder) -> None:
        latest = mapper.to_open3d()
        self.cloud.points = latest.points
        self.cloud.colors = latest.colors
        if not self._geometry_added:
            self.visualizer.add_geometry(self.cloud)
            self._geometry_added = True
        else:
            self.visualizer.update_geometry(self.cloud)
        self.visualizer.poll_events()
        self.visualizer.update_renderer()

    def close(self) -> None:
        self.visualizer.destroy_window()


def run_webcam_mapping(args: argparse.Namespace | None = None) -> Path:
    parsed = args or parse_args()
    config = build_runtime_config(parsed)
    output_dir = ensure_output_dir(config["output"]["output_dir"])

    source = create_frame_source(config["input"])
    depth_estimator = DepthEstimator(config["depth"])
    slam = SlamWrapper(config["slam"])
    mapper = PointCloudBuilder(config["camera"], config["mapping"])
    ros_bridge = AtlasRosBridge(config["ros2"])
    cloud_viewer = LivePointCloudViewer() if parsed.show_cloud else None

    processed = 0
    start_time = perf_counter()
    try:
        for frame in source.frames():
            mapper.update_camera_intrinsics(getattr(source, "get_camera_intrinsics", lambda: None)())
            with Timer() as depth_timer:
                depth_map = depth_estimator.predict(frame.image)
            with Timer() as mapping_timer:
                pose = slam.update(frame.image, depth_map, frame.timestamp)
                point_cloud = mapper.integrate(depth_map, frame.image, pose)

            ros_bridge.publish_depth(depth_map, frame.timestamp)
            ros_bridge.publish_pose(pose, frame.timestamp)
            ros_bridge.publish_trajectory(slam.trajectory, frame.timestamp)
            ros_bridge.publish_pointcloud(point_cloud, frame.timestamp)

            elapsed = max(perf_counter() - start_time, 1e-6)
            dashboard = DemoVideoRecorder.compose_frame(
                rgb=frame.image,
                depth_map=depth_map,
                trajectory=slam.trajectory,
                pose=pose,
                metrics={
                    "depth_ms": depth_timer.result.milliseconds,
                    "mapping_ms": mapping_timer.result.milliseconds,
                    "fps": (processed + 1) / elapsed,
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
                frame_size=(1280, 720),
            )
            cv2.imshow("Atlas Webcam Mapping", dashboard)
            if cloud_viewer is not None:
                cloud_viewer.update(mapper)

            processed += 1
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if parsed.max_frames > 0 and processed >= parsed.max_frames:
                break
    finally:
        source.close()
        slam.shutdown()
        ros_bridge.shutdown()
        if cloud_viewer is not None:
            cloud_viewer.close()
        cv2.destroyAllWindows()

    if config["output"].get("save_pointcloud", False):
        mapper.export_ply(output_dir / "frame_cloud.ply")
        if str(config["mapping"].get("representation", "pointcloud")).lower() == "tsdf":
            mapper.export_mesh(output_dir / "tsdf_mesh.ply")
    if config["output"].get("save_trajectory", False):
        slam.export_trajectory(output_dir / "trajectory.npy")

    LOGGER.info("Processed %s webcam frames", processed)
    return output_dir


def main() -> None:
    output_dir = run_webcam_mapping()
    print(f"Webcam mapping artifacts: {output_dir}")


if __name__ == "__main__":
    main()
