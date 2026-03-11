from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.depth.estimator import DepthEstimator
from src.io.camera import create_frame_source
from src.main import ensure_output_dir
from src.mapping.pointcloud import PointCloudBuilder
from src.ros2.nodes import AtlasRosBridge
from src.semantics.segmenter import SemanticSegmenter
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
WINDOW_NAME = "Atlas Webcam Mapping"
_CLOSE_BUTTON_RECT = (1110, 16, 1260, 56)


def _badge_style_for_mode(mode: str) -> tuple[str, tuple[int, int, int], tuple[int, int, int]]:
    normalized = str(mode).lower()
    if normalized == "dummy":
        return "SLAM: DUMMY", (255, 248, 230), (173, 112, 24)
    if normalized == "rtabmap":
        return "SLAM: RTABMAP", (231, 248, 238), (35, 122, 69)
    return "SLAM: DISABLED", (240, 240, 240), (90, 90, 90)


def _trajectory_title_for_mode(mode: str) -> str:
    normalized = str(mode).lower()
    if normalized == "dummy":
        return "Synthetic Camera Path (World XY)"
    if normalized == "rtabmap":
        return "Tracked Camera Pose (World XY)"
    return "Fixed Camera Pose (World XY)"


def _build_fixed_pose_panel(width: int, height: int) -> np.ndarray:
    panel = 255 * np.ones((height, width, 3), dtype=np.uint8)
    lines = [
        ("Pose mode: fixed identity", 0.9, (30, 30, 30), 2),
        ("No motion tracking is active.", 0.7, (45, 45, 45), 2),
        ("Webcam default keeps pose fixed", 0.64, (70, 70, 70), 1),
        ("to avoid synthetic tracking claims.", 0.64, (70, 70, 70), 1),
        ("Use --slam-mode dummy for a", 0.62, (90, 90, 90), 1),
        ("visualization-only synthetic path.", 0.62, (90, 90, 90), 1),
    ]
    y = 120
    for text, scale, color, thickness in lines:
        cv2.putText(panel, text, (40, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)
        y += 52
    cv2.rectangle(panel, (28, 28), (width - 28, height - 28), (210, 210, 210), 2)
    return panel


def _build_dummy_trajectory_panel(trajectory_image: np.ndarray) -> np.ndarray:
    panel = trajectory_image.copy()
    overlay = panel.copy()
    box_left = 34
    box_top = panel.shape[0] - 116
    box_right = panel.shape[1] - 34
    box_bottom = panel.shape[0] - 30
    cv2.rectangle(overlay, (box_left, box_top), (box_right, box_bottom), (248, 248, 248), -1)
    panel = cv2.addWeighted(overlay, 0.78, panel, 0.22, 0.0)
    cv2.rectangle(panel, (box_left, box_top), (box_right, box_bottom), (192, 192, 192), 1)
    cv2.putText(
        panel,
        "Synthetic path",
        (box_left + 20, box_top + 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.78,
        (100, 100, 100),
        2,
    )
    cv2.putText(
        panel,
        "For visualization only",
        (box_left + 20, box_top + 72),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.66,
        (118, 118, 118),
        2,
    )
    return panel


def _draw_mode_badge(dashboard: np.ndarray, mode: str) -> np.ndarray:
    label, fill_color, border_color = _badge_style_for_mode(mode)
    x1, y1, x2, y2 = 20, 16, 270, 58
    cv2.rectangle(dashboard, (x1, y1), (x2, y2), fill_color, -1)
    cv2.rectangle(dashboard, (x1, y1), (x2, y2), border_color, 2)
    cv2.putText(
        dashboard,
        label,
        (x1 + 16, y1 + 29),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        border_color,
        2,
    )
    return dashboard


class LiveWindowController:
    def __init__(self) -> None:
        self.should_close = False
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WINDOW_NAME, self._on_mouse)

    def draw_overlay(self, dashboard):
        x1, y1, x2, y2 = _CLOSE_BUTTON_RECT
        cv2.rectangle(dashboard, (x1, y1), (x2, y2), (40, 62, 204), -1)
        cv2.rectangle(dashboard, (x1, y1), (x2, y2), (18, 24, 120), 2)
        cv2.putText(dashboard, "Close", (x1 + 36, y1 + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(
            dashboard,
            "Press Q or Esc to quit",
            (x1 - 205, y2 - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (45, 45, 45),
            1,
        )
        return dashboard

    def _on_mouse(self, event, x, y, flags, param) -> None:
        del flags, param
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        x1, y1, x2, y2 = _CLOSE_BUTTON_RECT
        if x1 <= x <= x2 and y1 <= y <= y2:
            self.should_close = True


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
    semantic_segmenter = SemanticSegmenter(config.get("semantics"))
    slam = SlamWrapper(config["slam"])
    mapper = PointCloudBuilder(config["camera"], config["mapping"])
    ros_bridge = AtlasRosBridge(config["ros2"])
    cloud_viewer = LivePointCloudViewer() if parsed.show_cloud else None
    window_controller = LiveWindowController()

    processed = 0
    start_time = perf_counter()
    point_cloud = None
    slam_mode = str(config["slam"]["mode"]).lower()
    try:
        for frame in source.frames():
            mapper.update_camera_intrinsics(getattr(source, "get_camera_intrinsics", lambda: None)())
            with Timer() as depth_timer:
                depth_map = depth_estimator.predict(frame.image)
            with Timer() as semantic_timer:
                semantic_prediction = semantic_segmenter.predict(frame.image)
            with Timer() as mapping_timer:
                pose = slam.update(frame.image, depth_map, frame.timestamp)
                point_cloud = mapper.integrate(depth_map, frame.image, pose, semantics=semantic_prediction)

            ros_bridge.publish_depth(depth_map, frame.timestamp)
            ros_bridge.publish_pose(pose, frame.timestamp)
            ros_bridge.publish_trajectory(slam.trajectory, frame.timestamp)
            ros_bridge.publish_pointcloud(point_cloud, frame.timestamp)

            elapsed = max(perf_counter() - start_time, 1e-6)
            trajectory_image = None
            if slam_mode == "disabled":
                trajectory_image = _build_fixed_pose_panel(600, 320)
            elif slam_mode == "dummy":
                trajectory_image = _build_dummy_trajectory_panel(slam.trajectory.render_plot(size=600))
            dashboard = DemoVideoRecorder.compose_frame(
                rgb=frame.image,
                depth_map=depth_map,
                trajectory=slam.trajectory,
                pose=pose,
                metrics={
                    "depth_ms": depth_timer.result.milliseconds,
                    "semantic_ms": semantic_timer.result.milliseconds,
                    "mapping_ms": mapping_timer.result.milliseconds,
                    "fps": (processed + 1) / elapsed,
                    "points": point_cloud.points.shape[0],
                },
                runtime={
                    "input_mode": str(config["input"]["mode"]),
                    "slam_mode": slam_mode,
                    "frame_id": str(config["ros2"].get("frame_id", "atlas_camera")),
                    "depth_topic": str(config["ros2"].get("depth_topic", "/atlas/depth")),
                    "pose_topic": str(config["ros2"].get("pose_topic", "/atlas/pose")),
                    "path_topic": str(config["ros2"].get("path_topic", "/atlas/path")),
                    "pointcloud_topic": str(config["ros2"].get("pointcloud_topic", "/atlas/pointcloud")),
                    "rgb_title": "Live Camera Feed",
                    "depth_title": "Estimated Depth",
                    "trajectory_title": _trajectory_title_for_mode(slam_mode),
                    "status_title": "Runtime Status",
                },
                frame_size=(1280, 720),
                trajectory_image=trajectory_image,
            )
            dashboard = _draw_mode_badge(dashboard, slam_mode)
            dashboard = window_controller.draw_overlay(dashboard)
            cv2.imshow(WINDOW_NAME, dashboard)
            if cloud_viewer is not None:
                cloud_viewer.update(mapper)

            processed += 1
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")) or window_controller.should_close:
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
        if point_cloud is not None and point_cloud.semantic_labels is not None:
            mapper.export_semantic_ply(output_dir / "semantic_cloud.ply")
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
