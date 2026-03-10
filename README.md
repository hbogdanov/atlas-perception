# Atlas Perception

Atlas Perception is a ROS2-compatible robotics perception pipeline that converts camera streams into depth estimates and spatial outputs for downstream navigation in simulated environments.

Additional documentation:

- [docs/index.md](docs/index.md)
- [docs/architecture.md](docs/architecture.md)
- [docs/ros_topics.md](docs/ros_topics.md)
- [docs/sample_run.md](docs/sample_run.md)
- [docs/ubuntu_gazebo_setup.md](docs/ubuntu_gazebo_setup.md)

## Features

- Camera ingestion from webcam, video files, or ROS2 image topics
- ROS2 image-topic ingestion through `rclpy` and `cv_bridge`
- Real monocular depth backends for MiDaS or Depth Anything
- Depth + trajectory hooks with future SLAM backend integration
- Explicit SLAM modes for `disabled`, `dummy`, and `rtabmap`
- Point cloud generation with NumPy-native storage and Open3D `.ply` export
- ROS2 topic publishing for depth, pose, and colored point cloud outputs
- Config-driven simulator workflows for Isaac Sim and Gazebo
- Composite demo video export showing feed, depth, trajectory, and published-output status

## Pipeline

```text
camera input
   |
   v
depth estimation
   |
   +--> trajectory / future SLAM backend
   |
   v
point cloud projection + transform
   |
   v
map / cloud accumulation
   |
   v
ROS2 topic publishing
```

## Quickstart

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .[dev]
python -m src.main --config configs/default.yaml
python -m src.main --config configs/default.yaml --override-config configs/gazebo_demo.yaml
```

The first run of a Torch Hub backend may download model assets. For more reproducible setups, pin `torch` and backend dependencies in your environment and set `depth.local_weights_path` to a local checkpoint when available.
The base `configs/default.yaml` quickstart keeps ROS2 publishing disabled; simulator and ROS-specific override configs enable it explicitly.
The main pipeline can also save a demo-ready artifact set directly via `output.save_rgb_snapshot`, `output.save_depth_snapshot`, and `output.save_pointcloud`.
For simulator-backed showcase runs, `output.save_demo_video` writes a composite `.mp4` with the camera feed, depth output, trajectory plot, and ROS topic/status panel.

## Configuration

Primary runtime settings live in `configs/default.yaml`. You can layer an additional YAML file on top with `--override-config`, and nested dictionaries are merged recursively.
The configs are intentionally split by purpose:

- `configs/default.yaml`: safe baseline with ROS2 off and SLAM disabled
- `configs/tum_main_eval.yaml`: reproducible TUM artifact evaluation run
- `configs/gazebo_demo.yaml`: Gazebo demo with dummy motion for world-frame accumulation testing
- `configs/gazebo_rtabmap_demo.yaml`: Gazebo demo that consumes external RTAB-Map poses
- `configs/turtlebot3_gazebo_rtabmap.yaml`: TurtleBot3 Gazebo run that consumes camera topics and external RTAB-Map poses
- `configs/isaac_demo.yaml`: Isaac Sim demo with dummy motion and ROS2 publishing

For ROS2 ingestion, `input.source` is the camera topic. There is no separate duplicate image-topic field under `ros2`.
When available, `input.camera_info_topic` can provide live intrinsics from `sensor_msgs/CameraInfo`, overriding static `camera.fx`, `camera.fy`, `camera.cx`, and `camera.cy` during ROS2 or simulator runs.

Depth outputs are explicit:

- `depth.output_mode: relative_normalized` returns a per-frame normalized relative depth map in `[0, 1]`
- `depth.output_mode: raw` returns the backend's raw depth output without pretending it is metric depth
- `depth.postprocess.enabled: true` turns on post-inference cleanup for smoother but still edge-aware depth
- `depth.postprocess.bilateral_filter: true` applies spatial smoothing to reduce speckle without flattening the full scene
- `depth.postprocess.guided_refine: true` runs an RGB-guided filter so object boundaries track image edges better
- `depth.postprocess.temporal_fusion: true` blends consecutive depth frames to reduce flicker in video or ROS streams

SLAM modes are explicit:

- `slam.mode: disabled` keeps pose fixed at identity
- `slam.mode: dummy` generates synthetic forward motion for pipeline testing
- `slam.mode: rtabmap` consumes external RTAB-Map pose output from ROS2 and uses it for world-frame cloud alignment

Config validation runs before startup and fails early on invalid camera intrinsics, unsupported modes, or missing required sections.

Example:

```bash
python -m src.main --config configs/default.yaml --override-config configs/isaac_demo.yaml
```

RTAB-Map example:

```bash
python -m src.main --config configs/default.yaml --override-config configs/gazebo_rtabmap_demo.yaml
```

## Simulator Runs

Use the simulator-specific configs directly:

```bash
python -m src.main --config configs/default.yaml --override-config configs/isaac_demo.yaml
python -m src.main --config configs/default.yaml --override-config configs/gazebo_demo.yaml
```

If ROS2 `launch` is available, the launch files also wrap those flows:

```bash
ros2 launch launch/atlas_perception.launch.py
ros2 launch launch/sim_demo.launch.py sim_config:=configs/gazebo_demo.yaml
```

Expected simulator showcase outputs:

- `demo/videos/gazebo_demo.mp4`
- `demo/videos/gazebo_rtabmap_demo.mp4`
- `demo/videos/turtlebot3_gazebo_rtabmap.mp4`
- `demo/videos/isaac_demo.mp4`

## Ubuntu Gazebo Workflow

For a real ROS2 + Gazebo setup, use Ubuntu 22.04 with ROS2 Humble and TurtleBot3. The exact install and run steps are in [docs/ubuntu_gazebo_setup.md](docs/ubuntu_gazebo_setup.md).

Canonical simulator run:

```bash
ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py
python -m src.main --config configs/default.yaml --override-config configs/turtlebot3_gazebo_rtabmap.yaml --max-frames 300
```

Before running Atlas, verify the simulator is actually publishing:

- `/camera/image_raw`
- `/camera/camera_info`

If the camera topic names differ, update `input.source` and `input.camera_info_topic` in `configs/turtlebot3_gazebo_rtabmap.yaml`.

## World-Frame Mapping

Atlas now treats pose as a mapping input rather than a side output:

1. ingest RGB frame
2. estimate dense depth
3. get `T_world_camera` from the configured SLAM backend
4. back-project depth into camera-frame 3D points
5. transform points into the world frame with `T_world_camera`
6. accumulate the transformed cloud into the global map
7. publish depth, pose, path, and colored point cloud topics

## Sample Run Artifacts

Documented outputs for a full run are described in `docs/sample_run.md`. A successful run should produce:

- an RGB input frame capture
- a depth visualization image
- `data/outputs/frame_cloud.ply`
- a screenshot from the point cloud viewer

Current generated demo artifacts:

- [tum_rgb_frame.png](demo/screenshots/tum_rgb_frame.png)
- [tum_depth_map.png](demo/screenshots/tum_depth_map.png)
- [pointcloud_vis.png](demo/screenshots/pointcloud_vis.png)
- [frame_cloud.ply](data/outputs/tum_demo/frame_cloud.ply)

## Demo Visuals

Projected point cloud:

![TUM point cloud](demo/screenshots/pointcloud_vis.png)

RGB input:

![TUM RGB frame](demo/screenshots/tum_rgb_frame.png)

Estimated depth:

![TUM depth map](demo/screenshots/tum_depth_map.png)

Recommended first dataset: TUM RGB-D `fr1/xyz`. The official TUM page recommends the `xyz` series for first experiments, and `fr1/xyz` is the smallest of the suggested starter sequences at about `0.47GB`. Sources: [download page](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download), [dataset overview](https://cvg.cit.tum.de/data/datasets/rgbd-dataset).

One-frame artifact flow:

```bash
python tools/run_tum_artifact.py --rgb data/samples/tum_freiburg1_xyz/rgb/1305031102.175304.png --out-dir data/outputs/tum_demo
```

Main-entrypoint artifact flow:

```bash
python -m src.main --config configs/default.yaml --override-config configs/gazebo_demo.yaml --max-frames 30
```

Full functionality evaluation on the TUM-derived 30-frame video clip used:

```bash
python -m src.main --config configs/default.yaml --override-config configs/tum_main_eval.yaml --max-frames 30
```

## Runtime Metrics

Measured from one real 30-frame TUM `fr1/xyz` video-derived run through `src.main`:

- average depth inference: `80.79 ms`
- average mapping / projection: `3.18 ms`
- average throughput: `11.45 FPS`
- accumulated point count: `100000`

## Expected Outputs

When snapshot and export flags are enabled, the main pipeline writes:

- `rgb_frame.png`
- `depth_map.png`
- `frame_cloud.ply`
- `trajectory.npy`
- `trajectory.json`
- `trajectory.csv`
- `trajectory_plot.png`
- `atlas_demo.mp4`

Example artifact directory:

- `data/outputs/tum_main_eval/rgb_frame.png`
- `data/outputs/tum_main_eval/depth_map.png`
- `data/outputs/tum_main_eval/frame_cloud.ply`
- `data/outputs/tum_main_eval/trajectory.npy`
- `data/outputs/tum_main_eval/trajectory.json`
- `data/outputs/tum_main_eval/trajectory.csv`
- `data/outputs/tum_main_eval/trajectory_plot.png`
- `data/outputs/tum_main_eval/atlas_eval_demo.mp4`

## Known Limitations

- Monocular depth is relative by default unless a calibrated backend/scaling path is added.
- `slam.mode: rtabmap` expects an external RTAB-Map ROS2 node to already be running and publishing poses.
- Simulator bridges are lightweight runtime/launch adapters, not deep simulator-specific integrations.
- A real simulator-backed demo video still requires a local Gazebo or Isaac Sim runtime; this repo can render the showcase asset once those feeds are available.

## ROS2 Topics

Default topics:

- `/camera/image_raw`
- `/atlas/depth`
- `/atlas/pose`
- `/atlas/path`
- `/atlas/pointcloud`

See `docs/ros_topics.md` for the topic contract.

## Development

Install test tooling with:

```bash
pip install -e .[dev]
```

## Project Layout

- `src/io`: camera and stream adapters
- `src/depth`: depth inference and visualization
- `src/slam`: trajectory and odometry interfaces
- `src/mapping`: point cloud projection, fusion, and occupancy utilities
- `src/ros2`: ROS2 nodes, publishers, subscribers, and transforms
- `src/sim`: Isaac Sim and Gazebo bridges plus launch-time topic adaptation
- `docs`: architecture and ROS interface documentation
- `launch`: ROS2 launch entrypoints

## Status

The repository now includes real camera ingestion, monocular depth estimation, world-aligned point cloud accumulation, trajectory publishing, and simulator-aware runtime/launch configuration. RTAB-Map is supported as an external ROS2 pose source for world-frame mapping, while deeper SLAM integration remains an incremental next step.
