# Atlas Perception

Atlas Perception is a ROS2-compatible robotics perception pipeline that converts camera streams into depth estimates, localization cues, and spatial maps for downstream navigation in simulated environments.

## Features

- Camera ingestion from webcam, video files, or ROS2 image topics
- ROS2 image-topic ingestion through `rclpy` and `cv_bridge`
- Real monocular depth backends for MiDaS or Depth Anything
- Explicit SLAM modes for `disabled`, `dummy`, and backend wrappers
- Point cloud generation with NumPy-native storage and Open3D `.ply` export
- ROS2 topic publishing for depth, pose, and point cloud outputs
- Config-driven simulator workflows for Isaac Sim and Gazebo

## Pipeline

```text
camera input
   |
   v
depth estimation
   |
   v
point cloud projection
   |
   v
pose / visual odometry
   |
   v
map / cloud fusion
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

## Configuration

Primary runtime settings live in `configs/default.yaml`. You can layer an additional YAML file on top with `--override-config`, and nested dictionaries are merged recursively.

Depth outputs are explicit:

- `depth.output_mode: relative_normalized` returns a per-frame normalized relative depth map in `[0, 1]`
- `depth.output_mode: raw` returns the backend's raw depth output without pretending it is metric depth

SLAM modes are explicit:

- `slam.mode: disabled` keeps pose fixed at identity
- `slam.mode: dummy` generates synthetic forward motion for pipeline testing
- `slam.mode: orbslam_wrapper` reserves a backend integration point and currently raises until implemented

Example:

```bash
python -m src.main --config configs/default.yaml --override-config configs/isaac_demo.yaml
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

## ROS2 Topics

Default topics:

- `/camera/image_raw`
- `/atlas/depth`
- `/atlas/pose`
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

The repository now includes real ROS2 image ingestion, explicit depth and SLAM modes, NumPy-native point cloud accumulation with export adapters, and simulator-aware runtime/launch configuration for Isaac Sim and Gazebo.
