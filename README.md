# Atlas Perception

Atlas Perception is a ROS2-compatible robotics perception pipeline that converts camera streams into depth estimates, localization cues, and spatial maps for downstream navigation in simulated environments.

## Features

- Camera ingestion from webcam, video files, or ROS2 image topics
- Real monocular depth backends for MiDaS or Depth Anything
- Pose and trajectory hooks for visual odometry or SLAM systems
- Open3D point cloud generation and `.ply` export
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
.venv\Scripts\activate
pip install -r requirements.txt
python -m src.main --config configs/default.yaml
```

## Configuration

Primary runtime settings live in `configs/default.yaml`. Override with simulator-specific settings such as `configs/isaac_demo.yaml` or `configs/gazebo_demo.yaml`.

## ROS2 Topics

Default topics:

- `/camera/image_raw`
- `/atlas/depth`
- `/atlas/pose`
- `/atlas/pointcloud`

See `docs/ros_topics.md` for the topic contract.

## Project Layout

- `src/io`: camera and stream adapters
- `src/depth`: depth inference and visualization
- `src/slam`: trajectory and odometry interfaces
- `src/mapping`: point cloud projection, fusion, and occupancy utilities
- `src/ros2`: ROS2 nodes, publishers, subscribers, and transforms
- `src/sim`: Isaac Sim and Gazebo bridges
- `docs`: architecture and ROS interface documentation
- `launch`: ROS2 launch entrypoints

## Status

The repository now provides a modular perception scaffold with real depth backend hooks, Open3D point cloud export, ROS2 publishing interfaces, and simulator-specific configuration.
