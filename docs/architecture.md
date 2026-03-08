# Architecture

Atlas Perception is organized as a modular perception pipeline with explicit boundaries between sensor input, scene understanding, motion estimation, spatial mapping, and ROS2 integration.

## Pipeline

```text
Camera input
      |
      v
Depth Estimation
      |
      v
Point Cloud Projection
      |
      v
(Optional) Pose / Visual Odometry
      |
      v
Map / Cloud Fusion
      |
      v
ROS2 Topic Publishing
```

## Flow

1. `src/io` ingests frames from webcam, video, simulator, or ROS2 sources.
2. `src/depth` loads a pretrained depth backend and converts RGB frames into dense depth estimates.
3. `src/mapping` back-projects depth into point clouds using camera intrinsics.
4. `src/slam` consumes image observations and emits pose or trajectory updates.
5. `src/mapping` can fuse successive clouds into larger spatial artifacts.
6. `src/ros2` publishes depth, pose, and point cloud outputs to the rest of the robot stack.

## Core Modules

- `src/main.py`: process orchestration, config loading, frame loop, and artifact export
- `src/io/camera.py`: unified frame source abstraction for webcam and file streams
- `src/io/ros_image.py`: ROS2 image topic adapter
- `src/depth/estimator.py`: backend-facing depth inference interface for MiDaS and Depth Anything
- `src/slam/wrapper.py`: pose integration boundary for visual odometry or SLAM
- `src/mapping/pointcloud.py`: depth back-projection, Open3D point cloud construction, and `.ply` export
- `src/ros2/nodes.py`: ROS2 bridge used to publish outputs and subscribe to images

## Design Goals

- Config-driven execution for local runs and simulator demos
- Replaceable depth and SLAM backends without rewriting the frame loop
- Clear geometry utilities to keep transforms and projection logic isolated
- Simple artifacts on disk for debugging and repeatable demos
