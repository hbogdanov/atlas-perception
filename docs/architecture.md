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
7. `src/sim` normalizes simulator camera topics and launch settings for Isaac Sim and Gazebo runs.

## Core Modules

- `src/main.py`: process orchestration, config loading, frame loop, and artifact export
- `src/io/camera.py`: unified frame source abstraction for webcam, file, and ROS2 streams
- `src/io/ros_image.py`: ROS2 image topic subscriber backed by `rclpy` and `cv_bridge`
- `src/depth/estimator.py`: backend-facing depth inference interface for MiDaS and Depth Anything
- `src/slam/wrapper.py`: explicit pose modes for disabled, dummy, and backend SLAM integrations
- `src/mapping/pointcloud.py`: depth back-projection, point cloud storage, and export adapters
- `src/ros2/nodes.py`: ROS2 bridge used to publish outputs and subscribe to images
- `src/sim/common.py`: simulator bridge contract for runtime topic adaptation and launch arguments

## Design Goals

- Config-driven execution for local runs and simulator demos
- Replaceable depth and SLAM backends without rewriting the frame loop
- Clear geometry utilities to keep transforms and projection logic isolated
- Simple artifacts on disk for debugging and repeatable demos
