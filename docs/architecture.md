# Architecture

Atlas Perception is organized as a modular perception pipeline with explicit boundaries between sensor input, scene understanding, trajectory hooks, spatial mapping, and ROS2 integration.

## Pipeline

```text
Camera input
      |
      v
Depth Estimation
      |
      +--> (Optional) Trajectory / Future SLAM Backend
      |
      v
Point Cloud Projection + Transform
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
3. `src/slam` consumes observations and emits `T_world_camera` from `disabled`, `dummy`, or `rtabmap` backends.
4. `src/mapping` back-projects depth into camera-frame point clouds using camera intrinsics.
5. `src/mapping` transforms those points into the world frame and fuses successive clouds into a larger map.
6. `src/ros2` publishes depth, pose, path, and point cloud outputs to the rest of the robot stack.
7. `src/sim` normalizes simulator camera topics and launch settings for Isaac Sim and Gazebo runs.
8. `src/utils/demo_video.py` can render a composite demo artifact from the same run, combining RGB, depth, trajectory, and output status.

## Gazebo Reference Flow

The cleanest real simulator path is:

1. launch TurtleBot3 Gazebo on Ubuntu with ROS2 Humble
2. verify `/camera/image_raw` and `/camera/camera_info`
3. run Atlas with `configs/turtlebot3_gazebo_rtabmap.yaml`
4. consume external RTAB-Map pose on `/rtabmap/localization_pose`
5. accumulate the world-frame cloud and publish `/atlas/depth`, `/atlas/pose`, `/atlas/path`, and `/atlas/pointcloud`
6. write `demo/videos/turtlebot3_gazebo_rtabmap.mp4`

## Core Modules

- `src/main.py`: process orchestration, config loading, frame loop, and artifact export
- `src/io/camera.py`: unified frame source abstraction for webcam, file, and ROS2 streams
- `src/io/ros_image.py`: ROS2 image topic subscriber backed by `rclpy` and `cv_bridge`
- `src/depth/estimator.py`: backend-facing depth inference interface for MiDaS and Depth Anything
- `src/slam/wrapper.py`: backend contract plus `disabled`, `dummy`, and `rtabmap` trajectory integrations
- `src/mapping/pointcloud.py`: camera-frame back-projection, world-frame accumulation, and export adapters
- `src/ros2/nodes.py`: ROS2 bridge used to publish depth, pose, path, and colored point clouds
- `src/sim/common.py`: simulator bridge contract for runtime topic adaptation and launch arguments
- `src/utils/demo_video.py`: composite video renderer for showcase outputs written from the main pipeline

## Artifacts

The main run path can write:

- RGB snapshot
- depth visualization
- accumulated point cloud `.ply`
- trajectory array, JSON, CSV, and XY plot
- composite demo `.mp4`

Those artifacts are produced from `src/main.py` based on the `output.*` config flags rather than a separate post-processing pass.

## Design Goals

- Config-driven execution for local runs and simulator demos
- Replaceable depth and SLAM backends without rewriting the frame loop
- Clear geometry utilities to keep transforms and projection logic isolated
- Simple artifacts on disk for debugging and repeatable demos
