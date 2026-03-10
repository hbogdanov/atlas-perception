# Pipeline

This document explains the Atlas Perception runtime from RGB input to mapped output.

## Overview

Atlas runs a perception pipeline with five core stages:

1. depth inference
2. projection
3. fusion
4. mapping
5. trajectory estimation

At runtime, those stages are driven per frame, then fused across time.

## 1. Depth Inference

Input:

- RGB frame from webcam, video, ROS2, Gazebo, or Isaac Sim

Process:

- `src/depth/estimator.py` loads the configured depth backend
- current built-in backends are MiDaS and Depth Anything
- optional post-processing can apply:
  - bilateral filtering
  - guided refinement
  - temporal fusion

Output:

- dense depth map for the current frame

Why it matters:

- this is the geometry source for all downstream 3D reconstruction

## 2. Projection

Input:

- depth map
- RGB frame
- camera intrinsics `fx`, `fy`, `cx`, `cy`

Process:

- `src/utils/geometry.py` back-projects pixels into 3D camera-frame points
- `src/mapping/pointcloud.py` samples the frame using the configured stride
- per-point color comes from:
  - RGB by default
  - semantic class coloring when semantic fusion is enabled

Output:

- camera-frame point cloud for the current frame

Why it matters:

- this is the bridge from image-space perception to robot-space geometry

## 3. Fusion

Input:

- current frame point cloud
- current pose estimate `T_world_camera`

Process:

- Atlas transforms points from camera frame into world frame
- then fuses them using one of two representations:
  - point-cloud accumulation
  - TSDF volumetric fusion through Open3D

Optional semantic fusion:

- `src/semantics` can run YOLOv8 segmentation
- sampled semantic labels are attached to 3D points
- semantic point clouds can be exported as `semantic_cloud.ply`

Output:

- fused world-aligned geometry over time

Why it matters:

- this is what turns single-frame perception into a usable map

## 4. Mapping

Mapping in Atlas is pose-aware.

Process:

- each frame contributes new 3D structure
- the map grows in the world frame rather than staying camera-local
- exports can include:
  - `frame_cloud.ply`
  - `semantic_cloud.ply`
  - `tsdf_mesh.ply`

Supported mapping modes:

- `mapping.representation: pointcloud`
- `mapping.representation: tsdf`

Why it matters:

- this is the part robotics teams care about operationally: persistent spatial structure, not just per-frame depth

## 5. Trajectory Estimation

Atlas treats trajectory as a first-class mapping input.

Process:

- `src/slam/wrapper.py` provides the pose source
- supported modes are:
  - `disabled`
  - `dummy`
  - `rtabmap`
- pose graph bookkeeping records:
  - pose nodes
  - odometry edges
  - simple loop-closure edges

Exports:

- `trajectory.npy`
- `trajectory.json`
- `trajectory.csv`
- `trajectory_plot.png`
- `pose_graph.json`
- `pose_graph_edges.csv`

Why it matters:

- without trajectory, you only have frame-local geometry
- with trajectory, Atlas can accumulate geometry into a coherent map

## End-to-End Flow

Per frame, Atlas does:

1. ingest RGB
2. infer depth
3. optionally infer semantics
4. estimate or consume pose
5. project depth into 3D
6. transform into world coordinates
7. fuse into the global map
8. publish ROS2 outputs
9. optionally export artifacts

## Result

Atlas is not just a monocular depth demo.

It is a perception stack that can produce:

- depth
- semantic overlays
- semantic point clouds
- trajectories
- pose graphs
- fused geometric maps

That is the repo story a robotics reader should get immediately.
