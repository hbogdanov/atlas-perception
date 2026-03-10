# Documentation Index

Use these docs in this order:

- `README.md`: project overview, quickstart, artifact examples, and config entrypoints
- `docs/architecture.md`: pipeline stages, world-frame mapping flow, and simulator reference path
- `docs/ros_topics.md`: ROS2 subscriptions, publications, frame semantics, and SLAM topic contract
- `docs/sample_run.md`: reproducible artifact generation commands and expected outputs
- `docs/ubuntu_gazebo_setup.md`: Ubuntu 22.04 + ROS2 Humble + TurtleBot3 Gazebo setup for a real simulator-backed run

## Config Guide

The repository config split is:

- `configs/default.yaml`: safe baseline, ROS2 off, SLAM disabled
- `configs/tum_main_eval.yaml`: reproducible TUM evaluation and artifact export
- `configs/gazebo_demo.yaml`: Gazebo ROS2 demo with dummy trajectory
- `configs/gazebo_rtabmap_demo.yaml`: Gazebo ROS2 demo consuming external RTAB-Map pose
- `configs/turtlebot3_gazebo_rtabmap.yaml`: TurtleBot3 Gazebo reference run on Ubuntu
- `configs/isaac_demo.yaml`: Isaac Sim ROS2 demo with dummy trajectory

## Output Artifacts

Depending on config flags, Atlas can write:

- `rgb_frame.png`
- `depth_map.png`
- `frame_cloud.ply`
- `trajectory.npy`
- `trajectory.json`
- `trajectory.csv`
- `trajectory_plot.png`
- `atlas_demo.mp4`
- `tum_depth_eval.json`
- `tum_depth_eval.csv`
- `trajectory_eval.json`
- `trajectory_eval.csv`

## Current Status

Atlas currently documents and supports:

- camera ingestion from webcam, video, and ROS2 image topics
- MiDaS and Depth Anything depth backends
- quantitative monocular depth evaluation on TUM RGB-D with `AbsRel`, `RMSE`, `delta1`, and FPS reporting
- quantitative trajectory evaluation with `ATE` and `RPE` against TUM-format ground truth
- stable local pytest execution through repo-local test temp paths
- colored point cloud generation and `.ply` export
- ROS2 publishing for depth, pose, path, and point cloud outputs
- external RTAB-Map pose consumption through `slam.mode: rtabmap`
- composite demo video generation from the main pipeline

Atlas does not yet claim:

- calibrated metric monocular reconstruction by default
- a built-in SLAM backend beyond dummy motion and external RTAB-Map pose consumption
- deep simulator-specific integrations beyond topic and launch adaptation
