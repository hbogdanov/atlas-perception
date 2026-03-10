# Documentation Index

Use these docs in this order:

- `README.md`: project overview, quickstart, artifact examples, and config entrypoints
- `docs/architecture.md`: pipeline stages, world-frame mapping flow, and simulator reference path
- `docs/ros_topics.md`: ROS2 subscriptions, publications, frame semantics, and SLAM topic contract
- `docs/sample_run.md`: reproducible artifact generation commands and expected outputs
- `docs/isaac_sim_setup.md`: Isaac Sim ROS2 topic contract and Atlas launch commands
- `docs/ubuntu_gazebo_setup.md`: Ubuntu 22.04 + ROS2 Humble + TurtleBot3 Gazebo setup for a real simulator-backed run

## Config Guide

The repository config split is:

- `configs/default.yaml`: safe baseline, ROS2 off, SLAM disabled
- `configs/tum_demo.yaml`: looping TUM showcase run with dummy pose integration and GIF/video export
- `configs/webcam_mapping.yaml`: live webcam mapping preset for laptop-camera demos
- `configs/tum_main_eval.yaml`: reproducible TUM evaluation and artifact export
- `configs/gazebo_demo.yaml`: Gazebo ROS2 demo with dummy trajectory
- `configs/gazebo_rtabmap_demo.yaml`: Gazebo ROS2 demo consuming external RTAB-Map pose
- `configs/turtlebot3_gazebo_rtabmap.yaml`: TurtleBot3 Gazebo reference run on Ubuntu
- `configs/isaac_demo.yaml`: Isaac Sim ROS2 demo with dummy trajectory

## Output Artifacts

Depending on config flags, Atlas can write:

- `rgb_frame.png`
- `depth_map.png`
- `semantic_overlay.png`
- `frame_cloud.ply`
- `semantic_cloud.ply`
- `tsdf_mesh.ply`
- `trajectory.npy`
- `trajectory.json`
- `trajectory.csv`
- `trajectory_plot.png`
- `pose_graph.json`
- `pose_graph_edges.csv`
- `atlas_demo.mp4`
- `tum_demo.mp4`
- `tum_demo.gif`
- `tum_depth_eval.json`
- `tum_depth_eval.csv`
- `trajectory_eval.json`
- `trajectory_eval.csv`

## Current Status

Atlas currently documents and supports:

- camera ingestion from webcam, video, and ROS2 image topics
- plugin-based depth backend selection through `depth.depth_model`, with built-in MiDaS and Depth Anything backends
- optional semantic segmentation through `semantics.backend`, with a YOLOv8 path for semantic overlays and semantic point clouds
- quantitative monocular depth evaluation on TUM RGB-D with `AbsRel`, `RMSE`, `delta1`, and FPS reporting
- quantitative trajectory evaluation with `ATE` and `RPE` against TUM-format ground truth
- stable local pytest execution through repo-local test temp paths
- GitHub Actions CI for `pytest`, `black --check`, and `ruff check`
- colored point cloud generation and `.ply` export
- selectable point-cloud or TSDF volumetric mapping
- ROS2 publishing for depth, pose, path, and point cloud outputs
- external RTAB-Map pose consumption through `slam.mode: rtabmap`
- pose-graph export with simple loop-closure constraints
- composite demo video generation from the main pipeline
- one-command live webcam mapping through `python run_webcam_mapping.py`
- dedicated Isaac Sim runner through `python tools/run_isaac_demo.py`

Atlas does not yet claim:

- calibrated metric monocular reconstruction by default
- a fully optimized built-in SLAM backend beyond dummy motion, pose-graph bookkeeping, and external RTAB-Map pose consumption
- deep simulator-specific integrations beyond topic and launch adaptation
