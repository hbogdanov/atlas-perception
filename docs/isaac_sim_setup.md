# Isaac Sim Setup

Use this path when you want Atlas Perception to consume a live Isaac Sim RGB stream through ROS2.

## Goal

Pipeline:

`Isaac Sim -> RGB topic -> Atlas Perception -> depth -> map`

Atlas expects:

- an RGB image topic
- a matching `CameraInfo` topic
- optional external pose if you want `slam.mode: rtabmap`

## Expected Isaac Topics

Default Atlas Isaac config:

- RGB: `/isaac/camera/color`
- CameraInfo: `/isaac/camera/camera_info`

If your Isaac scene publishes different topics, pass them explicitly to the runner or launch file.

## Run Atlas Against Isaac Sim

Python entrypoint:

```bash
python tools/run_isaac_demo.py
```

ROS2 launch entrypoint:

```bash
ros2 launch launch/isaac_atlas.launch.py
```

Both use:

- `configs/default.yaml`
- `configs/isaac_demo.yaml`

## Common Variants

Use dummy motion so the map visibly accumulates even without an external pose source:

```bash
python tools/run_isaac_demo.py --slam-mode dummy
```

Use external RTAB-Map pose:

```bash
python tools/run_isaac_demo.py --slam-mode rtabmap
```

Override Isaac topics:

```bash
python tools/run_isaac_demo.py --camera-topic /rgb --camera-info-topic /camera_info
```

## Outputs

The Isaac override writes:

- `data/outputs/isaac/rgb_frame.png`
- `data/outputs/isaac/depth_map.png`
- `data/outputs/isaac/frame_cloud.ply`
- `data/outputs/isaac/trajectory.npy`
- `data/outputs/isaac/trajectory.json`
- `data/outputs/isaac/trajectory.csv`
- `data/outputs/isaac/trajectory_plot.png`
- `data/outputs/isaac/pose_graph.json`
- `data/outputs/isaac/pose_graph_edges.csv`
- `demo/videos/isaac_demo.mp4`

## ROS2 Notes

Atlas subscribes to:

- `input.source`
- `input.camera_info_topic`

Atlas publishes:

- `/atlas/depth`
- `/atlas/pose`
- `/atlas/path`
- `/atlas/pointcloud`

If Isaac Sim is bridged into ROS2 correctly, Atlas can attach directly to those camera topics without additional adapter code.
