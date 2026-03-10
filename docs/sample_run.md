# Sample Run Artifacts

Use this checklist for an end-to-end run artifact set:

1. Capture one representative RGB input frame from the source stream.
2. Save one rendered depth output image from the same sequence.
3. Export the accumulated point cloud to `data/outputs/frame_cloud.ply`.
4. Open the cloud in Open3D or RViz and capture a screenshot.
5. Export trajectory and pose-graph artifacts for pose-aware mapping runs.

Suggested layout:

- `demo/screenshots/tum_rgb_frame.png`
- `demo/screenshots/tum_depth_map.png`
- `data/outputs/tum_demo/frame_cloud.ply`
- `demo/screenshots/pointcloud_vis.png`

Example run command:

```bash
python -m src.main --config configs/default.yaml --override-config configs/gazebo_demo.yaml --max-frames 100
```

That Gazebo config also writes a composite showcase video to:

- `demo/videos/gazebo_demo.mp4`

Example RTAB-Map-assisted run:

```bash
python -m src.main --config configs/default.yaml --override-config configs/gazebo_rtabmap_demo.yaml --max-frames 100
```

That RTAB-Map demo config writes:

- `demo/videos/gazebo_rtabmap_demo.mp4`

Ubuntu TurtleBot3 reference run:

```bash
python -m src.main --config configs/default.yaml --override-config configs/turtlebot3_gazebo_rtabmap.yaml --max-frames 300
```

That run is documented in `docs/ubuntu_gazebo_setup.md` and is intended to write:

- `demo/videos/turtlebot3_gazebo_rtabmap.mp4`

If `output.save_rgb_snapshot`, `output.save_depth_snapshot`, and `output.save_pointcloud` are enabled, the main pipeline will save:

- `rgb_frame.png`
- `depth_map.png`
- `frame_cloud.ply`
- `trajectory.npy`
- `trajectory.json`
- `trajectory.csv`
- `trajectory_plot.png`
- `pose_graph.json`
- `pose_graph_edges.csv`

If `mapping.representation: tsdf`, the pipeline also writes:

- `tsdf_mesh.ply`

With the default config, the saved depth output is post-processed before export. The current cleanup stages are:

- bilateral smoothing for speckle reduction
- RGB-guided refinement for sharper depth boundaries
- optional temporal fusion for reduced frame-to-frame flicker in video or ROS streams

If `output.save_demo_video` is enabled, the pipeline also writes a composite `.mp4` that includes:

- source camera feed
- depth visualization
- trajectory plot
- live status panel with pose, metrics, and ROS topic names

The composite video is generated directly from `src.main` through `src/utils/demo_video.py`.

Latest evaluated full-pipeline run on a TUM-derived 30-frame clip produced:

- `data/outputs/tum_main_eval/rgb_frame.png`
- `data/outputs/tum_main_eval/depth_map.png`
- `data/outputs/tum_main_eval/frame_cloud.ply`
- `data/outputs/tum_main_eval/trajectory.npy`
- `data/outputs/tum_main_eval/trajectory.json`
- `data/outputs/tum_main_eval/trajectory.csv`
- `data/outputs/tum_main_eval/trajectory_plot.png`
- `data/outputs/tum_main_eval/pose_graph.json`
- `data/outputs/tum_main_eval/pose_graph_edges.csv`
- `data/outputs/tum_main_eval/atlas_eval_demo.mp4`

One-command demo wrapper:

```bash
python tools/run_demo.py --dataset tum
```

That wrapper runs the TUM preset through `src.main` and exports:

- `demo/videos/tum_demo.mp4`
- `demo/gifs/tum_demo.gif`
- `demo/screenshots/tum_trajectory_plot.png`

The preset loops the short TUM source clip and uses `slam.mode: dummy`, so the exported trajectory visibly grows instead of staying fixed.

Live webcam showcase:

```bash
python run_webcam_mapping.py --show-cloud
```

That live entrypoint opens:

- a real-time RGB + depth + trajectory dashboard
- an optional Open3D point-cloud window

Use `--save-artifacts` if you want the live run to export `frame_cloud.ply` and trajectory files on exit.

Isaac Sim showcase:

```bash
python tools/run_isaac_demo.py --slam-mode dummy
```

Or through ROS2 launch:

```bash
ros2 launch launch/isaac_atlas.launch.py
```

That path binds Atlas to Isaac Sim RGB and `CameraInfo` topics and writes the standard Isaac artifact set under `data/outputs/isaac/`.

Quantitative depth evaluation can be run separately on a TUM RGB-D folder:

```bash
python tools/evaluate_depth.py --dataset-root data/samples/tum_freiburg1_xyz --limit 30
```

That evaluator writes:

- `data/outputs/depth_eval/tum_depth_eval.json`
- `data/outputs/depth_eval/tum_depth_eval.csv`

The reported summary includes:

- `AbsRel`
- `RMSE`
- `delta1`
- `FPS`

For relative monocular backends, the evaluator median-aligns each predicted depth map to the valid ground-truth pixels before computing those scores.

Trajectory evaluation can be run against TUM ground truth with the exported Atlas trajectory:

```bash
python tools/evaluate_trajectory.py --estimated-json data/outputs/tum_main_eval/trajectory.json --groundtruth-tum data/samples/tum_freiburg1_xyz/groundtruth.txt
```

That evaluator writes:

- `data/outputs/trajectory_eval/trajectory_eval.json`
- `data/outputs/trajectory_eval/trajectory_eval.csv`

The reported summary includes:

- `ATE` RMSE
- `RPE` translational RMSE
- `RPE` rotational RMSE in degrees

Observed runtime summary from that run:

- `avg_depth_ms=80.79`
- `avg_mapping_ms=3.18`
- `avg_fps=11.45`
- `points=100000`

If using Torch Hub backends for the first time, allow time for model download or provide `depth.local_weights_path` in config.

## Recommended Dataset

Use the TUM RGB-D `fr1/xyz` sequence first. TUM explicitly recommends the `xyz` series for first experiments, and `fr1/xyz` is the smaller starter sequence. Official sources:

- Dataset overview: https://cvg.cit.tum.de/data/datasets/rgbd-dataset
- Download page: https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download
- File formats: https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats

The TUM files are time-stamped PNGs, with depth maps stored as 16-bit PNG where a value of `5000` corresponds to `1 meter`.

## Exact First Demo

1. Download `fr1/xyz` from the official TUM download page.
2. Extract it under `data/samples/tum_freiburg1_xyz/`.
3. Pick one RGB frame from `rgb/`.
4. Run:

```bash
python tools/run_tum_artifact.py --rgb data/samples/tum_freiburg1_xyz/rgb/1305031102.175304.png --out-dir data/outputs/tum_demo
```

This will produce:

- `data/outputs/tum_demo/rgb_frame.png`
- `data/outputs/tum_demo/depth_map.png`
- `data/outputs/tum_demo/frame_cloud.ply`

If you want to compare raw backend output against refined output, disable `depth.postprocess.enabled` in an override config and rerun the same command.

You can also copy the first two into `demo/screenshots/` for README assets:

- `demo/screenshots/tum_rgb_frame.png`
- `demo/screenshots/tum_depth_map.png`

Then open the PLY and save:

- `demo/screenshots/pointcloud_vis.png`

## Viewer Screenshot

Example with Open3D:

```bash
python -c "import open3d as o3d; p=o3d.io.read_point_cloud('data/outputs/tum_demo/frame_cloud.ply'); o3d.visualization.draw_geometries([p])"
```
