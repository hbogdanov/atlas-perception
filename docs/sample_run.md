# Sample Run Artifacts

Use this checklist for an end-to-end run artifact set:

1. Capture one representative RGB input frame from the source stream.
2. Save one rendered depth output image from the same sequence.
3. Export the accumulated point cloud to `data/outputs/frame_cloud.ply`.
4. Open the cloud in Open3D or RViz and capture a screenshot.

Suggested layout:

- `demo/screenshots/tum_rgb_frame.png`
- `demo/screenshots/tum_depth_map.png`
- `data/outputs/tum_demo/frame_cloud.ply`
- `demo/screenshots/pointcloud_vis.png`

Example run command:

```bash
python -m src.main --config configs/default.yaml --override-config configs/gazebo_demo.yaml --max-frames 100
```

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
