# Flagship RGB-D VO Run

The flagship artifact uses real TUM `fr1_xyz` RGB-D frames and Atlas's owned `rgbd_vo` pose source. It does not use dataset ground-truth poses for tracking or mapping.

## Inputs

Place the TUM `fr1_xyz` sequence at `rgbd_dataset_freiburg1_xyz/`. Atlas accepts the usual nested archive layout:

```text
rgbd_dataset_freiburg1_xyz/
  rgbd_dataset_freiburg1_xyz/
    rgb.txt
    depth.txt
    groundtruth.txt
    rgb/
    depth/
```

## Generate the Demo

```bash
python tools/run_demo.py --dataset rgbd_vo
```

This runs 60 frames with metric input depth, ORB/depth/PnP visual odometry, and bounded voxel fusion. It produces:

- `demo/videos/atlas_rgbd_vo_flagship.mp4`
- `demo/gifs/atlas_rgbd_vo_flagship.gif`
- `data/outputs/flagship_vo_demo/trajectory.json`
- `data/outputs/flagship_vo_demo/trajectory_plot.png`

The GIF has three panels: TUM RGB-D input, estimated trajectory, and world-space voxel reconstruction. Its header explicitly states `Pose source: Atlas RGB-D VO`.

## Evaluate the Trajectory

```bash
python tools/evaluate_trajectory.py --estimated-json data/outputs/flagship_vo_demo/trajectory.json --groundtruth-tum rgbd_dataset_freiburg1_xyz/rgbd_dataset_freiburg1_xyz/groundtruth.txt
```

ATE is reported after rigid trajectory alignment. Translation and rotation RPE use consecutive relative motions. See [experiments.md](experiments.md) for the checked-in 120-frame `fr1_xyz` and `fr2_xyz` evidence bundles.

## Boundary

This is a visual-odometry artifact. It does not demonstrate a real revisit loop closure. Atlas's appearance-loop path has synthetic regression coverage, but a published real-revisit comparison remains future evaluation.
