# Atlas Experiments

Atlas records a self-contained run directory for every pipeline execution. The directory includes the resolved
`config.json`, `manifest.json`, `runtime_metrics.json`, and, for TUM RGB-D, `associations.csv`. Mapping runs also
produce their trajectory and point-cloud artifacts when those outputs are enabled.

## Reconstruction Ablations

Use the named TUM scenarios to separate depth error from pose error:

```bash
python tools/run_reconstruction_benchmark.py --scenario gt_depth_gt_pose --max-frames 120
python tools/run_reconstruction_benchmark.py --scenario estimated_depth_gt_pose --max-frames 120
python tools/run_reconstruction_benchmark.py --scenario gt_depth_perturbed_pose --max-frames 120
python tools/run_reconstruction_benchmark.py --scenario estimated_depth_perturbed_pose --max-frames 120
```

The `gt_depth_gt_pose` scenario is the mapping ceiling. The estimated-depth and perturbed-pose scenarios each alter
one input category at a time; the final scenario compounds both sources of error. Do not compare runs with different
frame counts, intrinsics, map settings, or depth-alignment methods.

Compile completed benchmark artifacts into a portfolio-ready technical report and runtime/map-quality figure:

```bash
python tools/compile_flagship_study.py
```

The report labels supplied ground-truth and external poses explicitly. It is a study compiler, not a claim that every
scenario uses autonomous visual SLAM.

Run controlled calibration or timestamp-latency sweeps with injected errors:

```bash
python tools/run_sensitivity_study.py --kind calibration --max-frames 60
python tools/run_sensitivity_study.py --kind latency --max-frames 60
```

These outputs quantify sensitivity to injected reconstruction intrinsics or delayed supplied poses. They are not a
substitute for a hardware calibration study.

For the final video, enable `output.save_demo_video: true`. Its camera tile labels depth confidence, landmark-PnP
measurement quality, and visual-pose correction status alongside the depth and top-down map tiles.

## Depth Robustness

The robustness runner evaluates a clean baseline and deterministic brightness, noise, blur, resolution, and occlusion
conditions. It writes a JSON table, CSV table, and experiment `manifest.json` beside the chosen output JSON.

```bash
python tools/run_depth_robustness.py --dataset-root data/samples/tum_freiburg1_xyz --limit 30
```

Use `--alignment raw`, `--alignment median_scale`, and `--alignment scale_shift` as separate experiments. Relative
monocular outputs are not metric measurements unless the alignment method is stated beside the result.
