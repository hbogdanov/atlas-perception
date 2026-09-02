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

## Depth Robustness

The robustness runner evaluates a clean baseline and deterministic brightness, noise, blur, resolution, and occlusion
conditions. It writes a JSON table, CSV table, and experiment `manifest.json` beside the chosen output JSON.

```bash
python tools/run_depth_robustness.py --dataset-root data/samples/tum_freiburg1_xyz --limit 30
```

Use `--alignment raw`, `--alignment median_scale`, and `--alignment scale_shift` as separate experiments. Relative
monocular outputs are not metric measurements unless the alignment method is stated beside the result.
