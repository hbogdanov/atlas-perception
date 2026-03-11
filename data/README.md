# Data Layout

Raw datasets are not committed to this repository.

Atlas uses three data tiers locally:

- `data/raw/`: original downloaded datasets, never edited
- `data/processed/`: derived subsets, converted clips, resized frames, and other intermediate assets
- `data/outputs/`: pipeline artifacts such as point clouds, trajectories, screenshots, and evaluation files

Expected local structure:

```text
data/
  raw/
    tum/
      fr1_desk/
      fr1_room/
      fr3_long_office_household/
  processed/
    tum/
      fr1_desk/
      fr1_room/
      fr3_long_office_household/
  outputs/
    tum_demo_eval/
```

Archived-but-unused local sequences can be kept under:

```text
data/raw/tum/archive/
```

This keeps the working demo set separate from extra downloads without tracking any of it in git.

## TUM RGB-D Download

Official dataset pages:

- https://cvg.cit.tum.de/data/datasets/rgbd-dataset
- https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download
- https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats

## Demo Sequences Used

The polished demo set for this repo uses:

- `fr1_desk`: close-range tabletop / local fusion showcase
- `fr3_long_office_household`: larger indoor mapping showcase

Optional backup sequence:

- `fr1_room`

Archived local extras:

- `fr1_desk2`
- `fr2_desk`

## Processing / Conversion Scripts

Useful local scripts:

- `tools/run_demo.py`: runs the main TUM metric demo preset and exports the tracked hero assets
- `tools/generate_tum_dataset_showcase.py`: generates Freiburg comparison demos from local TUM RGB-D folders
- `tools/run_tum_artifact.py`: one-frame artifact export from a TUM RGB input

The current metric demo path runs directly from RGB-D folders and does not require converting the sequence into MP4 first.

Use `data/processed/` for any future derived subsets or converted clips you choose to create.
