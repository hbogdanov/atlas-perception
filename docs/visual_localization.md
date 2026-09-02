# Visual Landmark Localization

Atlas can estimate a camera pose from observed ArUco markers whose poses are
known in the map frame. This is a visual localization measurement for a
pre-mapped environment; it is not a replacement for visual odometry, loop
closure, or a full SLAM backend.

## Enable It

Set `visual_localization.enabled: true` and provide calibrated intrinsics plus
the marker map. Each landmark uses the marker's `T_world_marker` homogeneous
transform in meters:

```yaml
visual_localization:
  enabled: true
  dictionary: DICT_4X4_50
  marker_length_m: 0.16
  max_reprojection_error: 3.0
  landmarks:
    - id: 7
      T_world_marker:
        - [1.0, 0.0, 0.0, 2.0]
        - [0.0, 1.0, 0.0, 0.0]
        - [0.0, 0.0, 1.0, 0.8]
        - [0.0, 0.0, 0.0, 1.0]
```

Atlas detects configured markers, converts their corners into map-frame 3D
correspondences, and uses RANSAC PnP to recover `T_world_camera`. Accepted
measurements are published on the configured visual-pose ROS2 topic with a
reprojection-error-derived covariance.

## Pose Corrections

Set `visual_localization.pose_correction.apply_to_mapping: true` to use an
accepted visual pose as an opt-in mapping and trajectory correction. Atlas
rejects a correction when its timestamp, translation or rotation innovation,
or covariance exceeds the configured bounds. `blend_weight: 1.0` replaces the
incoming pose; lower values apply a bounded SE(3) blend. This is appropriate
for correcting an external tracker or controlled evaluation, not for filling
unobserved motion between landmarks.

## Evaluation Boundary

The included tests exercise the complete synthetic path from a rendered ArUco
marker through detection and PnP pose recovery. A real evaluation still needs
a calibrated camera, surveyed marker map, and a recorded or live scene with
visible markers. Report those results separately from Atlas's RGB-D mapping
and externally posed RTAB-Map workflows.
