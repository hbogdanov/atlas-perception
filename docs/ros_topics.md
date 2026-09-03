# ROS2 Topics

## Subscriptions

- `input.source`: RGB input stream when `input.mode` is `ros2`; simulator configs bind this to the Isaac or Gazebo camera topic
- `input.camera_info_topic`: optional `sensor_msgs/CameraInfo` stream used to update `fx`, `fy`, `cx`, and `cy` for ROS2 and simulator workflows
- `slam.pose_topic`: external pose input when `slam.mode` is `rtabmap`; the TurtleBot3 Gazebo reference setup uses `/rtabmap/localization_pose`

Default Isaac Sim binding:

- RGB: `/isaac/camera/color`
- CameraInfo: `/isaac/camera/camera_info`

Use `python tools/run_isaac_demo.py` or `ros2 launch launch/isaac_atlas.launch.py` to attach Atlas to those topics directly.

## Publications

- `/atlas/depth`: depth image payload with `header.stamp` and `header.frame_id`; content depends on `depth.output_mode`
- `/atlas/pose`: current trajectory or pose-estimate output with orientation derived from the 4x4 pose matrix
- `/atlas/path`: accumulated trajectory published from the current SLAM backend state
- `/atlas/pointcloud`: accumulated or frame-local colored point cloud projection with `header.stamp` and `header.frame_id`
  When semantics are enabled, the point cloud rows also include a `label` field for the sampled semantic class id.

## Header Semantics

Atlas uses consistent ROS headers across published messages:

- `header.stamp`: frame timestamp propagated from the current source frame when available
- `ros2.camera_frame_id`: frame for depth images
- `ros2.map_frame_id`: frame for world-aligned pose, path, point-cloud, and visual-pose outputs

This applies to:

- depth publications
- pose publications
- path publications
- point cloud publications

## Frames

- Atlas publishes depth in `camera_frame_id`; it publishes `T_world_camera` poses and world-aligned maps in `map_frame_id`.
- Transforms and pose conversions are handled in `src/ros2/transforms.py`.

## Intrinsics

Atlas can operate with:

- static intrinsics from `camera.fx`, `camera.fy`, `camera.cx`, `camera.cy`
- live intrinsics from `input.camera_info_topic` when running in ROS2 or simulator mode

When `CameraInfo` is present, Atlas updates intrinsics before mapping, RGB-D VO, appearance-loop verification, and landmark PnP.

## SLAM Notes

- `slam.mode: disabled` keeps `T_world_camera` at identity.
- `slam.mode: dummy` produces synthetic forward motion for testing map accumulation.
- `slam.mode: rtabmap` consumes `PoseWithCovarianceStamped` from `slam.pose_topic` by default. Set `slam.pose_message_type: pose_stamped` only for a custom source that publishes `PoseStamped`.
- `slam.mode: rgbd_vo` runs an in-process metric RGB-D visual-odometry frontend. It is appropriate only when the incoming depth is metric and calibrated; it publishes Atlas poses using the same map-frame convention.
- `rgbd_vo` can perform appearance-verified RGB-D loop closure and a global translation-only graph correction. It can relocalize only from a configured known landmark measurement, not arbitrary place recognition. Rotation optimization and map reintegration after a loop are still missing; use RTAB-Map when those mature capabilities are required.
- The Ubuntu TurtleBot3 Gazebo reference config is `configs/turtlebot3_gazebo_rtabmap.yaml`.
- The Isaac Sim reference config is `configs/isaac_demo.yaml`.
