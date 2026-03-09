# ROS2 Topics

## Subscriptions

- `input.source`: RGB input stream when `input.mode` is `ros2`; simulator configs bind this to the Isaac or Gazebo camera topic

## Publications

- `/atlas/depth`: depth image payload with `header.stamp` and `header.frame_id`; content depends on `depth.output_mode`
- `/atlas/pose`: current trajectory or pose-estimate output with orientation derived from the 4x4 pose matrix
- `/atlas/path`: accumulated trajectory published from the current SLAM backend state
- `/atlas/pointcloud`: accumulated or frame-local colored point cloud projection with `header.stamp` and `header.frame_id`

## Frames

- `ros2.frame_id` sets the published frame name for outputs.
- Transforms and pose conversions are handled in `src/ros2/transforms.py`.

## SLAM Notes

- `slam.mode: disabled` keeps `T_world_camera` at identity.
- `slam.mode: dummy` produces synthetic forward motion for testing map accumulation.
- `slam.mode: rtabmap` expects an external RTAB-Map ROS2 node publishing `slam.pose_topic` and Atlas consumes that pose for world-frame mapping.
