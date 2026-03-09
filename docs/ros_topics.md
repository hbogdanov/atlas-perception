# ROS2 Topics

## Subscriptions

- `input.source`: RGB input stream when `input.mode` is `ros2`; simulator configs bind this to the Isaac or Gazebo camera topic

## Publications

- `/atlas/depth`: depth image payload, either normalized relative depth or raw backend output depending on `depth.output_mode`
- `/atlas/pose`: current camera pose estimate with orientation derived from the 4x4 pose matrix; dummy SLAM modes currently emit identity rotation
- `/atlas/pointcloud`: accumulated or frame-local point cloud projection

## Frames

- `ros2.frame_id` sets the published frame name for outputs.
- Transforms and pose conversions are handled in `src/ros2/transforms.py`.
