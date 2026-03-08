# ROS2 Topics

## Subscriptions

- `/camera/image_raw`: RGB input stream when `input.mode` is `ros2`

## Publications

- `/atlas/depth`: depth image or normalized depth payload
- `/atlas/pose`: current camera pose estimate
- `/atlas/pointcloud`: accumulated or frame-local point cloud projection

## Frames

- `ros2.frame_id` sets the published frame name for outputs.
- Transforms and pose conversions are handled in `src/ros2/transforms.py`.
