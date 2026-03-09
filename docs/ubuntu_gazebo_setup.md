# Ubuntu Gazebo Setup

Use this path for a real simulator-backed Atlas run on Ubuntu 22.04 with ROS2 Humble, Gazebo, and TurtleBot3.

## Install ROS2 Humble

```bash
sudo apt update
sudo apt install ros-humble-desktop
source /opt/ros/humble/setup.bash
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
```

## Install Gazebo and TurtleBot3

```bash
sudo apt install ros-humble-gazebo-ros-pkgs
sudo apt install ros-humble-turtlebot3*
export TURTLEBOT3_MODEL=burger
echo "export TURTLEBOT3_MODEL=burger" >> ~/.bashrc
```

## Launch TurtleBot3 Gazebo

```bash
ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py
```

Atlas expects a camera-equipped robot. Verify the actual topic names before running Atlas:

```bash
ros2 topic list
ros2 topic hz /camera/image_raw
ros2 run rqt_image_view rqt_image_view
```

The intended topics are:

- `/camera/image_raw`
- `/camera/camera_info`
- `/odom`
- `/tf`

If your TurtleBot3 launch publishes different camera topic names, update the Atlas override config accordingly.

## Run Atlas

From the Atlas repo on the Ubuntu machine:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .[dev]
python -m src.main --config configs/default.yaml --override-config configs/turtlebot3_gazebo_rtabmap.yaml --max-frames 300
```

That run is intended to:

- subscribe to the simulator RGB feed
- ingest live `CameraInfo` intrinsics
- estimate depth
- consume external RTAB-Map pose on `/rtabmap/localization_pose`
- accumulate a world-aligned point cloud
- publish Atlas ROS2 outputs
- write a composite showcase video

## Optional RTAB-Map Pose Source

If you want Atlas to consume real external pose updates instead of dummy motion, start an RTAB-Map ROS2 node separately and verify:

```bash
ros2 topic echo /rtabmap/localization_pose
```

Atlas `slam.mode: rtabmap` depends on that topic being live.

## Config Used

The reference override config is:

- `configs/turtlebot3_gazebo_rtabmap.yaml`

Key settings in that config:

- `input.source: /camera/image_raw`
- `input.camera_info_topic: /camera/camera_info`
- `slam.mode: rtabmap`
- `slam.pose_topic: /rtabmap/localization_pose`
- `ros2.frame_id: turtlebot3_camera`
- `output.demo_video_path: demo/videos/turtlebot3_gazebo_rtabmap.mp4`

## Expected Atlas Outputs

The TurtleBot3 Gazebo override writes:

- `data/outputs/turtlebot3_gazebo/rgb_frame.png`
- `data/outputs/turtlebot3_gazebo/depth_map.png`
- `data/outputs/turtlebot3_gazebo/frame_cloud.ply`
- `data/outputs/turtlebot3_gazebo/trajectory.npy`
- `data/outputs/turtlebot3_gazebo/trajectory.json`
- `data/outputs/turtlebot3_gazebo/trajectory.csv`
- `data/outputs/turtlebot3_gazebo/trajectory_plot.png`
- `demo/videos/turtlebot3_gazebo_rtabmap.mp4`

## RViz Check

Open RViz on the same ROS graph:

```bash
rviz2
```

Add these displays:

- `Image`
- `Pose`
- `Path`
- `PointCloud2`

Atlas publishes:

- `/atlas/depth`
- `/atlas/pose`
- `/atlas/path`
- `/atlas/pointcloud`
