from src.sim.factory import create_sim_bridge


def test_gazebo_bridge_applies_camera_topic_to_runtime_config():
    config = {
        "input": {"mode": "webcam", "source": 0},
        "ros2": {"enabled": False},
        "sim": {
            "platform": "gazebo",
            "namespace": "/gazebo",
            "camera_topic": "/gazebo/camera/image_raw",
            "camera_info_topic": "/gazebo/camera/camera_info",
            "clock_topic": "/clock",
            "launch_package": "gazebo_ros",
            "launch_file": "gazebo.launch.py",
        },
    }
    bridge = create_sim_bridge(config["sim"])
    merged = bridge.apply(config)
    assert merged["input"]["mode"] == "ros2"
    assert merged["input"]["source"] == "/gazebo/camera/image_raw"
    assert merged["input"]["camera_info_topic"] == "/gazebo/camera/camera_info"
    assert merged["ros2"]["enabled"] is True


def test_isaac_bridge_exposes_launch_arguments():
    bridge = create_sim_bridge(
        {
            "platform": "isaac",
            "namespace": "/isaac",
            "camera_topic": "/isaac/camera/color",
            "camera_info_topic": "/isaac/camera/camera_info",
            "clock_topic": "/clock",
            "launch_package": "isaac_ros_launch",
            "launch_file": "isaac_sim.launch.py",
        }
    )
    arguments = bridge.launch_arguments()
    assert arguments["sim_platform"] == "isaac"
    assert arguments["camera_topic"] == "/isaac/camera/color"
