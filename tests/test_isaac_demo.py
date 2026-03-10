from tools.run_isaac_demo import build_command, build_runtime_config


def _args(**overrides):
    values = {
        "config": "configs/default.yaml",
        "override_config": "configs/isaac_demo.yaml",
        "camera_topic": "/isaac/camera/color",
        "camera_info_topic": "/isaac/camera/camera_info",
        "max_frames": 25,
        "slam_mode": "dummy",
    }
    values.update(overrides)
    return type("Args", (), values)()


def test_build_runtime_config_applies_isaac_topics_and_slam_mode():
    config = build_runtime_config(_args(camera_topic="/rgb", camera_info_topic="/camera_info", slam_mode="rtabmap"))
    assert config["input"]["mode"] == "ros2"
    assert config["input"]["source"] == "/rgb"
    assert config["input"]["camera_info_topic"] == "/camera_info"
    assert config["sim"]["platform"] == "isaac"
    assert config["slam"]["mode"] == "rtabmap"


def test_build_command_uses_src_main_with_runtime_override():
    command = build_command(_args())
    assert command[1:4] == ["-m", "src.main", "--config"]
    assert command[-2:] == ["--max-frames", "25"]
    assert "isaac_runtime.yaml" in command[6]
