from src.utils.config import load_config


def test_load_config():
    config = load_config("configs/default.yaml")
    assert config["input"]["mode"] == "webcam"
    assert config["ros2"]["depth_topic"] == "/atlas/depth"
    assert config["camera"]["fx"] == 525.0
