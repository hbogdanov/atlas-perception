import pytest

from src.io.camera import create_frame_source


class FakeRosSource:
    def __init__(self, topic: str, timeout_sec: float) -> None:
        self.topic = topic
        self.timeout_sec = timeout_sec


def test_create_frame_source_uses_ros2_subscriber(monkeypatch):
    created = {}

    def fake_factory(topic: str, timeout_sec: float):
        created["topic"] = topic
        created["timeout_sec"] = timeout_sec
        return FakeRosSource(topic, timeout_sec)

    monkeypatch.setattr("src.io.camera.RosImageSubscriber", fake_factory)

    source = create_frame_source({"mode": "ros2", "source": "/camera/image_raw", "timeout_sec": 2.5})

    assert isinstance(source, FakeRosSource)
    assert created == {"topic": "/camera/image_raw", "timeout_sec": 2.5}


def test_create_frame_source_rejects_unknown_mode():
    with pytest.raises(ValueError):
        create_frame_source({"mode": "unknown"})
