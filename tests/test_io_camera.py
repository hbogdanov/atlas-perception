import pytest

from src.io.camera import create_frame_source


class FakeRosSource:
    def __init__(self, topic: str, camera_info_topic: str | None, timeout_sec: float) -> None:
        self.topic = topic
        self.camera_info_topic = camera_info_topic
        self.timeout_sec = timeout_sec


def test_create_frame_source_uses_ros2_subscriber(monkeypatch):
    created = {}

    def fake_factory(topic: str, camera_info_topic: str | None, timeout_sec: float):
        created["topic"] = topic
        created["camera_info_topic"] = camera_info_topic
        created["timeout_sec"] = timeout_sec
        return FakeRosSource(topic, camera_info_topic, timeout_sec)

    monkeypatch.setattr("src.io.camera.RosImageSubscriber", fake_factory)

    source = create_frame_source(
        {
            "mode": "ros2",
            "source": "/camera/image_raw",
            "camera_info_topic": "/camera/camera_info",
            "timeout_sec": 2.5,
        }
    )

    assert isinstance(source, FakeRosSource)
    assert created == {
        "topic": "/camera/image_raw",
        "camera_info_topic": "/camera/camera_info",
        "timeout_sec": 2.5,
    }


def test_create_frame_source_rejects_unknown_mode():
    with pytest.raises(ValueError):
        create_frame_source({"mode": "unknown"})


def test_create_frame_source_passes_video_loop_flag(monkeypatch):
    created = {}

    def fake_video_source(path: str, loop: bool = False):
        created["path"] = path
        created["loop"] = loop
        return object()

    monkeypatch.setattr("src.io.camera.VideoFrameSource", fake_video_source)

    create_frame_source({"mode": "video", "source": "demo.mp4", "loop": True})

    assert created == {"path": "demo.mp4", "loop": True}


def test_create_frame_source_uses_tum_rgbd_source(monkeypatch):
    created = {}

    def fake_tum_source(path: str, tolerance: float = 0.03):
        created["path"] = path
        created["tolerance"] = tolerance
        return object()

    monkeypatch.setattr("src.io.camera.TumRgbdFrameSource", fake_tum_source)

    create_frame_source(
        {
            "mode": "rgbd_dataset",
            "source": "data/samples/tum_freiburg1_xyz",
            "association_tolerance": 0.05,
        }
    )

    assert created == {"path": "data/samples/tum_freiburg1_xyz", "tolerance": 0.05}
