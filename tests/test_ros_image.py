import numpy as np

from src.io.ros_image import RosImageSubscriber


def test_get_camera_intrinsics_returns_none_without_camera_info():
    subscriber = RosImageSubscriber.__new__(RosImageSubscriber)
    subscriber._latest_camera_info = None
    assert subscriber.get_camera_intrinsics() is None


def test_camera_info_callback_extracts_intrinsics():
    subscriber = RosImageSubscriber.__new__(RosImageSubscriber)
    subscriber.camera_info_topic = "/camera/camera_info"
    subscriber._latest_camera_info = None

    message = type(
        "CameraInfo",
        (),
        {
            "header": type("Header", (), {"stamp": type("Stamp", (), {"sec": 2, "nanosec": 250_000_000})()})(),
            "k": [525.0, 0.0, 319.5, 0.0, 525.0, 239.5, 0.0, 0.0, 1.0],
        },
    )()

    subscriber._camera_info_callback(message)

    intrinsics = subscriber.get_camera_intrinsics()
    assert intrinsics == {"fx": 525.0, "fy": 525.0, "cx": 319.5, "cy": 239.5}
