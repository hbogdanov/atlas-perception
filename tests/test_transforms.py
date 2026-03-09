import numpy as np
import pytest

from src.ros2.transforms import make_transform
from src.ros2.transforms import quaternion_to_rotation_matrix
from src.ros2.transforms import rotation_matrix_to_quaternion


def test_make_transform_writes_translation():
    matrix = make_transform((1.0, 2.0, 3.0))
    assert matrix[0, 3] == 1.0
    assert matrix[1, 3] == 2.0
    assert matrix[2, 3] == 3.0


def test_rotation_matrix_to_quaternion_identity():
    quat = rotation_matrix_to_quaternion(np.eye(3, dtype=np.float32))
    assert quat[0] == 0.0
    assert quat[1] == 0.0
    assert quat[2] == 0.0
    assert quat[3] == 1.0


def test_rotation_matrix_to_quaternion_z_rotation():
    rotation = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    quat = rotation_matrix_to_quaternion(rotation)
    assert quat[2] == pytest.approx(np.float32(np.sqrt(0.5)), rel=1e-6)
    assert quat[3] == pytest.approx(np.float32(np.sqrt(0.5)), rel=1e-6)


def test_quaternion_to_rotation_matrix_z_rotation():
    quat = np.array([0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)], dtype=np.float32)
    rotation = quaternion_to_rotation_matrix(quat)
    expected = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    assert np.allclose(rotation, expected, atol=1e-6)
