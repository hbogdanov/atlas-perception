from src.ros2.transforms import make_transform


def test_make_transform_writes_translation():
    matrix = make_transform((1.0, 2.0, 3.0))
    assert matrix[0, 3] == 1.0
    assert matrix[1, 3] == 2.0
    assert matrix[2, 3] == 3.0

