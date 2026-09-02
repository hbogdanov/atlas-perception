import pytest

from src.io.tum_rgbd import associate_tum_entries


def test_tum_association_is_timestamp_based_and_one_to_one():
    rgb = [(1.000, "rgb/1.png"), (1.020, "rgb/2.png"), (2.000, "rgb/3.png")]
    depth = [(1.010, "depth/a.png"), (2.015, "depth/b.png")]

    pairs, report = associate_tum_entries(rgb, depth, tolerance=0.03)

    assert pairs == [(1.000, "rgb/1.png", 1.010, "depth/a.png"), (2.000, "rgb/3.png", 2.015, "depth/b.png")]
    assert report.matched_pairs == 2
    assert report.unmatched_source == 1
    assert report.unmatched_target == 0
    assert report.mean_timestamp_error == pytest.approx(0.0125)
    assert report.max_timestamp_error == pytest.approx(0.015)
