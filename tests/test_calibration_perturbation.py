import pytest

from src.evaluation.calibration_perturbation import CalibrationPerturber


def test_calibration_perturber_changes_only_the_reconstruction_intrinsics():
    perturber = CalibrationPerturber(
        {"enabled": True, "fx_scale": 1.1, "fy_scale": 0.9, "cx_offset_px": 3.0, "cy_offset_px": -2.0}
    )

    perturbed = perturber.perturb({"fx": 100.0, "fy": 200.0, "cx": 50.0, "cy": 60.0})

    assert perturbed["fx"] == pytest.approx(110.0)
    assert perturbed["fy"] == pytest.approx(180.0)
    assert perturbed["cx"] == pytest.approx(53.0)
    assert perturbed["cy"] == pytest.approx(58.0)
