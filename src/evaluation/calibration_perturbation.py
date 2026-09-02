from __future__ import annotations


class CalibrationPerturber:
    """Apply an explicit reconstruction-camera calibration error for controlled studies."""

    def __init__(self, config: dict | None = None) -> None:
        settings = config or {}
        self.enabled = bool(settings.get("enabled", False))
        self.fx_scale = float(settings.get("fx_scale", 1.0))
        self.fy_scale = float(settings.get("fy_scale", 1.0))
        self.cx_offset_px = float(settings.get("cx_offset_px", 0.0))
        self.cy_offset_px = float(settings.get("cy_offset_px", 0.0))

    def perturb(self, intrinsics: dict) -> dict:
        values = dict(intrinsics)
        if not self.enabled:
            return values
        values["fx"] = float(values["fx"]) * self.fx_scale
        values["fy"] = float(values["fy"]) * self.fy_scale
        values["cx"] = float(values["cx"]) + self.cx_offset_px
        values["cy"] = float(values["cy"]) + self.cy_offset_px
        return values
