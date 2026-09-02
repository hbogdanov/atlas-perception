import numpy as np

from src.depth.input_depth import InputDepthProcessor


def test_input_depth_processor_preserves_metric_depth_without_torch():
    processor = InputDepthProcessor({"output_mode": "raw", "postprocess": {"enabled": False}})
    supplied_depth = np.array([[0.0, 1.25], [2.5, 3.75]], dtype=np.float32)

    prepared = processor.prepare(supplied_depth)

    assert np.array_equal(prepared, supplied_depth)
    assert prepared.dtype == np.float32
