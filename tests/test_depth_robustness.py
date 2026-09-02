from tools.run_depth_robustness import parse_csv_values


def test_parse_csv_values_drops_empty_items_and_trims_whitespace():
    assert parse_csv_values(" brightness, ,motion_blur ") == ["brightness", "motion_blur"]
