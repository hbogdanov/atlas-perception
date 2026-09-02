from tools.run_reconstruction_benchmark import build_command


def test_reconstruction_benchmark_builds_ground_truth_baseline_command():
    command = build_command("gt_depth_gt_pose", 25)

    assert command[1:4] == ["-m", "src.main", "--config"]
    assert command[-1] == "25"
    assert command[command.index("--override-config") + 1].endswith("tum_gt_depth_gt_pose.yaml")
