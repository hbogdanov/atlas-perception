import json

from tools.compile_flagship_study import compile_study


def _write_cloud(path, points):
    rows = "\n".join(f"{x} {y} {z}" for x, y, z in points)
    path.write_text(f"ply\nformat ascii 1.0\nelement vertex {len(points)}\nend_header\n{rows}\n", encoding="ascii")


def _write_run(run_dir, points):
    run_dir.mkdir()
    _write_cloud(run_dir / "frame_cloud.ply", points)
    (run_dir / "runtime_metrics.json").write_text(
        json.dumps({"avg_fps": 10.0, "avg_mapping_ms": 5.0, "point_count": 3})
    )
    (run_dir / "manifest.json").write_text(
        json.dumps({"slam": {"mode": "groundtruth"}, "depth": {"source_mode": "input"}})
    )


def test_compile_flagship_study_writes_tables_report_and_figure(tmp_path):
    root = tmp_path / "runs"
    root.mkdir()
    _write_run(root / "baseline", [(0, 0, 0), (1, 0, 0), (0, 1, 0)])
    _write_run(root / "variant", [(0, 0, 0), (1.1, 0, 0), (0, 1, 0)])

    rows = compile_study(root, tmp_path / "study", ["baseline", "variant"], "baseline", 100)

    assert len(rows) == 2
    assert rows[0]["chamfer_distance_m"] == 0.0
    assert (tmp_path / "study" / "headline_results.csv").exists()
    assert (tmp_path / "study" / "technical_report.md").exists()
    assert (tmp_path / "study" / "map_quality_runtime.png").exists()
