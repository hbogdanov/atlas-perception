from pathlib import Path

from src.io.tum_rgbd import AssociationReport
from src.utils.run_metadata import build_run_manifest, config_hash, write_json


def test_config_hash_is_stable_for_equivalent_dicts():
    assert config_hash({"depth": {"mode": "raw"}, "input": {"mode": "video"}}) == config_hash(
        {"input": {"mode": "video"}, "depth": {"mode": "raw"}}
    )


def test_manifest_records_tum_association_statistics(tmp_path: Path):
    report = AssociationReport(10, 9, 8, 2, 1, 0.01, 0.02)
    manifest = build_run_manifest({"input": {"mode": "rgbd_dataset"}}, tmp_path, report)
    output = tmp_path / "manifest.json"
    write_json(output, manifest)

    assert output.exists()
    assert manifest["associations"]["matched_pairs"] == 8
    assert manifest["associations"]["max_timestamp_error"] == 0.02
