import json
from pathlib import Path

from tools.round18_config_builder import build_round18_configs


def test_builder_18a_writes_split_artifacts(tmp_path):
    settings = json.loads(Path("config/round18_architecture_settings.json").read_text())
    outdir = tmp_path / "round18"
    result = build_round18_configs(
        "config/round18_architecture_settings.json",
        str(outdir),
        "18a",
    )
    assert Path(result["split_metadata"]).exists()
    assert Path(result["screening_3fold_assignments"]).exists()
    assert Path(result["formal_5fold_assignments"]).exists()
    meta = json.loads(Path(result["split_metadata"]).read_text())
    assert meta["group_column"] == settings["internal_test"]["group_column"]
    assert meta["n_development_rows"] + meta["n_internal_test_rows"] == meta["n_total_rows"]
