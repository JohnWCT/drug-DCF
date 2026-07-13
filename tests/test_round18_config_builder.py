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


def test_builder_18b_manifest_job_count(tmp_path):
    outdir = tmp_path / "round18b"
    # reuse existing settings; splits will be created
    result = build_round18_configs(
        "config/round18_architecture_settings.json",
        str(outdir),
        "18b",
    )
    import pandas as pd
    df = pd.read_csv(result["manifest"])
    assert result["n_jobs"] == 45
    assert len(df) == 45
    assert set(df["architecture_family"]) == {"pooled_mlp", "pooled_transformer"}


def test_builder_18c_manifest_exists(tmp_path):
    outdir = tmp_path / "round18c"
    result = build_round18_configs(
        "config/round18_architecture_settings.json",
        str(outdir),
        "18c",
    )
    import pandas as pd
    df = pd.read_csv(result["manifest"])
    assert result["n_jobs"] == 48
    assert len(df) == 48
    assert set(df["architecture_family"]) == {"cross_attention"}
    assert set(df["residual_mode"]) == {"pure", "pooled_residual"}
    assert set(df["omics_mode"]) == {"own_plus_summary", "own_proto_context_projected_16"}


def test_builder_18b_renames_historical_mask(tmp_path):
    outdir = tmp_path / "round18b_hist"
    result = build_round18_configs(
        "config/round18_architecture_settings.json",
        str(outdir),
        "18b",
    )
    import pandas as pd
    df = pd.read_csv(result["manifest"])
    ids = set(df["transformer_config_id"].dropna().astype(str))
    assert "P0_historical_hparams_corrected_mask" in ids
    assert "P0_historical" not in ids
