import json
import os
import pandas as pd
from tests.round9_test_helpers import write_minimal_checkpoint
from tools.build_round9_finetune_select import build_round9_finetune_select

def test_build_finetune_select_from_manifest(tmp_path):
    repro = tmp_path / "repro"
    exp_dir = write_minimal_checkpoint(str(repro), "exp_001")
    manifest_dir = repro / "manifests"
    manifest_dir.mkdir(parents=True)
    pd.DataFrame([{
        "job_id": "r9_exp_048_seed101",
        "status": "success",
        "result_dir": os.path.relpath(exp_dir, start=str(tmp_path)),
        "source_exp_id": "exp_048",
        "source_role": "primary",
        "reproduction_seed": 101,
    }]).to_csv(manifest_dir / "pretrain_sweep_manifest.csv", index=False)
    resolved = tmp_path / "resolved.csv"
    pd.DataFrame([{"exp_id": "exp_048", "resolved": True}]).to_csv(resolved, index=False)
    # fix result_dir to absolute via PROJECT_ROOT expectations - use full path in manifest
    df_manifest = pd.read_csv(manifest_dir / "pretrain_sweep_manifest.csv")
    df_manifest.loc[0, "result_dir"] = str(exp_dir)
    df_manifest.to_csv(manifest_dir / "pretrain_sweep_manifest.csv", index=False)
    df = build_round9_finetune_select(str(repro), str(resolved))
    assert len(df) == 1
    assert "result_folder" in df.columns
    assert df.iloc[0]["source_exp_id"] == "exp_048"
