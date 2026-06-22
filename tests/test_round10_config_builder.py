"""Tests for Round 10 config builder."""

import json
import os
import sys

import pandas as pd
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.round10_config_builder import MANIFEST_COLUMNS, build_round10_configs

SETTINGS_PATH = os.path.join(PROJECT_ROOT, "config/round10_cond_adv_settings.json")


def test_reads_settings_json():
    assert os.path.exists(SETTINGS_PATH)
    with open(SETTINGS_PATH, encoding="utf-8") as f:
        payload = json.load(f)
    assert payload["round"] == "round10"
    assert payload["primary_baseline_exp_id"] == "exp_048"


def test_round9_summary_missing_fail_fast(tmp_path):
    settings = tmp_path / "settings.json"
    settings.write_text(
        json.dumps(
            {
                "round9_summary": str(tmp_path / "missing.csv"),
                "round9_model_summary": str(tmp_path / "missing_model.csv"),
                "primary_baseline_exp_id": "exp_048",
                "resolved_baselines": str(tmp_path / "resolved.csv"),
                "seeds": [101],
                "branch_design": {"include_10A_global_repro": True},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(FileNotFoundError, match="Round 9 summary missing"):
        build_round10_configs(str(settings), str(tmp_path / "out"), force=True)


def test_baseline_missing_fail_fast(tmp_path):
    r9_summary = tmp_path / "r9_summary.csv"
    pd.DataFrame({"model_id": ["exp_048"]}).to_csv(r9_summary, index=False)
    r9_model = tmp_path / "r9_model.csv"
    pd.DataFrame({"model_id": ["exp_048"]}).to_csv(r9_model, index=False)
    resolved = tmp_path / "resolved.csv"
    pd.DataFrame(
        [{"exp_id": "exp_048", "resolved": False, "checkpoint_dir": "missing"}]
    ).to_csv(resolved, index=False)
    settings = tmp_path / "settings.json"
    settings.write_text(
        json.dumps(
            {
                "round9_summary": str(r9_summary),
                "round9_model_summary": str(r9_model),
                "primary_baseline_exp_id": "exp_048",
                "resolved_baselines": str(resolved),
                "seeds": [101],
                "branch_design": {
                    "include_10A_global_repro": True,
                    "include_10B_conditional_replacement": False,
                    "include_10C_conditional_plus_weak_global": False,
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(FileNotFoundError, match="not resolved"):
        build_round10_configs(str(settings), str(tmp_path / "out"), force=True)


def test_build_full_round10_manifest(tmp_path):
    outdir = tmp_path / "round10"
    manifest = build_round10_configs(SETTINGS_PATH, str(outdir), force=True)
    df = pd.read_csv(manifest)
    assert len(df) == 123
    assert set(df["round10_branch"].unique()) >= {
        "10A_global_adv_repro",
        "10B_conditional_replacement",
        "10C_conditional_plus_weak_global",
    }
    assert len(df[df["round10_branch"] == "10A_global_adv_repro"]) == 3
    assert len(df[df["round10_branch"] == "10B_conditional_replacement"]) == 108
    assert len(df[df["round10_branch"] == "10C_conditional_plus_weak_global"]) == 12
    b_rows = df[df["round10_branch"] == "10B_conditional_replacement"]
    assert (pd.to_numeric(b_rows["lambda_cond_adv"], errors="coerce") > 0).all()
    c_rows = df[df["round10_branch"] == "10C_conditional_plus_weak_global"]
    assert (pd.to_numeric(c_rows["lambda_global_adv_multiplier"], errors="coerce") == 0.25).all()
    for col in MANIFEST_COLUMNS:
        assert col in df.columns


def test_10b_disables_proto_and_vicreg(tmp_path):
    outdir = tmp_path / "round10"
    build_round10_configs(SETTINGS_PATH, str(outdir), force=True)
    config_dir = outdir / "configs"
    b_configs = list(config_dir.glob("round10B_*.json"))
    assert b_configs
    sample = json.loads(b_configs[0].read_text(encoding="utf-8"))["pretrain_param_combinations"][0]
    assert sample["lambda_proto"] == 0
    assert sample["lambda_tumor_supcon"] == 0
    assert sample["use_tumor_subspace"] is False
    assert sample["lambda_tumor_var"] == 0
    assert sample["lambda_tumor_cov"] == 0
