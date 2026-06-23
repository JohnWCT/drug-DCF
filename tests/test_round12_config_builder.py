"""Tests for Round 12 config builder."""

import os
import sys

import pandas as pd
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.round12_config_builder import build_round12_configs

ROUND11_ROOT = os.path.join(PROJECT_ROOT, "result/optimization_runs/round11_stability_recon")
SETTINGS = os.path.join(PROJECT_ROOT, "config/round12_proto_alignment_settings.json")


@pytest.mark.skipif(
    not os.path.isfile(os.path.join(ROUND11_ROOT, "pretrain/exp_035/params.json")),
    reason="Round 11 exp_035 baseline required",
)
def test_build_round12_configs_job_count(tmp_path):
    outdir = tmp_path / "round12_smoke"
    manifest = build_round12_configs(
        settings_path=SETTINGS,
        outdir=str(outdir),
        force=True,
        round11_root=ROUND11_ROOT,
        primary_baseline_exp_id="exp_035",
    )
    df = pd.read_csv(manifest)
    assert len(df) == 66
    assert set(df["round12_branch"].unique()) >= {
        "12B_proto_alignment_main",
        "12B_no_proto_control",
        "12C_recon_proto",
        "12D_euclidean_control",
    }
    active = df[df["source_anchor_proto_enabled"] == True]  # noqa: E712
    assert (active["lambda_proto_align"] > 0).all()
    mse_main = df[
        (df["round12_branch"] == "12B_proto_alignment_main")
        & (df["reconstruction_loss_type"] == "mse")
    ]
    assert len(mse_main) == 36
    no_proto = df[df["round12_branch"] == "12B_no_proto_control"]
    assert len(no_proto) == 3
    assert (no_proto["lambda_proto_align"] == 0).all()


@pytest.mark.skipif(
    os.path.isfile(os.path.join(ROUND11_ROOT, "pretrain/exp_035/params.json")),
    reason="only run fail-fast test when baseline missing",
)
def test_missing_baseline_fail_fast(tmp_path):
    with pytest.raises(FileNotFoundError):
        build_round12_configs(
            settings_path=SETTINGS,
            outdir=str(tmp_path / "bad"),
            force=True,
            round11_root="/nonexistent/round11",
            primary_baseline_exp_id="exp_035",
        )
