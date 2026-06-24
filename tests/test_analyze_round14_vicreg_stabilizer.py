#!/usr/bin/env python3
import os
import sys
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.analyze_round14_vicreg_stabilizer import analyze_round14


def test_analyze_partial_report(tmp_path):
    run_dir = tmp_path / "round14"
    pretrain_dir = run_dir / "pretrain" / "exp_001"
    os.makedirs(pretrain_dir, exist_ok=True)
    agg_dir = run_dir / "aggregate"
    os.makedirs(agg_dir, exist_ok=True)
    pd.DataFrame([{"Model_ID": "r14_exp_001_own_plus_summary", "Average_TCGA_AUC_mean": 0.615, "Global_TCGA_AUC_mean": 0.62}]).to_csv(agg_dir / "aggregate_scores.csv", index=False)
    report = analyze_round14(run_dir=str(run_dir), round13_root=str(tmp_path), round12_root=str(tmp_path), outdir=str(run_dir / "final_report"))
    assert os.path.isfile(report)
    text = open(report, encoding="utf-8").read()
    assert "Round 14 Final Report" in text


def test_normalize_vicreg_row_falls_back_to_g_loss(tmp_path):
    from tools.analyze_round14_vicreg_stabilizer import _normalize_vicreg_row

    exp_dir = tmp_path / "exp_001"
    exp_dir.mkdir()
    pd.DataFrame(
        {
            "tumor_vicreg_var_loss": [0.1, 0.3],
            "tumor_vicreg_cov_loss": [0.2, 0.4],
        }
    ).to_csv(exp_dir / "g_loss.csv", index=False)
    row = _normalize_vicreg_row({"model_id": "exp_001"}, str(exp_dir))
    assert row["tumor_vicreg_var_loss_mean"] == 0.2
    assert abs(row["tumor_vicreg_cov_loss_mean"] - 0.3) < 1e-9
    assert abs(row["tumor_vicreg_loss_mean"] - 0.5) < 1e-9


def test_normalize_vicreg_row_maps_legacy_keys():
    from tools.analyze_round14_vicreg_stabilizer import _normalize_vicreg_row

    row = _normalize_vicreg_row(
        {
            "tumor_vicreg_var_loss": 0.11,
            "tumor_vicreg_cov_loss": 0.22,
            "tumor_vicreg_cov_offdiag_mean_abs": 0.05,
        },
        "/nonexistent",
    )
    assert row["tumor_vicreg_var_loss_mean"] == 0.11
    assert row["tumor_vicreg_cov_loss_mean"] == 0.22
    assert row["tumor_vicreg_loss_mean"] == 0.33
    assert row["latent_cov_offdiag_mean"] == 0.05
