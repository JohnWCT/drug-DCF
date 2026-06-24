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
