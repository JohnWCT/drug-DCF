#!/usr/bin/env python3
import os
import pandas as pd
from tools.analyze_round13_proto_response import _z_vs_proto_delta, analyze_round13


def test_z_vs_proto_delta(tmp_path):
    df = pd.DataFrame(
        [
            {"Model_ID": "r13_exp_037_none", "Average_TCGA_AUC_mean": 0.59},
            {"Model_ID": "r13_exp_037_own_cancer", "Average_TCGA_AUC_mean": 0.60},
        ]
    )
    out = _z_vs_proto_delta(df)
    assert len(out) == 1
    assert out.iloc[0]["delta_proto_minus_z_only"] > 0


def test_writes_final_report(tmp_path):
    agg = pd.DataFrame([{"Model_ID": "r13_exp_037_own_cancer", "Average_TCGA_AUC_mean": 0.601}])
    agg_path = tmp_path / "aggregate_scores.csv"
    agg.to_csv(agg_path, index=False)
    report = analyze_round13(str(tmp_path), "result/optimization_runs/round12_proto_alignment", "result/optimization_runs/round11_stability_recon", str(tmp_path / "final_report"), aggregate_path=str(agg_path))
    assert os.path.isfile(report)
