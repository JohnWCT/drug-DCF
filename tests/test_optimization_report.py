import json
import os

import pandas as pd

from tools.optimization_report import generate_final_reports


def test_report_from_synthetic_aggregate(tmp_path):
    run_dir = tmp_path / "run"
    aggregate_dir = run_dir / "aggregate"
    selection_dir = run_dir / "selection"
    aggregate_dir.mkdir(parents=True)
    selection_dir.mkdir(parents=True)

    pd.DataFrame(
        [
            {"Model_ID": "exp_001", "Global_TCGA_AUC_mean": 0.7},
            {"Model_ID": "exp_002", "Global_TCGA_AUC_mean": 0.8},
        ]
    ).to_csv(aggregate_dir / "aggregate_scores.csv", index=False)

    pd.DataFrame(
        [
            {"ID": "exp_001", "score_total": 0.6, "lambda_proto": 0.0, "is_control": True},
            {"ID": "exp_002", "score_total": 0.9, "lambda_proto": 0.1, "is_control": False},
        ]
    ).to_csv(selection_dir / "pretrain_top10.csv", index=False)

    pd.DataFrame(
        [
            {"ID": "exp_001", "score_total": 0.6, "lambda_proto": 0.0, "fid": 10, "mmd": 0.04},
            {"ID": "exp_002", "score_total": 0.9, "lambda_proto": 0.1, "fid": 12, "mmd": 0.03},
        ]
    ).to_csv(selection_dir / "pretrain_filtered_candidates.csv", index=False)

    outputs = generate_final_reports(str(run_dir))
    assert os.path.exists(outputs["report_path"])
    with open(outputs["summary_path"], "r", encoding="utf-8") as f:
        summary = json.load(f)
    assert "control_vs_infonce" in summary
    assert summary["control_vs_infonce"]["fid"]["control_mean"] == 10.0
