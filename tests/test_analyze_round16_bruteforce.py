import pandas as pd

from tools.analyze_round16_bruteforce import analyze_round16


def test_analyze_round16_writes_report(tmp_path):
    agg = pd.DataFrame([
        {
            "model_id": "r13_exp_008_own_plus_summary",
            "round16_model_key": "r13_exp_008",
            "feature_mode": "own_plus_summary",
            "combo_id": 0,
            "Average_TCGA_AUC_mean": 0.612,
            "Global_TCGA_AUC_mean": 0.61,
        }
    ])
    all_df = agg.copy()
    agg_path = tmp_path / "aggregate_scores.csv"
    agg.to_csv(agg_path, index=False)
    (tmp_path / "aggregate").mkdir()
    all_df.to_csv(tmp_path / "aggregate" / "all_scores.csv", index=False)
    out = analyze_round16(str(tmp_path), str(agg_path), stage="16a", outdir=str(tmp_path / "reports"))
    assert (tmp_path / "reports" / "round16_final_report.md").is_file()
    assert out["seed_summary_rows"] >= 1
