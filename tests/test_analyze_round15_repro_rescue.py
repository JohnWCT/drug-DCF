import os
import pandas as pd
from tools.analyze_round15_repro_rescue import (
    ROUND13_BEST,
    _seed_stability_summary,
    _z_vs_summary_delta,
    analyze_round15,
)


def test_compare_round13_and_seed_stats():
    agg = pd.DataFrame(
        [
            {"Model_ID": "r15a_exp_008_none_s101", "Average_TCGA_AUC_mean": 0.58, "random_seed": 101},
            {"Model_ID": "r15a_exp_008_own_plus_summary_s101", "Average_TCGA_AUC_mean": 0.61, "random_seed": 101},
            {"Model_ID": "r15a_exp_008_own_plus_summary_s202", "Average_TCGA_AUC_mean": 0.60, "random_seed": 202},
        ]
    )
    seed_df = _seed_stability_summary(agg)
    assert not seed_df.empty
    summary = seed_df[seed_df.feature_mode == "own_plus_summary"].iloc[0]
    assert summary.mean_avg_tcga > ROUND13_BEST - 0.05
    delta = _z_vs_summary_delta(agg)
    assert delta.empty or "delta_own_plus_summary_minus_none" in delta.columns


def test_partial_report_without_aggregate(tmp_path):
    run_dir = tmp_path / "round15_partial"
    os.makedirs(run_dir / "manifests")
    pd.DataFrame([{"job_id": "ft_x"}]).to_csv(run_dir / "manifests/finetune_dispatch_manifest.csv", index=False)
    out = analyze_round15(str(run_dir), "result/optimization_runs/round13_proto_response", "result/optimization_runs/round14_vicreg_stabilizer", str(tmp_path / "out"))
    assert os.path.isfile(out["round15_final_report.md"])
