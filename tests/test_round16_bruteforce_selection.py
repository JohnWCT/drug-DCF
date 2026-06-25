import pandas as pd

from tools.round16_bruteforce_selection import aggregate_seed_stats, select_round16_bruteforce_candidates


def _make_all_df():
    rows = []
    for seed in (101, 202, 303):
        rows.append({
            "model_id": "r13_exp_008_none",
            "round16_model_key": "r13_exp_008",
            "feature_mode": "none",
            "combo_id": 0,
            "seed": seed,
            "Average_TCGA_AUC_mean": 0.59 + seed * 1e-6,
        })
        rows.append({
            "model_id": "r13_exp_008_own_plus_summary",
            "round16_model_key": "r13_exp_008",
            "feature_mode": "own_plus_summary",
            "combo_id": 0,
            "seed": seed,
            "Average_TCGA_AUC_mean": 0.61,
        })
    high_var = []
    for seed in (101, 202, 303):
        high_var.append({
            "model_id": "r15c_exp_005_own_plus_summary",
            "round16_model_key": "r15c_exp_005",
            "feature_mode": "own_plus_summary",
            "combo_id": 1,
            "seed": seed,
            "Average_TCGA_AUC_mean": 0.62 if seed == 101 else 0.58,
        })
    return pd.DataFrame(rows + high_var)


def test_selection_uses_seed_mean_not_single_best():
    all_df = _make_all_df()
    summary = aggregate_seed_stats(all_df)
    r15 = summary[(summary["round16_model_key"] == "r15c_exp_005") & (summary["combo_id"] == 1)].iloc[0]
    assert r15["mean_auc_across_seeds"] < r15["best_auc"]


def test_score_penalizes_high_std():
    all_df = _make_all_df()
    summary = aggregate_seed_stats(all_df)
    stable = summary[(summary["round16_model_key"] == "r13_exp_008") & (summary["feature_mode"] == "own_plus_summary")].iloc[0]
    volatile = summary[(summary["round16_model_key"] == "r15c_exp_005") & (summary["combo_id"] == 1)].iloc[0]
    assert stable["round16_bruteforce_score"] > volatile["round16_bruteforce_score"]


def test_forced_retention_includes_models_and_modes():
    all_df = _make_all_df()
    agg = aggregate_seed_stats(all_df)
    top, info = select_round16_bruteforce_candidates(agg, all_df, top_k=10)
    assert "r13_exp_008" in set(top["round16_model_key"])
    assert "none" in set(top["feature_mode"])
    assert "own_plus_summary" in set(top["feature_mode"])
    assert info["selection_mode"] == "round16_bruteforce_qc"
