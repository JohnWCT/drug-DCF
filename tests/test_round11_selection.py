"""Tests for Round 11 selection."""

import pandas as pd

from tools.round11_selection import annotate_round11_scores, select_round11_stability_candidates


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ID": "exp_111",
                "round11_branch": "11C_10C_recon_ablation",
                "reconstruction_loss_type": "smooth_l1",
                "smooth_l1_beta": 1.0,
                "lambda_cond_adv": 0.001,
                "kmeans_ari": 0.7,
                "wasserstein": 1.2,
                "fid": 40,
                "mean_conditional_leakage_strength": 0.3,
            },
            {
                "ID": "exp_002",
                "round11_branch": "11B_10C_stabilization",
                "reconstruction_loss_type": "mse",
                "smooth_l1_beta": 1.0,
                "lambda_cond_adv": 0.0003,
                "kmeans_ari": 0.65,
                "wasserstein": 1.5,
                "fid": 45,
                "mean_conditional_leakage_strength": 0.32,
            },
            {
                "ID": "exp_003",
                "round11_branch": "11C_global_recon_control",
                "reconstruction_loss_type": "hybrid_mse_smooth_l1",
                "smooth_l1_beta": 0.5,
                "lambda_cond_adv": 0.0,
                "kmeans_ari": 0.2,
                "wasserstein": 3.0,
                "fid": 80,
                "mean_conditional_leakage_strength": 0.4,
            },
        ]
    )


def test_optimization_runner_accepts_round11_mode():
    from tools.optimization_selection import SELECTION_MODES

    assert "round11_stability_qc" in SELECTION_MODES


def test_selection_keeps_exp111_and_controls():
    df = _sample_df()
    top, info = select_round11_stability_candidates(df, df, top_k=3, force_baseline_models=["exp_111"])
    ids = set(top["ID"].astype(str))
    assert "exp_111" in ids
    assert info["selection_mode"] == "round11_stability_qc"


def test_collapse_deprioritized():
    annotated = annotate_round11_scores(_sample_df())
    collapsed = annotated[annotated["ID"] == "exp_003"]["round11_stability_score"].iloc[0]
    healthy = annotated[annotated["ID"] == "exp_111"]["round11_stability_score"].iloc[0]
    assert healthy > collapsed
