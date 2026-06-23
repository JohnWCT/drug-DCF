"""Tests for Round 12 selection."""

import pandas as pd

from tools.optimization_selection import SELECTION_MODES
from tools.round12_selection import annotate_round12_scores, select_round12_proto_alignment_candidates


def _sample_pool():
    return pd.DataFrame(
        [
            {
                "ID": "exp_a",
                "round12_branch": "12B_proto_alignment_main",
                "lambda_proto_align": 0.0001,
                "source_anchor_proto_enabled": True,
                "reconstruction_loss_type": "mse",
                "mean_same_cancer_proto_distance": 0.35,
                "inter_cancer_margin": 1.2,
                "mean_conditional_leakage_strength": 0.4,
                "kmeans_ari": 0.7,
                "wasserstein": 0.9,
                "fid": 1.0,
                "Average_TCGA_AUC_mean": 0.55,
            },
            {
                "ID": "exp_b",
                "round12_branch": "12B_no_proto_control",
                "lambda_proto_align": 0.0,
                "source_anchor_proto_enabled": False,
                "reconstruction_loss_type": "mse",
                "mean_same_cancer_proto_distance": 0.42,
                "inter_cancer_margin": 1.1,
                "mean_conditional_leakage_strength": 0.38,
                "kmeans_ari": 0.2,
                "wasserstein": 1.1,
                "fid": 1.2,
                "Average_TCGA_AUC_mean": 0.50,
            },
            {
                "ID": "exp_c",
                "round12_branch": "12C_recon_proto",
                "lambda_proto_align": 0.001,
                "source_anchor_proto_enabled": True,
                "reconstruction_loss_type": "smooth_l1",
                "mean_same_cancer_proto_distance": 0.30,
                "inter_cancer_margin": 1.0,
                "mean_conditional_leakage_strength": 0.41,
                "kmeans_ari": 0.75,
                "wasserstein": 0.8,
                "fid": 0.9,
                "Average_TCGA_AUC_mean": 0.58,
            },
        ]
    )


def test_selection_mode_registered():
    assert "round12_proto_alignment_qc" in SELECTION_MODES


def test_collapse_candidate_penalized():
    scored = annotate_round12_scores(_sample_pool())
    low_ari = scored[scored["ID"] == "exp_b"]["round12_proto_alignment_score"].iloc[0]
    good = scored[scored["ID"] == "exp_c"]["round12_proto_alignment_score"].iloc[0]
    assert good > low_ari


def test_selection_keeps_mse_control_and_active_proto():
    pool = _sample_pool()
    top, info = select_round12_proto_alignment_candidates(pool, pool, top_k=3, force_baseline_models=[])
    ids = set(top["ID"].astype(str))
    assert "exp_c" in ids
    assert info["selected"] >= 2
