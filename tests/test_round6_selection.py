import pandas as pd
from tools.optimization_selection import SELECTION_MODES, apply_selection_ranking
from tools.round6_selection import annotate_sweetspot_scores, range_score, rank_round6_sweetspot

def _sample_df():
    return pd.DataFrame({
        "ID": ["low", "high"],
        "kmeans_ari": [0.60, 0.72],
        "wasserstein": [0.80, 0.62],
        "latent_size": [128, 32],
        "alignment_collapse": [False, False],
        "lambda_tumor_topology": [0, 0.0001],
        "lambda_class_gap": [0, 0],
        "lambda_tumor_supcon": [0, 0],
        "lambda_tumor_var": [0, 0],
        "lambda_tumor_cov": [0, 0],
        "lambda_subspace_ortho": [0, 0],
    })

def test_range_score_ideal_midpoint():
    assert range_score(0.70, 0.65, 0.78) == 1.0

def test_round6_selection_mode_registered():
    assert "round6_sweetspot" in SELECTION_MODES

def test_rank_round6_sweetspot_orders_by_score():
    ranked = rank_round6_sweetspot(_sample_df())
    assert ranked.iloc[0]["ID"] == "high"

def test_apply_selection_ranking_round6():
    out = apply_selection_ranking(_sample_df(), selection_mode="round6_sweetspot")
    assert "sweetspot_score" in out.columns
