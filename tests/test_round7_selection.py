import pandas as pd

from tools.optimization_selection import SELECTION_MODES
from tools.round7_selection import (
    annotate_round7_scores,
    compute_exp010_similarity_row,
    control_like_score,
    is_vicreg_active,
    select_round7_diverse_downstream_probe,
)


def _pool_df():
    return pd.DataFrame(
        {
            "ID": ["ctrl_a", "vicreg_a", "collapse_a", "kmeans_a", "wass_a", "sweet_a"],
            "kmeans_ari": [0.74, 0.70, 0.72, 0.82, 0.71, 0.69],
            "wasserstein": [0.63, 0.65, 0.60, 0.70, 0.62, 0.61],
            "latent_size": [64, 64, 64, 64, 64, 64],
            "alignment_collapse": [False, False, True, False, False, False],
            "structure_pass": [True, True, False, True, True, True],
            "lambda_proto": [0, 0, 0, 0, 0, 0],
            "lambda_tumor_topology": [0, 0, 0, 0, 0, 0],
            "lambda_class_gap": [0, 0, 0, 0, 0, 0],
            "lambda_tumor_supcon": [0, 0, 0, 0, 0, 0],
            "lambda_subspace_ortho": [0, 0, 0, 0, 0, 0],
            "lambda_tumor_var": [0, 0.0003, 0, 0, 0, 0],
            "lambda_tumor_cov": [0, 0.0003, 0, 0, 0, 0],
            "fid": [20, 22, 25, 18, 21, 19],
            "mmd": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        }
    )


def test_round7_selection_mode_registered():
    assert "round7_diverse_downstream_probe" in SELECTION_MODES


def test_exp010_similarity_high_for_latent64_moderate_metrics():
    row = pd.Series(
        {
            "kmeans_ari": 0.74,
            "wasserstein": 0.63,
            "latent_size": 64,
            "lambda_tumor_var": 0,
            "lambda_tumor_cov": 0,
            "lambda_tumor_topology": 0,
            "lambda_class_gap": 0,
            "lambda_tumor_supcon": 0,
            "lambda_subspace_ortho": 0,
        }
    )
    assert compute_exp010_similarity_row(row) >= 0.85
    assert control_like_score(row) == 1.0


def test_vicreg_active_group():
    row = pd.Series({"lambda_tumor_var": 0.0003, "lambda_tumor_cov": 0.0003, "lambda_tumor_topology": 0})
    assert is_vicreg_active(row)


def test_diverse_selection_keeps_forced_baselines_and_groups():
    pool = annotate_round7_scores(_pool_df())
    selected, info = select_round7_diverse_downstream_probe(
        pool,
        pool,
        top_k=12,
        force_baseline_models=["exp_010", "exp_012", "exp_746"],
    )
    assert "round7_selection_group" in selected.columns
    assert "round7_diversity_reason" in selected.columns
    forced = selected[selected["round7_selection_group"] == "G7_historical_baseline"]["ID"].astype(str).tolist()
    assert "exp_010" in forced
    assert "exp_012" in forced
    assert "exp_746" in forced
    assert "collapse_a" not in set(selected["ID"].astype(str))
    groups = set(selected["round7_selection_group"])
    assert "G2_vicreg_active" in groups or any(selected["round7_vicreg_active"].fillna(False))


def test_topk_not_all_one_type():
    pool = annotate_round7_scores(_pool_df())
    selected, _ = select_round7_diverse_downstream_probe(pool, pool, top_k=10)
    assert selected["round7_control_like"].fillna(False).any()
    assert selected["round7_vicreg_active"].fillna(False).any()
