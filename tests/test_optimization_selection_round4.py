import pandas as pd

from tools.optimization_selection import (
    apply_selection_ranking,
    select_top10_with_controls,
)


def _round4_df():
    return pd.DataFrame(
        [
            {"ID": "exp_a", "score_total": 0.9, "score_kmeans": 0.2, "kmeans_ari": 0.3, "wasserstein": 0.8, "fid": 30, "mmd": 0.1, "lambda_proto": 0.1},
            {"ID": "exp_b", "score_total": 0.5, "score_kmeans": 0.8, "kmeans_ari": 0.75, "wasserstein": 0.4, "fid": 20, "mmd": 0.05, "lambda_proto": 0.01},
            {"ID": "exp_c", "score_total": 0.7, "score_kmeans": 0.6, "kmeans_ari": 0.6, "wasserstein": 0.5, "fid": 22, "mmd": 0.04, "lambda_proto": 0.0},
            {"ID": "exp_d", "score_total": 0.6, "score_kmeans": 0.55, "kmeans_ari": 0.55, "wasserstein": 0.55, "fid": 24, "mmd": 0.06, "lambda_proto": 0.0},
        ]
    )


def test_score_total_mode_backward_compatible():
    df = _round4_df()
    ranked = apply_selection_ranking(df, selection_mode="score_total")
    assert ranked.iloc[0]["ID"] == "exp_a"


def test_round4_kmeans_first_prefers_better_kmeans():
    df = _round4_df()
    ranked = apply_selection_ranking(df, selection_mode="round4_kmeans_first")
    assert ranked.iloc[0]["ID"] == "exp_b"


def test_round4_selects_two_controls():
    rows = []
    for i in range(12):
        rows.append(
            {
                "ID": f"exp_{i:03d}",
                "score_total": 1.0 - i * 0.05,
                "score_kmeans": 0.5,
                "wasserstein": 0.5,
                "fid": 20,
                "mmd": 0.05,
                "lambda_proto": 0.0 if i in (3, 7) else 0.01,
            }
        )
    top10, info = select_top10_with_controls(pd.DataFrame(rows), selection_mode="round4_kmeans_first")
    assert len(top10) == 10
    assert info["controls_selected"] == 2
    assert info["selection_mode"] == "round4_kmeans_first"


def test_round4_weighted_adds_score_round4():
    df = _round4_df()
    ranked = apply_selection_ranking(df, selection_mode="round4_weighted")
    assert "score_round4" in ranked.columns
    assert ranked.iloc[0]["ID"] == "exp_b"
