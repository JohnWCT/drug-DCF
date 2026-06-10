import pandas as pd

from tools.optimization_selection import select_top10_with_controls


def _sample_df():
    rows = []
    for i in range(12):
        rows.append(
            {
                "ID": f"exp_{i:03d}",
                "score_total": 1.0 - i * 0.05,
                "lambda_proto": 0.0 if i in (3, 7) else 0.1,
            }
        )
    return pd.DataFrame(rows)


def test_top10_includes_two_controls_when_available():
    top10, info = select_top10_with_controls(_sample_df(), n_ranked=8, n_controls=2)
    assert len(top10) == 10
    assert info["controls_selected"] == 2
    assert (top10["lambda_proto"] == 0).sum() == 2


def test_handles_fewer_than_two_controls():
    df = _sample_df()
    df.loc[df["lambda_proto"] == 0, "lambda_proto"] = 0.1
    df.loc[0, "lambda_proto"] = 0.0
    top10, info = select_top10_with_controls(df, n_ranked=8, n_controls=2)
    assert info["controls_selected"] == 1
    assert info["shortage"] is True


def test_stable_ranking_for_ties():
    df = pd.DataFrame(
        [
            {"ID": "exp_a", "score_total": 0.5, "lambda_proto": 0.1},
            {"ID": "exp_b", "score_total": 0.5, "lambda_proto": 0.1},
            {"ID": "exp_c", "score_total": 0.4, "lambda_proto": 0.0},
            {"ID": "exp_d", "score_total": 0.3, "lambda_proto": 0.0},
        ]
    )
    top10_a, _ = select_top10_with_controls(df, n_ranked=2, n_controls=2)
    top10_b, _ = select_top10_with_controls(df, n_ranked=2, n_controls=2)
    assert top10_a["ID"].tolist() == top10_b["ID"].tolist()
