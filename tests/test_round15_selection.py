import pandas as pd
from tools.round15_selection import select_round15_repro_rescue_candidates, annotate_round15_scores


def _synthetic_pool():
    rows = []
    for i, (route, lam, branch) in enumerate(
        [
            ("exp008_proto_response_route", 0.0, "15C"),
            ("exp008_proto_response_route", 3e-6, "15C"),
            ("exp008_proto_response_route", 1e-5, "15C"),
            ("exp035_strong_zonly_route", 0.0, "15C"),
        ]
    ):
        rows.append(
            {
                "ID": f"exp_{i+1:03d}",
                "result_folder": f"/tmp/exp_{i+1:03d}",
                "route_id": route,
                "round15_branch": branch,
                "source_model": "exp_008" if "008" in route else "exp_035",
                "lambda_tumor_var": lam,
                "lambda_tumor_cov": lam,
                "kmeans_ari": 0.5 + i * 0.01,
                "score_total": 0.4 + i * 0.05,
            }
        )
    return pd.DataFrame(rows)


def test_selection_keeps_exp008_route():
    pool = _synthetic_pool()
    top, info = select_round15_repro_rescue_candidates(pool, pool, top_k=4)
    assert info["exp008_route_included"]
    assert (top["round15_route_id"].astype(str) == "exp008_proto_response_route").any()


def test_annotate_adds_score():
    pool = _synthetic_pool()
    ann = annotate_round15_scores(pool)
    assert "round15_repro_rescue_score" in ann.columns
