#!/usr/bin/env python3
import pandas as pd
from tools.round14_selection import annotate_round14_scores, select_round14_vicreg_stabilizer_candidates


def test_selection_keeps_both_routes():
    df = pd.DataFrame(
        [
            {"ID": "exp_001", "route_id": "exp008_proto_response_route", "round14_branch": "14B", "lambda_tumor_var": 0.0001, "lambda_tumor_cov": 0.0001, "kmeans_ari": 0.7, "latent_active_dims": 28, "sweetspot_tcga_proxy_score": 0.55},
            {"ID": "exp_002", "route_id": "exp035_strong_zonly_route", "round14_branch": "14C", "lambda_tumor_var": 0.0, "lambda_tumor_cov": 0.0, "kmeans_ari": 0.68, "latent_active_dims": 30, "sweetspot_tcga_proxy_score": 0.54},
            {"ID": "exp_003", "route_id": "exp008_proto_response_route", "round14_branch": "14B", "lambda_tumor_var": 0.0003, "lambda_tumor_cov": 0.0003, "kmeans_ari": 0.72, "latent_active_dims": 27, "sweetspot_tcga_proxy_score": 0.56},
        ]
    )
    top, info = select_round14_vicreg_stabilizer_candidates(df, df, top_k=3)
    assert info["selected_count"] == 3
    routes = top["round14_route_id"].astype(str).tolist()
    assert "exp008_proto_response_route" in routes
    assert "exp035_strong_zonly_route" in routes


def test_collapse_candidates_filtered():
    df = pd.DataFrame(
        [
            {"ID": "exp_bad", "route_id": "exp008_proto_response_route", "kmeans_ari": 0.1, "latent_active_dims": 2, "sweetspot_tcga_proxy_score": 0.9},
            {"ID": "exp_ok", "route_id": "exp008_proto_response_route", "kmeans_ari": 0.7, "latent_active_dims": 28, "sweetspot_tcga_proxy_score": 0.5},
        ]
    )
    top, _ = select_round14_vicreg_stabilizer_candidates(df, df, top_k=2)
    assert "exp_bad" not in top["ID"].astype(str).tolist()


def test_annotate_adds_score_column():
    df = pd.DataFrame([{"ID": "exp_001", "lambda_tumor_var": 0.0001, "lambda_tumor_cov": 0.0001, "kmeans_ari": 0.7}])
    out = annotate_round14_scores(df)
    assert "round14_vicreg_stabilizer_score" in out.columns
    assert bool(out["round14_vicreg_active"].iloc[0])


def test_missing_latent_active_dims_not_treated_as_collapse():
    from tools.round14_selection import _collapse_risk, select_round14_vicreg_stabilizer_candidates
    import pandas as pd

    row = pd.Series({"kmeans_ari": 0.6, "latent_active_dims": float("nan")})
    assert _collapse_risk(row) is False
    df = pd.DataFrame(
        [
            {"ID": "exp_a", "kmeans_ari": 0.6, "lambda_tumor_var": 0.0001, "lambda_tumor_cov": 0.0001},
            {"ID": "exp_b", "kmeans_ari": 0.5, "lambda_tumor_var": 0.0, "lambda_tumor_cov": 0.0},
        ]
    )
    top, info = select_round14_vicreg_stabilizer_candidates(df, df, top_k=2)
    assert len(top) == 2
