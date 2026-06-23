#!/usr/bin/env python3
import pandas as pd
from tools.round13_selection import annotate_round13_scores, select_round13_proto_response_candidates


def test_selection_keeps_z_only_baselines():
    df = pd.DataFrame(
        [
            {"Model_ID": "r13_exp_037_none", "source_model_id": "exp_037", "prototype_feature_mode": "none", "Average_TCGA_AUC_mean": 0.59},
            {"Model_ID": "r13_exp_037_own_cancer", "source_model_id": "exp_037", "prototype_feature_mode": "own_cancer", "Average_TCGA_AUC_mean": 0.60},
            {"Model_ID": "r13_exp_035_none", "source_model_id": "exp_035", "prototype_feature_mode": "none", "Average_TCGA_AUC_mean": 0.58},
            {"Model_ID": "r13_exp_035_own_cancer", "source_model_id": "exp_035", "prototype_feature_mode": "own_cancer", "Average_TCGA_AUC_mean": 0.581},
        ]
    )
    top, info = select_round13_proto_response_candidates(df, df, top_k=4, force_baseline_models=["r13_exp_037_none", "r13_exp_035_none"])
    assert info["selected"] >= 2
    assert "r13_exp_037_none" in top["Model_ID"].astype(str).tolist()


def test_annotate_round13_scores_adds_columns():
    df = pd.DataFrame([{"Model_ID": "r13_exp_037_own_cancer", "Average_TCGA_AUC_mean": 0.6, "prototype_feature_mode": "own_cancer"}])
    out = annotate_round13_scores(df)
    assert "round13_proto_feature_score" in out.columns
