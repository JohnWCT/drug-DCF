import pandas as pd
from tools.analyze_round6_pretrain import build_group_summaries
from tools.round6_selection import annotate_sweetspot_scores

def test_build_group_summaries_with_sweetspot():
    df = annotate_sweetspot_scores(pd.DataFrame({
        "ID": ["exp_1"], "pretrain_run_tag": ["vaewc_round6A_tumor_topology"],
        "kmeans_ari": [0.7], "wasserstein": [0.6], "fid": [20], "mmd": [0.1],
        "latent_size": [32], "alignment_collapse": [False], "structure_pass": [True],
        "lambda_tumor_topology": [0.0001], "lambda_class_gap": [0],
        "lambda_tumor_supcon": [0], "lambda_tumor_var": [0], "lambda_tumor_cov": [0],
        "lambda_subspace_ortho": [0], "tumor_topology_valid": [True],
    }))
    summary = build_group_summaries(df)
    assert not summary.empty
    assert "mean_sweetspot_score" in summary.columns
