import pandas as pd
from tools.analyze_round5_pretrain import _summarize_group


def test_best_noncollapse_model_excludes_collapsed():
    df = pd.DataFrame([
        {"ID": "collapsed", "kmeans_ari": 0.1, "wasserstein": 0.2, "fid": 10, "mmd": 0.01, "alignment_collapse": True, "structure_pass": False, "proto_invalid": False},
        {"ID": "ok", "kmeans_ari": 0.7, "wasserstein": 0.5, "fid": 20, "mmd": 0.02, "alignment_collapse": False, "structure_pass": True, "proto_invalid": False},
    ])
    summary = _summarize_group(df)
    assert summary["best_noncollapse_model"] == "ok"
