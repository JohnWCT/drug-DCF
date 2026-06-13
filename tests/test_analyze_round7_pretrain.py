import pandas as pd

from tools.analyze_round7_pretrain import build_branch_summaries, infer_branch
from tools.round7_selection import annotate_round7_scores


def _sample():
    return annotate_round7_scores(
        pd.DataFrame(
            {
                "ID": ["a", "b"],
                "pretrain_run_tag": [
                    "vaewc_round7A_exp010_control_refinement",
                    "vaewc_round7B_vicreg_focused_ablation",
                ],
                "kmeans_ari": [0.74, 0.70],
                "wasserstein": [0.63, 0.65],
                "fid": [20, 21],
                "alignment_collapse": [False, False],
                "structure_pass": [True, True],
                "latent_size": [64, 64],
                "lambda_tumor_var": [0, 0.0003],
                "lambda_tumor_cov": [0, 0.0003],
                "lambda_tumor_topology": [0, 0],
                "lambda_class_gap": [0, 0],
                "lambda_tumor_supcon": [0, 0],
                "lambda_subspace_ortho": [0, 0],
            }
        )
    )


def test_infer_branch():
    assert infer_branch("vaewc_round7A_exp010_control_refinement") == "7A"
    assert infer_branch("vaewc_round7B_vicreg_focused_ablation") == "7B"


def test_build_branch_summaries():
    df = _sample()
    df["branch"] = df["pretrain_run_tag"].map(infer_branch)
    summary = build_branch_summaries(df)
    assert not summary.empty
    assert set(summary["branch"]) >= {"7A", "7B", "combined"}
    assert "best_control_model" in summary.columns
    assert "best_vicreg_model" in summary.columns


def test_empty_branch_no_crash():
    summary = build_branch_summaries(pd.DataFrame())
    assert summary.empty
