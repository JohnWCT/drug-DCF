import pandas as pd

from tools.analyze_round8_pretrain import build_branch_summaries, infer_branch, write_reports


def test_infer_branch_round8():
    assert infer_branch("vaewc_round8A_control_arch_broad") == "8A"
    assert infer_branch("vaewc_round8B_vicreg_arch_broad") == "8B"


def test_build_branch_summaries_empty():
    assert build_branch_summaries(pd.DataFrame()).empty


def test_build_branch_summaries_combined():
    df = pd.DataFrame(
        {
            "ID": ["a", "b"],
            "pretrain_run_tag": ["vaewc_round8A_control_arch_broad", "vaewc_round8B_vicreg_arch_broad"],
            "kmeans_ari": [0.5, 0.6],
            "wasserstein": [0.5, 0.6],
            "alignment_collapse": [False, False],
            "structure_pass": [True, True],
            "round8_downstream_probe_score": [0.5, 0.6],
            "round8_vicreg_active": [False, True],
            "round8_control_like": [True, False],
        }
    )
    summary = build_branch_summaries(df)
    assert "combined" in set(summary["branch"])
    assert len(summary) >= 3


def test_write_reports_empty(tmp_path):
    paths = write_reports(pd.DataFrame(), str(tmp_path))
    assert paths["csv_path"].endswith("round8_pretrain_diagnostics.csv")
