"""Tests for Round 10 selection."""

import pandas as pd
import pytest

from tools.optimization_runner import build_parser
from tools.round10_selection import annotate_round10_scores, select_round10_cond_adv_candidates


def test_runner_parser_accepts_round10_selection_mode():
    parser = build_parser()
    args = parser.parse_args(
        [
            "select",
            "--run-dir",
            "result/optimization_runs/round10_cond_adv",
            "--selection-mode",
            "round10_cond_adv_qc",
        ]
    )
    assert args.selection_mode == "round10_cond_adv_qc"


def _sample_pool():
    rows = []
    for i, branch in enumerate(
        [
            "10A_global_adv_repro",
            "10B_conditional_replacement",
            "10C_conditional_plus_weak_global",
        ]
    ):
        for j in range(3):
            rows.append(
                {
                    "model_id": f"exp_{i}{j}",
                    "round10_branch": branch,
                    "lambda_cond_adv": 0.0 if branch.startswith("10A") else 0.0003,
                    "conditional_adv_enabled": not branch.startswith("10A"),
                    "mean_conditional_leakage_strength": 0.9 - i * 0.1 - j * 0.01,
                    "macro_conditional_domain_auc": 0.85,
                    "kmeans_ari": 0.55,
                    "wasserstein": 0.5,
                    "fid": 0.5,
                    "inter_cancer_margin": 0.4,
                    "collapse_flag": False,
                }
            )
    return pd.DataFrame(rows)


def test_selection_output_contains_group():
    pool = _sample_pool()
    selected, info = select_round10_cond_adv_candidates(pool, pool, top_k=6)
    assert "round10_selection_group" in selected.columns
    assert len(selected) >= 3
    assert info["selection_mode"] == "round10_cond_adv_qc"


def test_fail_fast_without_nonzero_conditional():
    pool = _sample_pool()
    pool = pool[pool["round10_branch"] == "10A_global_adv_repro"]
    with pytest.raises(ValueError, match="No nonzero conditional ADV candidates"):
        select_round10_cond_adv_candidates(pool, pool, top_k=5)
