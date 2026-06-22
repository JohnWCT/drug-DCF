"""Tests for Round 11 analyze QC."""

import json
import os
import tempfile

import pandas as pd

from tools.analyze_round11_qc import (
    _build_per_cancer_qc_delta,
    _collect_pretrain_summaries,
    analyze_round11,
)


def test_analyze_pretrain_only():
    with tempfile.TemporaryDirectory() as tmp:
        os.makedirs(os.path.join(tmp, "pretrain", "exp_001"))
        reports = os.path.join(tmp, "reports_out")
        path = analyze_round11(tmp, "", "", reports)
        assert os.path.exists(path)
        assert os.path.exists(os.path.join(reports, "round11_final_report.md"))


def test_collect_empty_run_dir():
    with tempfile.TemporaryDirectory() as tmp:
        assert _collect_pretrain_summaries(tmp).empty


def test_merge_round11a_and_per_cancer_delta():
    with tempfile.TemporaryDirectory() as tmp:
        reports = os.path.join(tmp, "round11a_qc", "reports")
        os.makedirs(reports)
        qc = pd.DataFrame(
            [
                {
                    "model_id": "exp_111",
                    "macro_conditional_domain_auc": 0.72,
                    "mean_conditional_leakage_strength": 0.28,
                },
                {
                    "model_id": "exp_048",
                    "macro_conditional_domain_auc": 0.85,
                    "mean_conditional_leakage_strength": 0.35,
                },
            ]
        )
        qc.to_csv(os.path.join(reports, "round11a_round10_conditional_qc.csv"), index=False)
        per_cancer = pd.DataFrame(
            [
                {"model_id": "exp_048", "cancer_type": "Brain", "logistic_regression_domain_auc": 0.90},
                {"model_id": "exp_111", "cancer_type": "Brain", "logistic_regression_domain_auc": 0.82},
                {"model_id": "exp_048", "cancer_type": "Lung", "logistic_regression_domain_auc": 0.88},
                {"model_id": "exp_111", "cancer_type": "Lung", "logistic_regression_domain_auc": 0.80},
            ]
        )
        per_cancer.to_csv(os.path.join(reports, "round11a_per_cancer_delta.csv"), index=False)

        outdir = os.path.join(tmp, "final_report")
        analyze_round11(tmp, "", "", outdir)
        assert os.path.exists(os.path.join(outdir, "round11_per_cancer_qc_delta.csv"))
        delta = pd.read_csv(os.path.join(outdir, "round11_per_cancer_qc_delta.csv"))
        assert not delta.empty
        assert (delta["delta_domain_auc_vs_baseline"] < 0).all()

        report = open(os.path.join(outdir, "round11_final_report.md")).read()
        assert "exp_111" in report
        assert "Per-cancer conditional leakage delta" in report


def test_build_per_cancer_qc_delta_empty():
    assert _build_per_cancer_qc_delta(pd.DataFrame()).empty
