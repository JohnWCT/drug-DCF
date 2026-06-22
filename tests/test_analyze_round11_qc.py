"""Tests for Round 11 analyze QC."""

import os
import tempfile

from tools.analyze_round11_qc import analyze_round11, _collect_pretrain_summaries


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
