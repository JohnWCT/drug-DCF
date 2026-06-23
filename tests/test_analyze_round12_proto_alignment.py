"""Tests for Round 12 analyzer."""

import json
import os
import sys

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.analyze_round12_proto_alignment import analyze_round12


def test_pretrain_only_report(tmp_path):
    pretrain = tmp_path / "pretrain" / "exp_001"
    pretrain.mkdir(parents=True)
    summary = {
        "exp_id": "exp_001",
        "params": {
            "round12_branch": "12B_proto_alignment_main",
            "lambda_proto_align": 0.001,
            "source_anchor_proto_enabled": True,
        },
        "metrics": {
            "kmeans_ari": 0.7,
            "mean_target_to_source_anchor_distance": 0.25,
            "wasserstein": 0.8,
        },
    }
    with open(pretrain / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f)

    outdir = tmp_path / "reports"
    report = analyze_round12(
        run_dir=str(tmp_path),
        round11_root=str(tmp_path),
        outdir=str(outdir),
    )
    assert os.path.isfile(report)
    text = open(report, encoding="utf-8").read()
    assert "Downstream aggregate not available" in text


@pytest.mark.skipif(
    not os.path.isdir(
        os.path.join(PROJECT_ROOT, "result/optimization_runs/round11_stability_recon/pretrain/exp_035")
    ),
    reason="Round 11 artifacts required",
)
def test_baseline_qc_integration(tmp_path):
    from tools.analyze_round12_baseline_prototype_gaps import analyze_round12_baseline_gaps

    outdir = tmp_path / "12a"
    report = analyze_round12_baseline_gaps(
        round11_root=os.path.join(PROJECT_ROOT, "result/optimization_runs/round11_stability_recon"),
        outdir=str(outdir),
        selection_path=os.path.join(
            PROJECT_ROOT,
            "result/optimization_runs/round11_stability_recon/selection/pretrain_top10.csv",
        ),
        top_k=5,
        force_models=["exp_035"],
    )
    assert os.path.isfile(report)
    assert os.path.isfile(outdir / "round11_top_prototype_gap_summary.csv")
