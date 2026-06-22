"""Tests for analyze_round10_cond_adv.py."""

import os
import sys

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.analyze_round10_cond_adv import analyze_round10


def test_round10_success_status_output(tmp_path):
    run_dir = tmp_path / "run"
    for branch, leak, kmeans in [
        ("10A_global_adv_repro", 0.9, 0.5),
        ("10B_conditional_replacement", 0.5, 0.5),
    ]:
        p = run_dir / "pretrain" / branch
        p.mkdir(parents=True)
        (p / "gan_metrics.json").write_text(
            f'{{"round10_branch":"{branch}","mean_conditional_leakage_strength":{leak},'
            f'"kmeans_ari":{kmeans}}}',
            encoding="utf-8",
        )
    r9_dir = tmp_path / "r9"
    r9_dir.mkdir()
    pd.DataFrame({"model_id": ["exp_048"]}).to_csv(
        r9_dir / "round9_model_level_summary.csv", index=False
    )
    pd.DataFrame({"model_id": ["exp_048"]}).to_csv(
        r9_dir / "round9_seed_reproducibility_summary.csv", index=False
    )
    outputs = analyze_round10(str(run_dir), str(r9_dir), str(tmp_path / "out"))
    assert outputs["round10_success_status"] == "success_conditional_only"
    assert os.path.exists(outputs["report"])
