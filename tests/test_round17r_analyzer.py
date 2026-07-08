#!/usr/bin/env python3
"""Round 17R analyzer smoke tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from tools.analyze_round17r_18class import analyze_round17r


def test_analyzer_writes_expected_outputs(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    feat = run_dir / "features" / "r13_exp_008" / "own_plus_summary"
    feat.mkdir(parents=True)
    (feat / "feature_metadata.json").write_text(
        '{"n_trainable_cancer_types":18,"uses_legacy_28class_cache":false,'
        '"prototype_class_source":"checkpoint_metadata","source_prototypes_used":18,'
        '"target_prototypes_used":18,"prototype_feature_mode":"own_plus_summary"}'
    )
    agg = tmp_path / "aggregate_scores.csv"
    pd.DataFrame(
        [
            {
                "Model_ID": "r13_exp_008_own_plus_summary",
                "Average_TCGA_AUC_mean": 0.59,
                "Average_TCGA_AUC_std": 0.01,
                "Integrated5_TargetMacro_TCGA_AUC_mean": 0.56,
                "Integrated5_TargetMacro_TCGA_AUC_std": 0.01,
                "Integrated5_DrugMacro_TCGA_AUC_mean": 0.55,
                "Integrated5_DrugMacro_TCGA_AUC_std": 0.01,
                "n_finetune_runs": 3,
            }
        ]
    ).to_csv(agg, index=False)
    outdir = tmp_path / "reports"
    result = analyze_round17r(
        str(run_dir),
        "config/round17r_18class_focused_settings.json",
        str(agg),
        "17r_b",
        str(outdir),
    )
    assert Path(result["report"]).is_file()
    assert (outdir / "round17r_top_candidates.csv").is_file()
    assert (outdir / "round17r_historical_ranking.csv").is_file()
    assert (outdir / "round17r_integrated5_ranking.csv").is_file()
    assert (outdir / "round17r_feature_qc_summary.csv").is_file()
