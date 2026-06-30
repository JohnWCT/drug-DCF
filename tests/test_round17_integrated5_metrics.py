"""Round 17 Phase 0: Integrated5 macro metrics."""

from __future__ import annotations

import numpy as np

from tools.finetune_tcga_eval import (
    HISTORICAL_TCGA_EVAL_KEYS,
    INTEGRATED5_METRIC_KEYS,
    ROUND17_TCGA_EVAL_KEYS,
    compute_integrated5_tcga_metrics,
    compute_integrated_tcga_metrics,
    flatten_tcga_eval_metrics,
)


def _synthetic_target_result(avg_auc: float, drug_aucs: dict[str, float]) -> dict:
    return {
        "Global_Metrics": {"AUC": avg_auc, "AUPRC": avg_auc - 0.05},
        "Average_Metrics": {"AUC": avg_auc, "AUPRC": avg_auc - 0.05},
        "Drug_Metrics": {
            drug: {"AUC": auc, "AUPRC": auc - 0.05}
            for drug, auc in drug_aucs.items()
        },
        "Sample_Predictions": [],
    }


def _synthetic_five_target_suite() -> dict:
    suite = {}
    for idx, key in enumerate(ROUND17_TCGA_EVAL_KEYS):
        avg = 0.50 + 0.05 * idx
        suite[key] = _synthetic_target_result(avg, {f"drug_{idx}": avg, f"drug_{idx}_b": avg + 0.01})
    return suite


def test_integrated5_macro_metrics_exist():
    suite = _synthetic_five_target_suite()
    metrics = compute_integrated5_tcga_metrics(suite)
    assert metrics["Integrated5_n_tcga_eval_targets"] == 5
    assert metrics["Integrated5_n_tcga_drugs_with_valid_auc"] == 10
    for key in INTEGRATED5_METRIC_KEYS:
        assert key in metrics
    assert not np.isnan(metrics["Integrated5_TargetMacro_TCGA_AUC"])
    assert not np.isnan(metrics["Integrated5_DrugMacro_TCGA_AUC"])


def test_integrated_historical_still_uses_three_targets():
    suite = _synthetic_five_target_suite()
    historical = compute_integrated_tcga_metrics(suite)
    assert historical["Integrated_n_tcga_eval_targets"] == len(HISTORICAL_TCGA_EVAL_KEYS)
    expected_target_macro = float(np.mean([0.50, 0.55, 0.60]))
    assert abs(historical["Integrated_Average_TCGA_AUC"] - expected_target_macro) < 1e-6


def test_flatten_includes_integrated5_and_preserves_headline_aliases():
    suite = _synthetic_five_target_suite()
    flat = flatten_tcga_eval_metrics(suite)
    for key in INTEGRATED5_METRIC_KEYS:
        assert key in flat
    assert flat["Global_TCGA_AUC"] == suite["gdsc_intersect13"]["Global_Metrics"]["AUC"]
    assert flat["Average_TCGA_AUC"] == suite["gdsc_intersect13"]["Average_Metrics"]["AUC"]
    assert flat["aacdr_tcga_only_Global_TCGA_AUC"] == suite["aacdr_tcga_only"]["Global_Metrics"]["AUC"]
    assert flat["aacdr_gdsc_intersect_Average_TCGA_AUC"] == suite["aacdr_gdsc_intersect"]["Average_Metrics"]["AUC"]
