import numpy as np

from tools.finetune_tcga_eval import flatten_tcga_eval_metrics, is_per_drug_tcga_column


def _mk_result(auc, drug):
    return {
        "Global_Metrics": {"AUC": auc, "AUPRC": 0.5},
        "Average_Metrics": {"AUC": auc, "AUPRC": 0.5},
        "Drug_Metrics": {drug: {"AUC": auc, "AUPRC": 0.5}},
        "Sample_Predictions": [
            {"confidence": 0.9, "ground_truth": 1.0, "drug_id": drug, "sample_id": "p1", "domain": "TCGA"},
            {"confidence": 0.1, "ground_truth": 0.0, "drug_id": drug, "sample_id": "p2", "domain": "TCGA"},
        ],
    }


def test_flatten_includes_all_targets_and_integrated():
    suite = {
        "gdsc_intersect13": _mk_result(0.7, "Paclitaxel"),
        "tcga_only3": _mk_result(0.6, "Doxorubicin"),
        "dapl": _mk_result(0.65, "temozolomide"),
    }
    flat = flatten_tcga_eval_metrics(suite)
    assert flat["Global_TCGA_AUC"] == 0.7
    assert flat["tcga_only3_Global_TCGA_AUC"] == 0.6
    assert flat["dapl_Global_TCGA_AUC"] == 0.65
    assert flat["TCGA2_Global_TCGA_AUC"] == 0.65
    assert not np.isnan(flat["Integrated_Global_TCGA_AUC"])
    assert abs(flat["Integrated_Average_TCGA_AUC"] - 0.65) < 1e-6


def test_is_per_drug_tcga_column():
    assert is_per_drug_tcga_column("Paclitaxel_TCGA_AUC") is True
    assert is_per_drug_tcga_column("Global_TCGA_AUC") is False
    assert is_per_drug_tcga_column("tcga_only3_Global_TCGA_AUC") is False
