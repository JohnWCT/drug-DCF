import pandas as pd

from tools.round18_cv_metrics import calculate_robust_drug_macro_metrics, early_stop_score


def test_robust_drug_macro_filters_insufficient_drugs():
    rows = []
    for i in range(12):
        rows.append({"DRUG_NAME": "A", "Label": int(i % 2 == 0), "probability": 0.1 + 0.05 * i})
    for i in range(5):
        rows.append({"DRUG_NAME": "B", "Label": int(i % 2 == 0), "probability": 0.2})
    for i in range(12):
        rows.append({"DRUG_NAME": "C", "Label": 1, "probability": 0.9})

    metrics = calculate_robust_drug_macro_metrics(pd.DataFrame(rows))
    assert metrics["n_total_drugs"] == 3
    assert metrics["n_valid_auc_drugs"] == 1
    assert metrics["DrugMacro_AUC"] is not None
    assert metrics["Global_AUC"] is not None


def test_early_stop_fallback_when_few_valid_drugs():
    rows = []
    for i in range(12):
        rows.append({"DRUG_NAME": "A", "Label": int(i % 2 == 0), "probability": 0.1 + 0.04 * i})
    metrics = calculate_robust_drug_macro_metrics(pd.DataFrame(rows))
    score = early_stop_score(metrics, min_valid_drugs_for_early_stop=3)
    assert score["fallback_used"] is True
    assert score["score_name"] == "Global_AUC"
