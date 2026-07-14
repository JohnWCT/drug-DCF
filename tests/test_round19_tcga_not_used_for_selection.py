"""Round 19 selection lock / no-TCGA-for-selection tests."""
from __future__ import annotations

import pytest

from tools.round19_selection_lock import scan_mapping_for_forbidden


def test_tcga_not_used_for_selection_payload():
    ok = {
        "selection_metric": "CV_DrugMacro_AUC",
        "candidates": [{"mean_DrugMacro_AUC": 0.6, "mean_DrugMacro_AUPRC": 0.4}],
    }
    scan_mapping_for_forbidden(ok)
    with pytest.raises(AssertionError):
        scan_mapping_for_forbidden({"Integrated5_DrugMacro_TCGA_AUC": 0.5})
    with pytest.raises(AssertionError):
        scan_mapping_for_forbidden({"nested": {"internal_test_auc": 0.7}})
