"""Cancer-type-held-out split QC."""
from __future__ import annotations

from pathlib import Path

import pytest

from tools.round19_cancer_type_groups import build_modelid_cancer_type_map
from tools.round19_stage19e_splits import build_cancer_type_heldout

ROOT = Path("result/optimization_runs/round19_factorial")


@pytest.mark.integration
def test_cancer_mapping_no_missing():
    if not (ROOT / "splits" / "development_rows.csv").is_file():
        pytest.skip("development rows missing")
    import pandas as pd

    dev = pd.read_csv(ROOT / "splits" / "development_rows.csv")
    mapping, meta = build_modelid_cancer_type_map(dev)
    assert mapping["cancer_type"].notna().all()
    assert "UNKNOWN" not in set(mapping["cancer_type"].astype(str))
    row = mapping[mapping.ModelID == "ACH-000708"]
    if len(row):
        assert str(row.iloc[0]["cancer_type"]) == "COREAD"


@pytest.mark.integration
def test_cancer_heldout_no_overlap():
    if not (ROOT / "splits" / "development_rows.csv").is_file():
        pytest.skip("development rows missing")
    assign, qc, n_folds = build_cancer_type_heldout(ROOT)
    assert n_folds in (3, 5)
    for fold in range(n_folds):
        f = assign[assign.fold_id == fold]
        tr_c = set(f.loc[f.split_role == "train", "cancer_type"].astype(str))
        va_c = set(f.loc[f.split_role == "val", "cancer_type"].astype(str))
        assert not (tr_c & va_c)
        tr_m = set(f.loc[f.split_role == "train", "ModelID"].astype(str))
        va_m = set(f.loc[f.split_role == "val", "ModelID"].astype(str))
        assert not (tr_m & va_m)
        assert len(va_c) >= 2
    assert all(r["valid_drugmacro_drugs"] >= 3 for r in qc)
