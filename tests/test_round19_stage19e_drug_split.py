"""Drug-held-out split QC."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from tools.round19_stage19e_splits import build_drug_heldout

ROOT = Path("result/optimization_runs/round19_factorial")
SMILES = "data/GDSC_drug_merge_pubchem_dropNA_MACCS_AACDR_extended.csv"


@pytest.mark.integration
def test_drug_heldout_no_overlap():
    if not (ROOT / "splits" / "development_rows.csv").is_file():
        pytest.skip("development rows missing")
    assign, qc = build_drug_heldout(ROOT, drug_smiles_path=SMILES)
    assert set(assign.fold_id) == set(range(5))
    for fold in range(5):
        f = assign[assign.fold_id == fold]
        tr = set(f.loc[f.split_role == "train", "normalized_drug_id"])
        va = set(f.loc[f.split_role == "val", "normalized_drug_id"])
        assert not (tr & va)
    val = assign[assign.split_role == "val"]
    assert not val["_row_id"].duplicated().any()
    it = pd.read_csv(ROOT / "splits" / "internal_test_split.csv")
    assert not set(assign["_row_id"]) & set(it["_row_id"])
    assert all(r["valid_drugmacro_drugs"] >= 3 for r in qc)
