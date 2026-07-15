"""Tests for Round 19D confirmation splits."""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import pytest
from tools.round19_cv_splits import build_round19d_splits, validate_round19d_assignments

ROOT = Path("result/optimization_runs/round19_factorial")

@pytest.mark.integration
def test_round19d_splits_qc():
    if not (ROOT / "splits" / "development_rows.csv").is_file():
        pytest.skip("development rows missing")
    paths = build_round19d_splits(ROOT, split_seeds=[52], n_folds=5)
    assert "52" in paths
    assign = pd.read_csv(paths["52"])
    dev = pd.read_csv(ROOT / "splits" / "development_rows.csv")
    it = pd.read_csv(ROOT / "splits" / "internal_test_split.csv")
    validate_round19d_assignments(assign, development=dev, internal_test=it, split_seed=52, n_folds=5)
    assert "split_role" in assign.columns
    assert set(assign.fold_id) == {0,1,2,3,4}
