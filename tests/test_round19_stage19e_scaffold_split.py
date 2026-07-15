"""Scaffold-held-out split QC."""
from __future__ import annotations

from pathlib import Path

import pytest

from tools.round19_scaffold_groups import murcko_scaffold_id
from tools.round19_stage19e_splits import build_scaffold_heldout

ROOT = Path("result/optimization_runs/round19_factorial")
SMILES = "data/GDSC_drug_merge_pubchem_dropNA_MACCS_AACDR_extended.csv"


def test_acyclic_fallback_unique():
    a = murcko_scaffold_id("CCCC")
    b = murcko_scaffold_id("CCCCCC")
    assert a.startswith("ACYCLIC:")
    assert b.startswith("ACYCLIC:")
    assert a != b
    ring = murcko_scaffold_id("c1ccccc1")
    assert ring.startswith("MURCKO:")


@pytest.mark.integration
def test_scaffold_heldout_no_overlap():
    if not (ROOT / "splits" / "development_rows.csv").is_file():
        pytest.skip("development rows missing")
    assign, qc = build_scaffold_heldout(ROOT, drug_smiles_path=SMILES)
    for fold in range(5):
        f = assign[assign.fold_id == fold]
        for col in ("scaffold_id", "DRUG_NAME", "canonical_smiles"):
            tr = set(f.loc[f.split_role == "train", col].astype(str))
            va = set(f.loc[f.split_role == "val", col].astype(str))
            assert not (tr & va), col
    assert all(r["valid_drugmacro_drugs"] >= 3 for r in qc)
