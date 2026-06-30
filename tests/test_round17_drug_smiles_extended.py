"""Round 17 Phase 0: AACDR extended SMILES coverage."""

from __future__ import annotations

import os

import pandas as pd
import pytest

from tools.finetune_tcga_eval import (
    FIXED_DRUG_SMILES_AACDR_EXTENDED,
    FIXED_TCGA_EVAL_AACDR_GDSC_INTERSECT,
    FIXED_TCGA_EVAL_AACDR_TCGA_ONLY,
)
from tools.optimization_runner import FIXED_DRUG_SMILES_AACDR_EXTENDED as RUNNER_DEFAULT_SMILES

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEGACY_SMILES = "data/GDSC_drug_merge_pubchem_dropNA_MACCS.csv"
EXTENDED_SMILES = FIXED_DRUG_SMILES_AACDR_EXTENDED


def _resolve(rel_path: str) -> str:
    return os.path.join(PROJECT_ROOT, rel_path)


def _smiles_lookup(smiles_df: pd.DataFrame) -> set[str]:
    lookup = set(smiles_df.index.astype(str).str.strip().str.lower())
    if "DRUG_NAME" in smiles_df.columns:
        lookup |= set(smiles_df["DRUG_NAME"].dropna().astype(str).str.strip().str.lower())
    return lookup


@pytest.fixture(scope="module")
def extended_smiles_df() -> pd.DataFrame:
    path = _resolve(EXTENDED_SMILES)
    if not os.path.isfile(path):
        pytest.skip(f"missing extended SMILES table: {EXTENDED_SMILES}")
    return pd.read_csv(path, index_col=0)


def test_round17_extended_smiles_has_smiles_column(extended_smiles_df: pd.DataFrame):
    assert "SMILES" in extended_smiles_df.columns
    assert extended_smiles_df["SMILES"].notna().all()


def test_round17_extended_smiles_superset_of_legacy(extended_smiles_df: pd.DataFrame):
    legacy_path = _resolve(LEGACY_SMILES)
    if not os.path.isfile(legacy_path):
        pytest.skip(f"missing legacy SMILES table: {LEGACY_SMILES}")
    legacy_df = pd.read_csv(legacy_path, index_col=0)
    extended_lookup = _smiles_lookup(extended_smiles_df)
    missing = sorted(_smiles_lookup(legacy_df) - extended_lookup)
    assert not missing, f"extended SMILES missing legacy drugs: {missing[:10]}"


@pytest.mark.parametrize(
    "tcga_path",
    [FIXED_TCGA_EVAL_AACDR_TCGA_ONLY, FIXED_TCGA_EVAL_AACDR_GDSC_INTERSECT],
)
def test_aacdr_tcga_drugs_all_have_smiles(extended_smiles_df: pd.DataFrame, tcga_path: str):
    full_tcga_path = _resolve(tcga_path)
    if not os.path.isfile(full_tcga_path):
        pytest.skip(f"missing AACDR TCGA file: {tcga_path}")
    tcga_df = pd.read_csv(full_tcga_path)
    lookup = _smiles_lookup(extended_smiles_df)
    drugs = tcga_df["drug_name"].astype(str).str.strip().str.lower().unique()
    missing = sorted(d for d in drugs if d not in lookup)
    assert not missing, f"{tcga_path} drugs missing SMILES: {missing}"


def test_optimization_runner_default_drug_smiles_path():
    assert RUNNER_DEFAULT_SMILES == EXTENDED_SMILES


def test_append_drug_smiles_arg_forwards_path():
    from tools.optimization_runner import _append_drug_smiles_arg

    cmd: list[str] = []
    _append_drug_smiles_arg(cmd, EXTENDED_SMILES)
    assert cmd == ["--drug-smiles-path", _resolve(EXTENDED_SMILES)]
