"""Murcko scaffold grouping for Round 19 shift validation."""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


EMPTY_SCAFFOLD_FALLBACK = "SCAFFOLD_EMPTY"


def canonicalize_smiles(smiles: str) -> Optional[str]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def murcko_scaffold_id(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")
    try:
        core = MurckoScaffold.GetScaffoldForMol(mol)
        s = Chem.MolToSmiles(core, canonical=True) if core is not None else ""
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Scaffold extraction failed for {smiles!r}: {exc}") from exc
    if not s:
        return EMPTY_SCAFFOLD_FALLBACK
    return s


def build_scaffold_map(drug_to_smiles: Dict[str, str]) -> Dict[str, str]:
    out = {}
    for drug, smiles in drug_to_smiles.items():
        out[str(drug)] = murcko_scaffold_id(str(smiles))
    return out


def assert_scaffold_not_split_across_folds(
    rows: Iterable[dict],
    *,
    scaffold_col: str = "scaffold_id",
    fold_col: str = "fold_id",
) -> None:
    seen = {}
    for r in rows:
        sc = str(r[scaffold_col])
        fold = int(r[fold_col])
        if sc in seen and seen[sc] != fold:
            raise AssertionError(f"Scaffold {sc} appears in folds {seen[sc]} and {fold}")
        seen[sc] = fold
