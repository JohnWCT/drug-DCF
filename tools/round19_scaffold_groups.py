"""Murcko scaffold grouping for Round 19E shift validation.

Uses the same desalt + largest-fragment preprocessing as graph encoding
(`tools.dataprocess.smile_to_graph`) so scaffold groups match graph cache SMILES.
"""
from __future__ import annotations

import hashlib
from typing import Dict, Iterable, Optional

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


def graph_canonical_smiles(smiles: str) -> Optional[str]:
    """Canonical SMILES after desalt / largest-fragment (graph-encoder aligned)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    if "." in smiles:
        frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
        if frags:
            mol = max(frags, key=lambda m: m.GetNumAtoms())
    return Chem.MolToSmiles(mol, canonical=True)


def canonicalize_smiles(smiles: str) -> Optional[str]:
    return graph_canonical_smiles(smiles)


def murcko_scaffold_id(smiles: str) -> str:
    """Return MURCKO:<scaffold> or ACYCLIC:<sha256(canonical)> (never a shared empty bucket)."""
    canon = graph_canonical_smiles(smiles)
    if canon is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")
    mol = Chem.MolFromSmiles(canon)
    if mol is None:
        raise ValueError(f"Invalid canonical SMILES: {canon!r}")
    try:
        core = MurckoScaffold.GetScaffoldForMol(mol)
        s = Chem.MolToSmiles(core, canonical=True) if core is not None else ""
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Scaffold extraction failed for {smiles!r}: {exc}") from exc
    if s:
        return f"MURCKO:{s}"
    digest = hashlib.sha256(canon.encode("utf-8")).hexdigest()
    return f"ACYCLIC:{digest}"


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
