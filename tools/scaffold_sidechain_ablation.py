#!/usr/bin/env python3
"""Bemis-Murcko scaffold/side-chain atom partitions for feature-zero ablation."""
from __future__ import annotations

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


def scaffold_sidechain_partition(smiles: str) -> dict[str, list[int]]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"invalid SMILES: {smiles!r}")
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    matches = mol.GetSubstructMatches(scaffold) if scaffold.GetNumAtoms() else ()
    scaffold_atoms = set(matches[0]) if matches else set()
    all_atoms = set(range(mol.GetNumAtoms()))
    return {
        "scaffold": sorted(scaffold_atoms),
        "sidechain": sorted(all_atoms - scaffold_atoms),
    }


def ablation_rows(smiles: str) -> list[dict]:
    partition = scaffold_sidechain_partition(smiles)
    return [
        {"ablation": name, "atom_indices": indices, "applicable": bool(indices)}
        for name, indices in partition.items()
    ]
