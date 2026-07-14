"""Round 19 graph/bond feature builders and disk cache helpers."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from rdkit import Chem
from torch_geometric.data import Data

from tools.dataprocess import smile_to_graph

BOND_TYPE_LIST = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]

STEREO_LIST = [
    Chem.rdchem.BondStereo.STEREONONE,
    Chem.rdchem.BondStereo.STEREOANY,
    Chem.rdchem.BondStereo.STEREOZ,
    Chem.rdchem.BondStereo.STEREOE,
    Chem.rdchem.BondStereo.STEREOCIS,
    Chem.rdchem.BondStereo.STEREOTRANS,
]

BOND_FEATURE_DIM = len(BOND_TYPE_LIST) + 2 + len(STEREO_LIST)  # type + conjugated + in_ring + stereo
BOND_SCHEMA = {
    "bond_type_onehot": [str(b) for b in BOND_TYPE_LIST],
    "is_conjugated": True,
    "is_in_ring": True,
    "stereo_onehot": [str(s) for s in STEREO_LIST],
    "bond_feature_dim": BOND_FEATURE_DIM,
}


def bond_schema_hash() -> str:
    payload = json.dumps(BOND_SCHEMA, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _one_hot(x, choices: Sequence) -> List[float]:
    return [1.0 if x == c else 0.0 for c in choices]


def bond_features(bond: Chem.Bond) -> List[float]:
    feats = []
    feats.extend(_one_hot(bond.GetBondType(), BOND_TYPE_LIST))
    feats.append(1.0 if bond.GetIsConjugated() else 0.0)
    feats.append(1.0 if bond.IsInRing() else 0.0)
    feats.extend(_one_hot(bond.GetStereo(), STEREO_LIST))
    if len(feats) != BOND_FEATURE_DIM:
        raise RuntimeError(f"bond feature dim mismatch: {len(feats)} != {BOND_FEATURE_DIM}")
    return feats


def smile_to_graph_with_bonds(smiles: str) -> Tuple[int, List[List[float]], List[List[int]], List[List[float]]]:
    """Atom78 graph + undirected edge_attr aligned to directed edge_index."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")
    if "." in smiles:
        frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
        if frags:
            mol = max(frags, key=lambda m: m.GetNumAtoms())

    c_size, features, edge_index = smile_to_graph(Chem.MolToSmiles(mol))
    # Rebuild bond attrs from the same mol used for atoms.
    # smile_to_graph may have desalted; re-parse canonical desalted smiles for bond map.
    mol2 = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
    if mol2 is None:
        raise ValueError(f"Failed to rebuild mol for bonds: {smiles!r}")

    undirected: Dict[Tuple[int, int], List[float]] = {}
    for bond in mol2.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bf = bond_features(bond)
        undirected[(i, j)] = bf
        undirected[(j, i)] = bf

    edge_attrs: List[List[float]] = []
    for e1, e2 in edge_index:
        key = (int(e1), int(e2))
        if key not in undirected:
            # Fallback zeros if topology mismatch (should be rare)
            edge_attrs.append([0.0] * BOND_FEATURE_DIM)
        else:
            edge_attrs.append(undirected[key])
    return c_size, features, edge_index, edge_attrs


def build_pyg_data(smiles: str, *, with_bonds: bool) -> Data:
    if with_bonds:
        _, features, edge_index, edge_attrs = smile_to_graph_with_bonds(smiles)
        return Data(
            x=torch.tensor(np.asarray(features, dtype=np.float32)),
            edge_index=torch.tensor(np.asarray(edge_index, dtype=np.int64).T),
            edge_attr=torch.tensor(np.asarray(edge_attrs, dtype=np.float32)),
        )
    _, features, edge_index = smile_to_graph(smiles)
    return Data(
        x=torch.tensor(np.asarray(features, dtype=np.float32)),
        edge_index=torch.tensor(np.asarray(edge_index, dtype=np.int64).T),
    )


def cache_metadata(
    *,
    encoder_type: str,
    atom_feature_dim: int = 78,
    bond_feature_dim: Optional[int] = None,
    cache_version: str,
) -> Dict:
    try:
        import rdkit

        rdkit_version = rdkit.__version__
    except Exception:  # noqa: BLE001
        rdkit_version = "unknown"
    meta = {
        "encoder_type": encoder_type,
        "atom_feature_dim": int(atom_feature_dim),
        "bond_feature_dim": bond_feature_dim,
        "atom_schema_hash": "atom78_v1",
        "bond_schema_hash": bond_schema_hash() if bond_feature_dim else None,
        "rdkit_version": rdkit_version,
        "cache_version": cache_version,
        "bond_schema": BOND_SCHEMA if bond_feature_dim else None,
    }
    return meta


def ensure_cache_dir(root: Path, name: str, meta: Dict) -> Path:
    path = Path(root) / name
    path.mkdir(parents=True, exist_ok=True)
    (path / "cache_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return path
