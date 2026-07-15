"""Round 19 graph/bond feature builders and disk cache helpers."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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


def legacy_graph_metadata(smiles: str) -> Dict[str, Any]:
    """Describe the exact molecule consumed by legacy ``smile_to_graph``.

    This intentionally mirrors its parse/largest-fragment behavior and does not
    canonicalize, neutralize, or otherwise change the input.
    """
    text = str(smiles)
    mol = Chem.MolFromSmiles(text)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {text!r}")
    original_fragments = list(Chem.GetMolFrags(mol, asMols=False))
    selected_fragment_index = 0
    selected_original_indices = tuple(range(mol.GetNumAtoms()))
    graph_mol = mol
    if "." in text and original_fragments:
        # Same stable first-max tie behavior as tools.dataprocess.smile_to_graph.
        selected_fragment_index = max(
            range(len(original_fragments)), key=lambda index: len(original_fragments[index])
        )
        selected_original_indices = tuple(original_fragments[selected_fragment_index])
        fragment_mols = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
        graph_mol = fragment_mols[selected_fragment_index]

    atoms = []
    for graph_index, atom in enumerate(graph_mol.GetAtoms()):
        atoms.append(
            {
                "graph_atom_index": graph_index,
                "original_atom_index": int(selected_original_indices[graph_index]),
                "symbol": atom.GetSymbol(),
                "atomic_number": atom.GetAtomicNum(),
                "formal_charge": atom.GetFormalCharge(),
                "is_aromatic": atom.GetIsAromatic(),
                "is_in_ring": atom.IsInRing(),
                "isotope": atom.GetIsotope(),
                "atom_map_number": atom.GetAtomMapNum(),
            }
        )
    bonds = []
    for bond_index, bond in enumerate(graph_mol.GetBonds()):
        bonds.append(
            {
                "graph_bond_index": bond_index,
                "begin_atom_index": bond.GetBeginAtomIdx(),
                "end_atom_index": bond.GetEndAtomIdx(),
                "bond_type": str(bond.GetBondType()),
                "is_aromatic": bond.GetIsAromatic(),
                "is_conjugated": bond.GetIsConjugated(),
                "is_in_ring": bond.IsInRing(),
                "stereo": str(bond.GetStereo()),
            }
        )
    return {
        "legacy_input_smiles": text,
        "graph_smiles": Chem.MolToSmiles(graph_mol, canonical=False),
        "graph_smiles_canonical_identity": Chem.MolToSmiles(graph_mol, canonical=True),
        "desalt_applied": "." in text,
        "fragment_count": len(original_fragments),
        "selected_fragment_index": selected_fragment_index,
        "selected_original_atom_indices": list(selected_original_indices),
        "atom_metadata": atoms,
        "bond_metadata": bonds,
        "_graph_mol": graph_mol,
    }


def smile_to_graph_with_bonds(smiles: str) -> Tuple[int, List[List[float]], List[List[int]], List[List[float]]]:
    """Atom78 graph + undirected edge_attr aligned to directed edge_index."""
    metadata = legacy_graph_metadata(smiles)
    mol2 = metadata["_graph_mol"]
    # Call legacy with the original string so atom order and desalting semantics
    # are exactly the same as GIN.
    c_size, features, edge_index = smile_to_graph(smiles)
    if c_size != mol2.GetNumAtoms():
        raise AssertionError("legacy graph atom count disagrees with metadata molecule")

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
            raise AssertionError(f"Legacy edge topology has no matching bond: {key}")
        edge_attrs.append(undirected[key])
    return c_size, features, edge_index, edge_attrs


def _edge_index_tensor(edge_index) -> torch.Tensor:
    """Always return LongTensor of shape (2, E), including empty (2, 0)."""
    if not edge_index:
        return torch.empty((2, 0), dtype=torch.long)
    # smile_to_graph returns list of [u, v] pairs → (E, 2).T → (2, E)
    return torch.tensor(edge_index, dtype=torch.long).t().contiguous()


def build_pyg_data(smiles: str, *, with_bonds: bool) -> Data:
    metadata = legacy_graph_metadata(smiles)
    if with_bonds:
        _, features, edge_index, edge_attrs = smile_to_graph_with_bonds(smiles)
        ei = _edge_index_tensor(edge_index)
        if edge_attrs:
            ea = torch.tensor(np.asarray(edge_attrs, dtype=np.float32))
        else:
            ea = torch.empty((0, BOND_FEATURE_DIM), dtype=torch.float32)
        if ea.shape[0] != ei.shape[1]:
            raise RuntimeError(f"edge_attr rows {ea.shape[0]} != edge_index cols {ei.shape[1]}")
        data = Data(
            x=torch.tensor(np.asarray(features, dtype=np.float32)),
            edge_index=ei,
            edge_attr=ea,
        )
    else:
        _, features, edge_index = smile_to_graph(smiles)
        data = Data(
            x=torch.tensor(np.asarray(features, dtype=np.float32)),
            edge_index=_edge_index_tensor(edge_index),
        )
    # Plain attributes survive Data access and preserve per-encoder atom order.
    data.legacy_input_smiles = metadata["legacy_input_smiles"]
    data.graph_smiles = metadata["graph_smiles"]
    data.graph_smiles_canonical_identity = metadata["graph_smiles_canonical_identity"]
    data.graph_metadata = {key: value for key, value in metadata.items() if key != "_graph_mol"}
    return data


def graph_smiles_identity(smiles: str) -> str:
    metadata = legacy_graph_metadata(smiles)
    payload = json.dumps(
        {
            "input": metadata["legacy_input_smiles"],
            "actual": metadata["graph_smiles"],
            "canonical": metadata["graph_smiles_canonical_identity"],
            "atom_map": metadata["selected_original_atom_indices"],
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


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
