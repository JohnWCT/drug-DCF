"""Drug graph helpers and batch contracts."""
from __future__ import annotations

from typing import List, Optional

import torch
from torch_geometric.data import Batch, Data


def make_chain_graph(num_atoms: int, *, feature_dim: int = 78, drug_id: str = "drug") -> Data:
    """Synthetic chain graph for unit tests."""
    if num_atoms <= 0:
        raise ValueError("num_atoms must be positive")
    x = torch.randn(num_atoms, feature_dim)
    if num_atoms == 1:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    else:
        src = torch.arange(num_atoms - 1)
        dst = src + 1
        edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0)
    data = Data(
        x=x,
        edge_index=edge_index,
        model_atom_index=torch.arange(num_atoms, dtype=torch.long),
        drug_id=drug_id,
    )
    return data


def batch_drug_graphs(graphs: List[Data]) -> Batch:
    return Batch.from_data_list(graphs)
