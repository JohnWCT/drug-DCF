#!/usr/bin/env python3
"""GINE-only bond-feature interventions preserving graph topology."""
from __future__ import annotations

import torch


def bond_feature_zero(data, bond_indices, *, encoder_type: str):
    if str(encoder_type).lower() != "gine":
        raise ValueError("bond intervention is applicable only to GINE")
    if not hasattr(data, "edge_attr") or data.edge_attr is None:
        raise ValueError("GINE bond intervention requires edge_attr")
    out = data.clone()
    out.edge_attr = data.edge_attr.clone()
    selected = set(map(int, bond_indices))
    edge_index = data.edge_index
    directed = [
        idx for idx in range(edge_index.shape[1])
        if min(int(edge_index[0, idx]), int(edge_index[1, idx])) * data.x.shape[0]
        + max(int(edge_index[0, idx]), int(edge_index[1, idx])) in selected
    ]
    if directed:
        out.edge_attr[
            torch.as_tensor(directed, dtype=torch.long, device=out.edge_attr.device)
        ] = 0
    if not torch.equal(out.edge_index, data.edge_index):
        raise AssertionError("bond feature intervention changed topology")
    return out


def canonical_bond_id(left: int, right: int, num_nodes: int) -> int:
    return min(left, right) * int(num_nodes) + max(left, right)
