"""Atom metadata batch contract for dense attention alignment."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor


@dataclass
class AtomMetadataBatch:
    model_atom_index: Tensor
    batch_index: Tensor
    atom_ptr: Tensor
    original_atom_index: Optional[Tensor] = None
    rdkit_atom_index: Optional[Tensor] = None


def build_atom_ptr(batch_index: Tensor, num_graphs: Optional[int] = None) -> Tensor:
    """CSR-style pointer [B+1] into flat atom rows."""
    if num_graphs is None:
        num_graphs = int(batch_index.max().item()) + 1 if batch_index.numel() else 0
    counts = torch.bincount(batch_index, minlength=num_graphs)
    ptr = torch.zeros(num_graphs + 1, dtype=torch.long, device=batch_index.device)
    if num_graphs > 0:
        ptr[1:] = counts.cumsum(0)
    return ptr
