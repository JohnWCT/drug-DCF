"""Unified BioCDA forward output contract."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class BioCDAOutput:
    logits: torch.Tensor
    probabilities: Optional[torch.Tensor] = None

    sample_representation: Optional[torch.Tensor] = None
    omics_latent: Optional[torch.Tensor] = None
    biological_context: Optional[torch.Tensor] = None

    drug_representation: Optional[torch.Tensor] = None
    node_embeddings: Optional[torch.Tensor] = None

    atom_attention: Optional[torch.Tensor] = None
    atom_attention_logits: Optional[torch.Tensor] = None
    atom_mask: Optional[torch.Tensor] = None
    attention_probabilities_used: Optional[torch.Tensor] = None

    atom_batch_index: Optional[torch.Tensor] = None
    atom_ptr: Optional[torch.Tensor] = None
    model_atom_index: Optional[torch.Tensor] = None
    original_atom_index: Optional[torch.Tensor] = None
    rdkit_atom_index: Optional[torch.Tensor] = None

    architecture_version: Optional[str] = None
