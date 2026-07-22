"""BioCDA-XA v2 output contract."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class BioCDAXAOutput:
    logits: torch.Tensor
    probabilities: torch.Tensor

    base_omics_latent: Optional[torch.Tensor] = None
    prototype_context: Optional[torch.Tensor] = None
    sample_features: Optional[torch.Tensor] = None

    initial_query: Optional[torch.Tensor] = None
    final_query: Optional[torch.Tensor] = None

    node_embeddings: Optional[torch.Tensor] = None
    dense_atom_tokens: Optional[torch.Tensor] = None

    attention_logits: Optional[torch.Tensor] = None
    attention_probabilities: Optional[torch.Tensor] = None
    atom_mask: Optional[torch.Tensor] = None

    atom_batch_index: Optional[torch.Tensor] = None
    atom_ptr: Optional[torch.Tensor] = None

    model_atom_index: Optional[torch.Tensor] = None
    original_atom_index: Optional[torch.Tensor] = None
    rdkit_atom_index: Optional[torch.Tensor] = None

    architecture_version: str = "biocda-xa-v2"
