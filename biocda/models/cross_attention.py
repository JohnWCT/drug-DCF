"""Sample-to-atom cross-attention (BioCDA core)."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn
from torch_geometric.utils import to_dense_batch

from biocda.utils.masked_softmax import masked_softmax


@dataclass
class CrossAttentionOutput:
    drug_representation: Tensor
    attention_logits: Tensor
    attention_probabilities: Tensor
    attention_probabilities_used: Optional[Tensor]
    atom_mask: Tensor


class SampleAtomCrossAttention(nn.Module):
    """Patient-conditioned atom attention: S queries atom node embeddings."""

    def __init__(
        self,
        sample_dim: int,
        node_dim: int,
        attention_dim: int = 64,
        num_heads: int = 4,
        dropout: float = 0.1,
        temperature: float = 1.0,
        use_bias: bool = True,
    ) -> None:
        super().__init__()
        if attention_dim % num_heads != 0:
            raise ValueError("attention_dim must be divisible by num_heads")
        self.attention_dim = int(attention_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.attention_dim // self.num_heads
        self.temperature = float(temperature)

        self.query_projection = nn.Linear(sample_dim, attention_dim, bias=use_bias)
        self.key_projection = nn.Linear(node_dim, attention_dim, bias=use_bias)
        self.value_projection = nn.Linear(node_dim, attention_dim, bias=use_bias)
        self.output_projection = nn.Linear(attention_dim, attention_dim, bias=use_bias)
        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in (self.query_projection, self.key_projection, self.value_projection, self.output_projection):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        sample_repr: Tensor,
        node_embeddings: Tensor,
        batch_index: Tensor,
    ) -> CrossAttentionOutput:
        dense_nodes, atom_mask = to_dense_batch(node_embeddings, batch_index)
        if atom_mask.sum(dim=-1).eq(0).any():
            raise ValueError("SampleAtomCrossAttention: empty graph in batch")

        batch_size, max_atoms, _ = dense_nodes.shape
        query = self.query_projection(sample_repr)
        key = self.key_projection(dense_nodes)
        value = self.value_projection(dense_nodes)

        query = query.view(batch_size, self.num_heads, self.head_dim)
        key = key.view(batch_size, max_atoms, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, max_atoms, self.num_heads, self.head_dim).transpose(1, 2)

        scale = math.sqrt(self.head_dim) * self.temperature
        attention_logits = torch.einsum("bhd,bhnd->bhn", query, key) / scale

        head_mask = atom_mask.unsqueeze(1).expand(-1, self.num_heads, -1)
        attention_probabilities = masked_softmax(attention_logits, head_mask, dim=-1)
        attention_probabilities_used = self.dropout(attention_probabilities) if self.training else attention_probabilities

        attended = torch.einsum("bhn,bhnd->bhd", attention_probabilities_used, value)
        attended = attended.reshape(batch_size, self.attention_dim)
        drug_representation = self.output_projection(attended)

        return CrossAttentionOutput(
            drug_representation=drug_representation,
            attention_logits=attention_logits,
            attention_probabilities=attention_probabilities,
            attention_probabilities_used=attention_probabilities_used if self.training else None,
            atom_mask=atom_mask,
        )
