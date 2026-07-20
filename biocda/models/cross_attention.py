"""Sample-to-atom cross-attention (BioCDA core)."""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch_geometric.utils import to_dense_batch


@dataclass
class CrossAttentionOutput:
    drug_representation: Tensor
    attention_probabilities: Tensor
    attention_logits: Tensor
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
    ) -> None:
        super().__init__()
        if attention_dim % num_heads != 0:
            raise ValueError("attention_dim must be divisible by num_heads")
        self.attention_dim = int(attention_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.attention_dim // self.num_heads

        self.query_projection = nn.Linear(sample_dim, attention_dim)
        self.key_projection = nn.Linear(node_dim, attention_dim)
        self.value_projection = nn.Linear(node_dim, attention_dim)
        self.output_projection = nn.Linear(attention_dim, attention_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        sample_repr: Tensor,
        node_embeddings: Tensor,
        batch_index: Tensor,
    ) -> CrossAttentionOutput:
        dense_nodes, atom_mask = to_dense_batch(node_embeddings, batch_index)
        batch_size, max_atoms, _ = dense_nodes.shape

        query = self.query_projection(sample_repr)
        key = self.key_projection(dense_nodes)
        value = self.value_projection(dense_nodes)

        query = query.view(batch_size, self.num_heads, self.head_dim)
        key = key.view(batch_size, max_atoms, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, max_atoms, self.num_heads, self.head_dim).transpose(1, 2)

        attention_logits = torch.einsum("bhd,bhnd->bhn", query, key) / math.sqrt(self.head_dim)
        attention_logits = attention_logits.masked_fill(~atom_mask.unsqueeze(1), float("-inf"))

        attention_probabilities = torch.softmax(attention_logits, dim=-1)
        attended = torch.einsum(
            "bhn,bhnd->bhd",
            self.dropout(attention_probabilities),
            value,
        )
        attended = attended.reshape(batch_size, self.attention_dim)
        drug_representation = self.output_projection(attended)

        return CrossAttentionOutput(
            drug_representation=drug_representation,
            attention_probabilities=attention_probabilities,
            attention_logits=attention_logits,
            atom_mask=atom_mask,
        )
