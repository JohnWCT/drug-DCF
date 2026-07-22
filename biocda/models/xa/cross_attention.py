"""Omics-conditioned atom-level cross-attention (BioCDA-XA v2)."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import Tensor, nn

from biocda.utils.masked_softmax import masked_softmax


@dataclass
class StackedCrossAttentionOutput:
    final_query: Tensor
    attention_logits: Tensor  # [L,B,H,1,N]
    attention_probabilities: Tensor  # [L,B,H,1,N] pre-dropout
    atom_mask: Tensor  # [B,N]
    dense_atom_tokens: Tensor  # [B,N,d_model]


class CrossAttentionBlock(nn.Module):
    """One sample-query ← atom-token cross-attention + FFN block."""

    def __init__(
        self,
        d_model: int = 128,
        num_heads: int = 4,
        ffn_dim: int = 256,
        attention_dropout: float = 0.1,
        block_dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.head_dim = self.d_model // self.num_heads

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.wo = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(attention_dropout)
        self.block_dropout = nn.Dropout(block_dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(block_dropout),
            nn.Linear(ffn_dim, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in (self.wq, self.wk, self.wv, self.wo):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
        self,
        query: Tensor,
        atom_tokens: Tensor,
        atom_mask: Tensor,
        *,
        attention_override: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            query: [B,1,d]
            atom_tokens: [B,N,d]
            atom_mask: [B,N] bool
            attention_override: optional [B,H,1,N] probabilities (eval/interpretability)
        Returns:
            updated_query [B,1,d], logits [B,H,1,N], probs_raw [B,H,1,N]
        """
        batch_size, _, _ = query.shape
        max_atoms = atom_tokens.shape[1]

        q = self.wq(query).view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.wk(atom_tokens).view(batch_size, max_atoms, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.wv(atom_tokens).view(batch_size, max_atoms, self.num_heads, self.head_dim).transpose(1, 2)

        scale = math.sqrt(self.head_dim)
        logits = torch.matmul(q, k.transpose(-2, -1)) / scale  # [B,H,1,N]
        head_mask = atom_mask.unsqueeze(1).unsqueeze(2).expand(-1, self.num_heads, 1, -1)
        probs_raw = masked_softmax(logits, head_mask, dim=-1)

        if attention_override is not None:
            probs_used = attention_override
        else:
            probs_used = self.attn_dropout(probs_raw) if self.training else probs_raw

        attended = torch.matmul(probs_used, v)  # [B,H,1,head]
        attended = attended.transpose(1, 2).contiguous().view(batch_size, 1, self.d_model)
        attended = self.wo(attended)

        x = self.norm1(query + self.block_dropout(attended))
        x = self.norm2(x + self.block_dropout(self.ffn(x)))
        return x, logits, probs_raw


class OmicsAtomCrossAttentionStack(nn.Module):
    """Two-layer sample-to-atom cross-attention (fixed XA v2)."""

    def __init__(
        self,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        ffn_dim: int = 256,
        attention_dropout: float = 0.1,
        block_dropout: float = 0.2,
        node_dim: int = 32,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.num_layers = int(num_layers)
        self.atom_projection = nn.Sequential(
            nn.Linear(node_dim, d_model),
            nn.LayerNorm(d_model),
        )
        self.layers = nn.ModuleList(
            [
                CrossAttentionBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    ffn_dim=ffn_dim,
                    attention_dropout=attention_dropout,
                    block_dropout=block_dropout,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        query: Tensor,
        node_embeddings: Tensor,
        batch_index: Tensor,
        *,
        attention_override: Optional[Tensor] = None,
    ) -> StackedCrossAttentionOutput:
        from torch_geometric.utils import to_dense_batch

        dense_nodes, atom_mask = to_dense_batch(node_embeddings, batch_index)
        if atom_mask.sum(dim=-1).eq(0).any():
            raise ValueError("OmicsAtomCrossAttentionStack: empty graph in batch")
        atom_tokens = self.atom_projection(dense_nodes)

        logits_layers = []
        probs_layers = []
        q = query
        for layer in self.layers:
            q, logits, probs = layer(
                q,
                atom_tokens,
                atom_mask,
                attention_override=attention_override,
            )
            logits_layers.append(logits)
            probs_layers.append(probs)

        return StackedCrossAttentionOutput(
            final_query=q,
            attention_logits=torch.stack(logits_layers, dim=0),
            attention_probabilities=torch.stack(probs_layers, dim=0),
            atom_mask=atom_mask,
            dense_atom_tokens=atom_tokens,
        )
