"""Round 19 fusion models with fixed adapters and compatibility matrix."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor, nn
from torch_geometric.utils import to_dense_batch

from tools.cross_attention_switch import CrossAttentionSwitch
from tools.round18_response_head import Round18ResponseHead
from tools.transformer_switch import TransformerSwitch


COMPATIBLE_CELLS: List[Tuple[str, str]] = [
    ("D0", "P0"),
    ("D0", "P1"),
    ("D0", "P2"),
    ("D1", "P0"),
    ("D1", "P1"),
    ("D2", "P0"),
    ("D2", "P1"),
    ("D2", "P2"),
    ("D3", "P0"),
    ("D3", "P1"),
    ("D3", "P2"),
    ("D4", "P0"),
    ("D4", "P1"),
]


def assert_compatible(drug_id: str, predictor_id: str) -> None:
    if (drug_id, predictor_id) not in COMPATIBLE_CELLS:
        raise AssertionError(f"Incompatible drug×predictor cell: {drug_id}×{predictor_id}")


def _reject_atom_attention(predictor_id: str) -> None:
    raise ValueError(
        f"{predictor_id} has no atom-level attention; only P2 may export real atom attention"
    )


class AdapterMLPFusion(nn.Module):
    """P0: omics/drug adapters (64+64) then concat."""

    def __init__(self, omics_dim: int, drug_dim: int, adapter_dim: int = 64):
        super().__init__()
        self.omics_adapter = nn.Sequential(nn.Linear(omics_dim, adapter_dim), nn.LayerNorm(adapter_dim))
        self.drug_adapter = nn.Sequential(nn.Linear(drug_dim, adapter_dim), nn.LayerNorm(adapter_dim))
        self.output_dim = adapter_dim * 2

    def forward(
        self,
        omics_vector: Tensor,
        drug_vector: Tensor,
        *,
        return_attention: bool = False,
        return_interpretability: bool = False,
    ) -> Tensor:
        if return_attention or return_interpretability:
            _reject_atom_attention("P0")
        return torch.cat([self.omics_adapter(omics_vector), self.drug_adapter(drug_vector)], dim=-1)


class PooledTransformerFusionR19(nn.Module):
    """P1: two-token compact Transformer (omics + drug)."""

    def __init__(
        self,
        omics_dim: int,
        drug_dim: int,
        d_model: int = 64,
        n_heads: int = 4,
        num_layers: int = 1,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.omics_proj = nn.Sequential(nn.Linear(omics_dim, d_model), nn.LayerNorm(d_model))
        self.drug_proj = nn.Sequential(nn.Linear(drug_dim, d_model), nn.LayerNorm(d_model))
        self.token_type = nn.Embedding(2, d_model)
        self.encoder = TransformerSwitch(
            d_model=d_model,
            n_heads=n_heads,
            num_encoder_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            attn_dropout=dropout,
            temperature=1.0,
            use_mask=False,
            use_positional_encoding=False,
            attn_out_mlp=True,
            attn_out_activation="GELU",
            ffn_activation="ReLU",
        )
        self.output_dim = d_model

    def forward(
        self,
        omics_vector: Tensor,
        drug_vector: Tensor,
        *,
        return_attention: bool = False,
        return_interpretability: bool = False,
    ) -> Tensor:
        if return_attention or return_interpretability:
            _reject_atom_attention("P1")
        omics_tok = self.omics_proj(omics_vector)
        drug_tok = self.drug_proj(drug_vector)
        tokens = torch.stack([omics_tok, drug_tok], dim=1)
        type_ids = torch.tensor([0, 1], device=tokens.device).unsqueeze(0).expand(tokens.size(0), -1)
        tokens = tokens + self.token_type(type_ids)
        encoded, _ = self.encoder(tokens, attn_mask=None)
        return encoded[:, 0, :]


class AtomCrossAttentionFusionR19(nn.Module):
    """P2: X3 pure atom cross-attention (omics CLS queries atom nodes)."""

    def __init__(
        self,
        omics_dim: int,
        node_dim: int,
        d_model: int = 128,
        n_heads: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.omics_proj = nn.Sequential(nn.Linear(omics_dim, d_model), nn.LayerNorm(d_model))
        self.atom_proj = nn.Sequential(nn.Linear(node_dim, d_model), nn.LayerNorm(d_model))
        self.token_type = nn.Embedding(2, d_model)
        self.cross_attn = CrossAttentionSwitch(
            d_model=d_model,
            n_heads=n_heads,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            attn_dropout=dropout,
            temperature=1.0,
        )
        self.output_dim = d_model

    def forward(
        self,
        omics_vector: Tensor,
        node_embeddings: Tensor,
        batch_index: Tensor,
        return_attention: bool = False,
        return_interpretability: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Tensor], Dict[str, Tensor]]:
        omics_cls = self.omics_proj(omics_vector).unsqueeze(1)
        omics_cls = omics_cls + self.token_type.weight[0].view(1, 1, -1)
        atom_dense, atom_valid = to_dense_batch(node_embeddings, batch=batch_index)
        atom_tokens = self.atom_proj(atom_dense) + self.token_type.weight[1].view(1, 1, -1)
        key_padding_mask = ~atom_valid
        if return_attention or return_interpretability:
            updated, attn = self.cross_attn(
                omics_cls, atom_tokens, key_padding_mask=key_padding_mask, return_attention=True
            )
            if attn.ndim != 5 or attn.shape[3] != 1:
                raise RuntimeError(f"atom attention must be [layers,batch,heads,1,atoms], got {attn.shape}")
            primary = attn[-1].mean(dim=1).squeeze(1)
            if return_interpretability:
                return {
                    "representation": updated.squeeze(1),
                    "attention_raw": attn,
                    "attention_primary": primary,
                    "atom_valid_mask": atom_valid,
                }
            return updated.squeeze(1), attn
        updated = self.cross_attn(
            omics_cls, atom_tokens, key_padding_mask=key_padding_mask, return_attention=False
        )
        return updated.squeeze(1)


def build_predictor(
    predictor_id: str,
    *,
    omics_dim: int,
    drug_dim: int,
    node_dim: int = 32,
) -> nn.Module:
    pid = str(predictor_id).upper()
    if pid == "P0":
        return AdapterMLPFusion(omics_dim=omics_dim, drug_dim=drug_dim, adapter_dim=64)
    if pid == "P1":
        return PooledTransformerFusionR19(
            omics_dim=omics_dim,
            drug_dim=drug_dim,
            d_model=64,
            n_heads=4,
            num_layers=1,
            dim_feedforward=128,
            dropout=0.1,
        )
    if pid == "P2":
        return AtomCrossAttentionFusionR19(
            omics_dim=omics_dim,
            node_dim=node_dim,
            d_model=128,
            n_heads=4,
            num_layers=2,
            dim_feedforward=256,
            dropout=0.2,
        )
    raise ValueError(f"Unknown predictor_id={predictor_id!r}")


def build_predictor_and_head(
    predictor_id: str,
    *,
    omics_dim: int,
    drug_dim: int,
    node_dim: int = 32,
) -> Tuple[nn.Module, Round18ResponseHead]:
    fusion = build_predictor(
        predictor_id, omics_dim=omics_dim, drug_dim=drug_dim, node_dim=node_dim
    )
    head = Round18ResponseHead(input_dim=fusion.output_dim)
    return fusion, head
