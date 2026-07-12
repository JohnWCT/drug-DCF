"""Round 18 fusion architectures (representation only; response head is separate)."""
from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import torch
from torch import Tensor, nn
from torch_geometric.utils import to_dense_batch

from tools.cross_attention_switch import CrossAttentionSwitch
from tools.round18_response_head import Round18ResponseHead
from tools.transformer_switch import TransformerSwitch


def build_omics_cls(omics_vector: Tensor, projector: nn.Module) -> Tensor:
    return projector(omics_vector).unsqueeze(1)


class OmicsProjector(nn.Module):
    def __init__(self, input_dim: int, d_model: int):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(input_dim, d_model), nn.LayerNorm(d_model))

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(x)


class PooledMLPFusion(nn.Module):
    """A0–A2: concat(omics, pooled GIN) representation."""

    def __init__(self, omics_dim: int, graph_dim: int):
        super().__init__()
        self.output_dim = omics_dim + graph_dim

    def forward(self, omics_vector: Tensor, graph_embedding: Tensor) -> Tensor:
        return torch.cat([omics_vector, graph_embedding], dim=-1)


class PooledTransformerFusion(nn.Module):
    """A3–A5: 2-token TransformerSwitch; return updated omics CLS."""

    def __init__(
        self,
        omics_dim: int,
        graph_dim: int,
        d_model: int = 128,
        n_heads: int = 4,
        num_layers: int = 1,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        attn_dropout: float = 0.1,
        temperature: float = 1.0,
        attn_out_mlp: bool = True,
        attn_out_activation: str = "GELU",
        ffn_activation: str = "ReLU",
        requested_use_mask: bool = True,
    ):
        super().__init__()
        self.omics_proj = OmicsProjector(omics_dim, d_model)
        self.drug_proj = nn.Sequential(nn.Linear(graph_dim, d_model), nn.LayerNorm(d_model))
        self.token_type = nn.Embedding(2, d_model)
        self.requested_use_mask = bool(requested_use_mask)
        self.effective_use_mask = False
        self.mask_reason = "two dense tokens without padding"
        self.encoder = TransformerSwitch(
            d_model=d_model,
            n_heads=n_heads,
            num_encoder_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            attn_dropout=attn_dropout,
            temperature=temperature,
            use_mask=False,
            use_positional_encoding=False,
            attn_out_mlp=attn_out_mlp,
            attn_out_activation=attn_out_activation,
            ffn_activation=ffn_activation,
        )
        self.output_dim = d_model

    def metadata(self) -> Dict:
        return {
            "requested_use_mask": self.requested_use_mask,
            "effective_use_mask": self.effective_use_mask,
            "mask_reason": self.mask_reason,
        }

    def forward(self, omics_vector: Tensor, graph_embedding: Tensor) -> Tensor:
        omics_tok = self.omics_proj(omics_vector)
        drug_tok = self.drug_proj(graph_embedding)
        tokens = torch.stack([omics_tok, drug_tok], dim=1)
        type_ids = torch.tensor([0, 1], device=tokens.device).unsqueeze(0).expand(tokens.size(0), -1)
        tokens = tokens + self.token_type(type_ids)
        encoded, _ = self.encoder(tokens, attn_mask=None)
        return encoded[:, 0, :]


class AtomCrossAttentionFusion(nn.Module):
    """C0/C1: omics CLS queries atom tokens (+ optional pooled residual)."""

    def __init__(
        self,
        omics_dim: int,
        node_dim: int,
        graph_dim: int,
        d_model: int = 128,
        n_heads: int = 4,
        num_layers: int = 1,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        attn_dropout: float = 0.1,
        temperature: float = 1.0,
        residual_mode: str = "pure",
    ):
        super().__init__()
        assert residual_mode in {"pure", "pooled_residual"}
        self.residual_mode = residual_mode
        self.omics_proj = OmicsProjector(omics_dim, d_model)
        self.atom_proj = nn.Sequential(nn.Linear(node_dim, d_model), nn.LayerNorm(d_model))
        self.token_type = nn.Embedding(2, d_model)
        self.cross_attn = CrossAttentionSwitch(
            d_model=d_model,
            n_heads=n_heads,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            attn_dropout=attn_dropout,
            temperature=temperature,
        )
        if residual_mode == "pooled_residual":
            self.graph_proj = nn.Sequential(nn.Linear(graph_dim, d_model), nn.LayerNorm(d_model))
            self.output_dim = d_model * 3
        else:
            self.output_dim = d_model

    def forward(
        self,
        omics_vector: Tensor,
        node_embeddings: Tensor,
        batch_index: Tensor,
        graph_embedding: Optional[Tensor] = None,
        return_attention: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        omics_cls = build_omics_cls(omics_vector, self.omics_proj)
        omics_cls = omics_cls + self.token_type.weight[0].view(1, 1, -1)

        atom_dense, atom_valid = to_dense_batch(node_embeddings, batch=batch_index)
        atom_tokens = self.atom_proj(atom_dense) + self.token_type.weight[1].view(1, 1, -1)
        key_padding_mask = ~atom_valid

        if return_attention:
            updated, attn = self.cross_attn(
                omics_cls, atom_tokens, key_padding_mask=key_padding_mask, return_attention=True
            )
        else:
            updated = self.cross_attn(
                omics_cls, atom_tokens, key_padding_mask=key_padding_mask, return_attention=False
            )
            attn = None

        updated_vec = updated.squeeze(1)
        if self.residual_mode == "pure":
            repr_vec = updated_vec
        else:
            if graph_embedding is None:
                raise ValueError("pooled_residual requires graph_embedding")
            repr_vec = torch.cat(
                [updated_vec, self.graph_proj(graph_embedding), omics_cls.squeeze(1)], dim=-1
            )

        if return_attention:
            return repr_vec, attn
        return repr_vec


def build_fusion_model(
    architecture_family: str,
    *,
    omics_dim: int,
    graph_dim: int = 32,
    node_dim: int = 32,
    residual_mode: str = "pure",
    transformer_cfg: Optional[Dict] = None,
    cross_attn_cfg: Optional[Dict] = None,
) -> nn.Module:
    family = architecture_family.lower()
    if family in {"pooled_mlp", "mlp", "a0", "a1", "a2"}:
        return PooledMLPFusion(omics_dim=omics_dim, graph_dim=graph_dim)
    if family in {"pooled_transformer", "transformer", "a3", "a4", "a5"}:
        cfg = transformer_cfg or {}
        return PooledTransformerFusion(
            omics_dim=omics_dim,
            graph_dim=graph_dim,
            d_model=int(cfg.get("d_model", 128)),
            n_heads=int(cfg.get("n_heads", 4)),
            num_layers=int(cfg.get("num_layers", 1)),
            dim_feedforward=int(cfg.get("dim_feedforward", cfg.get("d_ff", 128))),
            dropout=float(cfg.get("dropout", 0.1)),
            attn_dropout=float(cfg.get("attn_dropout", 0.1)),
            temperature=float(cfg.get("temperature", 1.0)),
            attn_out_mlp=bool(cfg.get("attn_out_mlp", True)),
            attn_out_activation=str(cfg.get("attn_out_activation", "GELU")),
            ffn_activation=str(cfg.get("ffn_activation", "ReLU")),
            requested_use_mask=bool(cfg.get("use_mask", True)),
        )
    if family in {"cross_attention", "atom_cross_attention", "c0", "c1"}:
        cfg = cross_attn_cfg or {}
        mode = residual_mode
        if family == "c1":
            mode = "pooled_residual"
        if family == "c0":
            mode = "pure"
        return AtomCrossAttentionFusion(
            omics_dim=omics_dim,
            node_dim=node_dim,
            graph_dim=graph_dim,
            d_model=int(cfg.get("d_model", 128)),
            n_heads=int(cfg.get("n_heads", 4)),
            num_layers=int(cfg.get("num_layers", 1)),
            dim_feedforward=int(cfg.get("dim_feedforward", cfg.get("d_ff", 256))),
            dropout=float(cfg.get("dropout", 0.1)),
            attn_dropout=float(cfg.get("attn_dropout", 0.1)),
            temperature=float(cfg.get("temperature", 1.0)),
            residual_mode=mode,
        )
    raise ValueError(f"Unknown architecture_family: {architecture_family}")


def build_fusion_and_head(
    architecture_family: str,
    *,
    omics_dim: int,
    graph_dim: int = 32,
    node_dim: int = 32,
    residual_mode: str = "pure",
    transformer_cfg: Optional[Dict] = None,
    cross_attn_cfg: Optional[Dict] = None,
    head_hidden: int = 128,
    head_dropout: float = 0.1,
) -> Tuple[nn.Module, Round18ResponseHead]:
    fusion = build_fusion_model(
        architecture_family,
        omics_dim=omics_dim,
        graph_dim=graph_dim,
        node_dim=node_dim,
        residual_mode=residual_mode,
        transformer_cfg=transformer_cfg,
        cross_attn_cfg=cross_attn_cfg,
    )
    head = Round18ResponseHead(fusion.output_dim, hidden_dim=head_hidden, dropout=head_dropout)
    return fusion, head
