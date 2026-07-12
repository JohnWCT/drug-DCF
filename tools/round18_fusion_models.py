"""Round 18 fusion architectures: pooled MLP / Transformer / cross-attention."""
from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import torch
from torch import Tensor, nn
from torch_geometric.utils import to_dense_batch

from tools.cross_attention_switch import CrossAttentionSwitch
from tools.round18_response_head import Round18ResponseHead
from tools.transformer_switch import TransformerSwitch


def build_omics_cls(omics_vector: Tensor, projector: nn.Module) -> Tensor:
    """omics_vector [B, D_in] -> omics_cls [B, 1, d_model]."""
    return projector(omics_vector).unsqueeze(1)


class OmicsProjector(nn.Module):
    def __init__(self, input_dim: int, d_model: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(x)


class PooledMLPFusion(nn.Module):
    """A0–A2: concat(omics, pooled GIN graph embedding) -> fixed head."""

    def __init__(
        self,
        omics_dim: int,
        graph_dim: int,
        head_hidden: int = 128,
        head_dropout: float = 0.1,
    ):
        super().__init__()
        self.head = Round18ResponseHead(
            input_dim=omics_dim + graph_dim,
            hidden_dim=head_hidden,
            dropout=head_dropout,
        )

    def forward(self, omics_vector: Tensor, graph_embedding: Tensor) -> Tensor:
        return self.head(torch.cat([omics_vector, graph_embedding], dim=-1))


class PooledTransformerFusion(nn.Module):
    """A3–A5: 2-token TransformerSwitch (omics + pooled drug), take token0."""

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
        use_mask: bool = True,
        head_hidden: int = 128,
        head_dropout: float = 0.1,
    ):
        super().__init__()
        self.omics_proj = OmicsProjector(omics_dim, d_model)
        self.drug_proj = nn.Sequential(nn.Linear(graph_dim, d_model), nn.LayerNorm(d_model))
        self.token_type = nn.Embedding(2, d_model)
        self.encoder = TransformerSwitch(
            d_model=d_model,
            n_heads=n_heads,
            num_encoder_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            attn_dropout=attn_dropout,
            temperature=temperature,
            use_mask=False,  # Round 18: fixed 2 tokens, disable auto-pad / mask fill
            use_positional_encoding=False,
            attn_out_mlp=attn_out_mlp,
        )
        self.head = Round18ResponseHead(d_model, hidden_dim=head_hidden, dropout=head_dropout)
        self.use_mask = False

    def forward(self, omics_vector: Tensor, graph_embedding: Tensor) -> Tensor:
        omics_tok = self.omics_proj(omics_vector)
        drug_tok = self.drug_proj(graph_embedding)
        tokens = torch.stack([omics_tok, drug_tok], dim=1)  # [B, 2, D]
        type_ids = torch.tensor([0, 1], device=tokens.device).unsqueeze(0).expand(tokens.size(0), -1)
        tokens = tokens + self.token_type(type_ids)

        # Two dense tokens: no padding. Avoid TransformerSwitch masked_fill(-1e9)
        # which overflows under AMP float16 even when the mask is all-False.
        encoded, _ = self.encoder(tokens, attn_mask=None)
        updated_cls = encoded[:, 0, :]
        return self.head(updated_cls)


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
        residual_mode: str = "pure",  # pure | pooled_residual
        head_hidden: int = 128,
        head_dropout: float = 0.1,
    ):
        super().__init__()
        assert residual_mode in {"pure", "pooled_residual"}
        self.residual_mode = residual_mode
        self.omics_proj = OmicsProjector(omics_dim, d_model)
        self.atom_proj = nn.Sequential(nn.Linear(node_dim, d_model), nn.LayerNorm(d_model))
        self.token_type = nn.Embedding(2, d_model)  # 0=omics, 1=atom
        self.cross_attn = CrossAttentionSwitch(
            d_model=d_model,
            n_heads=n_heads,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            attn_dropout=attn_dropout,
            temperature=temperature,
        )
        if residual_mode == "pure":
            head_in = d_model
        else:
            self.graph_proj = nn.Sequential(nn.Linear(graph_dim, d_model), nn.LayerNorm(d_model))
            head_in = d_model * 3  # updated CLS + pooled GIN + original projected omics
        self.head = Round18ResponseHead(head_in, hidden_dim=head_hidden, dropout=head_dropout)

    def forward(
        self,
        omics_vector: Tensor,
        node_embeddings: Tensor,
        batch_index: Tensor,
        graph_embedding: Optional[Tensor] = None,
        return_attention: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        omics_cls = build_omics_cls(omics_vector, self.omics_proj)  # [B,1,D]
        omics_cls = omics_cls + self.token_type.weight[0].view(1, 1, -1)

        atom_dense, atom_valid = to_dense_batch(node_embeddings, batch=batch_index)
        atom_tokens = self.atom_proj(atom_dense)
        atom_tokens = atom_tokens + self.token_type.weight[1].view(1, 1, -1)
        key_padding_mask = ~atom_valid  # True = padding

        if return_attention:
            updated, attn = self.cross_attn(
                omics_cls,
                atom_tokens,
                key_padding_mask=key_padding_mask,
                return_attention=True,
            )
        else:
            updated = self.cross_attn(
                omics_cls,
                atom_tokens,
                key_padding_mask=key_padding_mask,
                return_attention=False,
            )
            attn = None

        updated_vec = updated.squeeze(1)
        if self.residual_mode == "pure":
            logits = self.head(updated_vec)
        else:
            if graph_embedding is None:
                raise ValueError("pooled_residual requires graph_embedding")
            graph_vec = self.graph_proj(graph_embedding)
            omics_vec = omics_cls.squeeze(1)
            logits = self.head(torch.cat([updated_vec, graph_vec, omics_vec], dim=-1))

        if return_attention:
            return logits, attn
        return logits


def build_fusion_model(
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
) -> nn.Module:
    family = architecture_family.lower()
    if family in {"pooled_mlp", "mlp", "a0", "a1", "a2"}:
        return PooledMLPFusion(
            omics_dim=omics_dim,
            graph_dim=graph_dim,
            head_hidden=head_hidden,
            head_dropout=head_dropout,
        )
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
            use_mask=bool(cfg.get("use_mask", True)),
            head_hidden=head_hidden,
            head_dropout=head_dropout,
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
            head_hidden=head_hidden,
            head_dropout=head_dropout,
        )
    raise ValueError(f"Unknown architecture_family: {architecture_family}")
