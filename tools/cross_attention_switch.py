"""Cross-attention module for Round 18 (omics CLS → atom tokens).

Does not modify tools/transformer_switch.py interfaces.
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class _FeedForward(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = "ReLU",
        layer_norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.ff1 = nn.Linear(d_model, d_ff)
        self.ff2 = nn.Linear(d_ff, d_model)
        act = activation.upper()
        if act == "GELU":
            self.act = nn.GELU()
        elif act == "SILU":
            self.act = nn.SiLU()
        else:
            self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model, eps=layer_norm_eps)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.ff2(self.dropout(self.act(self.ff1(x))))
        return self.norm(residual + self.dropout(x))


class CrossAttentionLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        attn_dropout: float = 0.1,
        temperature: float = 1.0,
        attn_out_mlp: bool = True,
        attn_out_activation: str = "GELU",
        ffn_activation: str = "ReLU",
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.temperature = float(temperature)

        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.fc = nn.Linear(d_model, d_model, bias=False)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.out_dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.use_out_mlp = attn_out_mlp
        if attn_out_mlp:
            act = attn_out_activation.lower()
            if act == "silu":
                act_layer: nn.Module = nn.SiLU()
            elif act == "relu":
                act_layer = nn.ReLU()
            else:
                act_layer = nn.GELU()
            self.out_mlp = nn.Sequential(
                nn.Linear(d_model, d_model),
                act_layer,
                nn.Dropout(dropout),
                nn.Linear(d_model, d_model),
            )
        else:
            self.out_mlp = None

        self.ffn = _FeedForward(
            d_model=d_model,
            d_ff=dim_feedforward,
            dropout=dropout,
            activation=ffn_activation,
        )

    def forward(
        self,
        query_tokens: Tensor,
        key_value_tokens: Tensor,
        key_padding_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        query_tokens: [B, Lq, D] (Round 18: Lq=1)
        key_value_tokens: [B, Lk, D]
        key_padding_mask: [B, Lk] True = padding (masked out)
        returns: updated_query [B, Lq, D], attn [B, H, Lq, Lk]
        """
        residual = query_tokens
        B, Lq, D = query_tokens.shape
        Lk = key_value_tokens.size(1)

        Q = self.W_Q(query_tokens).view(B, Lq, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_K(key_value_tokens).view(B, Lk, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_V(key_value_tokens).view(B, Lk, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1))
        scores = scores / (math.sqrt(self.d_k) * max(self.temperature, 1e-8))

        if key_padding_mask is not None:
            # [B, Lk] -> [B, 1, 1, Lk]
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask.bool(), float("-inf"))

        attn = F.softmax(scores, dim=-1)
        # Replace NaNs from all-masked rows (should not happen for valid queries)
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.attn_dropout(attn)

        ctx = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, Lq, D)
        out = self.fc(ctx)
        out = self.out_dropout(out)
        if self.use_out_mlp and self.out_mlp is not None:
            out = self.out_mlp(out)
        out = self.norm1(residual + out)
        out = self.ffn(out)
        return out, attn


class CrossAttentionSwitch(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        attn_dropout: float = 0.1,
        temperature: float = 1.0,
        attn_out_mlp: bool = True,
        attn_out_activation: str = "GELU",
        ffn_activation: str = "ReLU",
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                CrossAttentionLayer(
                    d_model=d_model,
                    n_heads=n_heads,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    attn_dropout=attn_dropout,
                    temperature=temperature,
                    attn_out_mlp=attn_out_mlp,
                    attn_out_activation=attn_out_activation,
                    ffn_activation=ffn_activation,
                )
                for _ in range(num_layers)
            ]
        )
        self.num_layers = num_layers
        self.n_heads = n_heads

    def forward(
        self,
        query_tokens: Tensor,
        key_value_tokens: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        return_attention: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        attn_all: List[Tensor] = []
        x = query_tokens
        for layer in self.layers:
            x, attn = layer(x, key_value_tokens, key_padding_mask=key_padding_mask)
            if return_attention:
                attn_all.append(attn)
        if return_attention:
            # [num_layers, B, H, Lq, Lk]
            return x, torch.stack(attn_all, dim=0)
        return x
