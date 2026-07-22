"""Sample query projector: Z64 + C32 → Q0 [B,1,128]."""
from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor, nn


class SampleQueryProjector(nn.Module):
    """Separate LayerNorm on Z/C, concat to 96-d, project to d_model query."""

    def __init__(self, omics_dim: int = 64, context_dim: int = 32, d_model: int = 128) -> None:
        super().__init__()
        self.omics_dim = int(omics_dim)
        self.context_dim = int(context_dim)
        self.d_model = int(d_model)
        self.omics_norm = nn.LayerNorm(self.omics_dim)
        self.context_norm = nn.LayerNorm(self.context_dim)
        self.projection = nn.Sequential(
            nn.Linear(self.omics_dim + self.context_dim, self.d_model),
            nn.LayerNorm(self.d_model),
        )

    def forward(self, z64: Tensor, c32: Tensor) -> Tuple[Tensor, Tensor]:
        if z64.shape[-1] != self.omics_dim:
            raise ValueError(f"Expected Z dim {self.omics_dim}, got {z64.shape[-1]}")
        if c32.shape[-1] != self.context_dim:
            raise ValueError(f"Expected C dim {self.context_dim}, got {c32.shape[-1]}")
        z = self.omics_norm(z64)
        c = self.context_norm(c32)
        sample_features = torch.cat([z, c], dim=-1)
        if sample_features.shape[-1] != 96:
            raise AssertionError(f"sample_features must be 96-d, got {sample_features.shape[-1]}")
        query = self.projection(sample_features).unsqueeze(1)  # [B,1,d_model]
        return sample_features, query


class SampleQueryProjectorZOnly(nn.Module):
    """Ablation X3: Z64 only padded/projected without context."""

    def __init__(self, omics_dim: int = 64, d_model: int = 128) -> None:
        super().__init__()
        self.omics_dim = int(omics_dim)
        self.d_model = int(d_model)
        self.omics_norm = nn.LayerNorm(self.omics_dim)
        self.projection = nn.Sequential(
            nn.Linear(self.omics_dim, self.d_model),
            nn.LayerNorm(self.d_model),
        )

    def forward(self, z64: Tensor, c32: Tensor) -> Tuple[Tensor, Tensor]:
        del c32
        if z64.shape[-1] != self.omics_dim:
            raise ValueError(f"Expected Z dim {self.omics_dim}, got {z64.shape[-1]}")
        z = self.omics_norm(z64)
        query = self.projection(z).unsqueeze(1)
        return z, query
