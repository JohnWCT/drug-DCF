"""Frozen omics latent encoder (precomputed Z or passthrough)."""
from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn


class FrozenOmicsEncoder(nn.Module):
    """Pass-through or linear projection for frozen omics latent Z."""

    def __init__(
        self,
        *,
        latent_dim: int,
        projection: Optional[nn.Module] = None,
        frozen: bool = True,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.projection = projection or nn.Identity()
        self.frozen = bool(frozen)
        if self.frozen:
            for p in self.parameters():
                p.requires_grad = False

    def forward(self, omics: Tensor) -> Tensor:
        if omics.shape[-1] != self.latent_dim:
            raise ValueError(
                f"Expected omics latent dim {self.latent_dim}, got {omics.shape[-1]}"
            )
        return self.projection(omics)
