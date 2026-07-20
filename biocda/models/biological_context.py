"""Sample representation variants for M1 (Z) and M2 (Z+C)."""
from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn


class SampleRepresentationZC(nn.Module):
    """M2 BioCDA-XA-ZC: LayerNorm(Z) + LayerNorm(C) → concat → projection."""

    def __init__(
        self,
        omics_dim: int,
        context_dim: int,
        output_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.omics_dim = int(omics_dim)
        self.context_dim = int(context_dim)
        self.omics_norm = nn.LayerNorm(self.omics_dim)
        self.context_norm = nn.LayerNorm(self.context_dim)
        input_dim = self.omics_dim + self.context_dim
        out_dim = int(output_dim) if output_dim is not None else input_dim
        self.projection = nn.Sequential(
            nn.Linear(input_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )
        self.output_dim = out_dim
        self.uses_context = True

    def forward(self, omics_latent: Tensor, biological_context: Tensor) -> Tensor:
        z = self.omics_norm(omics_latent)
        c = self.context_norm(biological_context)
        return self.projection(torch.cat([z, c], dim=-1))


class SampleRepresentationZ(nn.Module):
    """M1 BioCDA-XA-Z: query from Z only (context not in query path)."""

    def __init__(self, omics_dim: int, output_dim: Optional[int] = None) -> None:
        super().__init__()
        self.omics_dim = int(omics_dim)
        self.omics_norm = nn.LayerNorm(self.omics_dim)
        out_dim = int(output_dim) if output_dim is not None else self.omics_dim
        self.projection = nn.Sequential(
            nn.Linear(self.omics_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )
        self.output_dim = out_dim
        self.uses_context = False

    def forward(self, omics_latent: Tensor, biological_context: Tensor) -> Tensor:
        del biological_context
        return self.projection(self.omics_norm(omics_latent))


# Backward-compatible alias
SampleRepresentation = SampleRepresentationZC
