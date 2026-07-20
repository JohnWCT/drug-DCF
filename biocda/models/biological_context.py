"""Sample representation: concat omics latent Z and biological context C."""
from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn


class SampleRepresentation(nn.Module):
    """Build cross-attention query from [Z ; C]."""

    def __init__(
        self,
        omics_dim: int,
        context_dim: int,
        output_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        input_dim = int(omics_dim) + int(context_dim)
        if output_dim is None:
            self.projection = nn.Identity()
            self.output_dim = input_dim
        else:
            self.projection = nn.Sequential(
                nn.Linear(input_dim, int(output_dim)),
                nn.LayerNorm(int(output_dim)),
                nn.GELU(),
            )
            self.output_dim = int(output_dim)

    def forward(self, omics_latent: Tensor, biological_context: Tensor) -> Tensor:
        combined = torch.cat([omics_latent, biological_context], dim=-1)
        return self.projection(combined)
