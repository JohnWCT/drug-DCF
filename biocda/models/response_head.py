"""Response predictor head — sample + attended drug representation only."""
from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor, nn


class BioCDAResponseHead(nn.Module):
    """Fuse sample representation with patient-conditioned drug representation."""

    def __init__(
        self,
        sample_dim: int,
        drug_dim: int,
        hidden_dims: Sequence[int] = (256, 128),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        input_dim = int(sample_dim) + int(drug_dim)
        layers: list[nn.Module] = []
        prev = input_dim
        for hidden in hidden_dims:
            layers.extend([nn.Linear(prev, int(hidden)), nn.ReLU(), nn.Dropout(dropout)])
            prev = int(hidden)
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
        self.input_dim = input_dim

    def forward(self, sample_repr: Tensor, drug_repr: Tensor) -> Tensor:
        fusion = torch.cat([sample_repr, drug_repr], dim=-1)
        return self.net(fusion).squeeze(-1)
