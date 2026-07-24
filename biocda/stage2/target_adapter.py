"""Zero-initialized residual adapter for target latents (Round 25 AADA)."""

from __future__ import annotations

import torch
from torch import nn


class TargetResidualAdapter(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, target_latent: torch.Tensor) -> torch.Tensor:
        return target_latent + self.net(target_latent)
