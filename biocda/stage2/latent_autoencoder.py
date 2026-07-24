"""Latent autoencoder used as AADA domain discriminator (Round 25 S1)."""

from __future__ import annotations

from torch import nn


class LatentAutoencoder(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        hidden_dim = latent_dim * 2
        bottleneck_dim = max(latent_dim // 2, 8)
        self.encoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, latent):
        return self.decoder(self.encoder(latent))
