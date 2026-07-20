"""Round 20 omics adapter: frozen latent vs raw-omics end-to-end-capable path."""
from __future__ import annotations

import torch
from torch import Tensor, nn


class Round20OmicsAdapter(nn.Module):
    def __init__(
        self,
        *,
        mode: str,
        encoder: nn.Module | None,
        context_dim: int,
    ) -> None:
        super().__init__()
        if mode not in {"frozen_latent", "raw_omics"}:
            raise ValueError(f"Unsupported mode: {mode}")
        if mode == "raw_omics" and encoder is None:
            raise ValueError("raw_omics mode requires encoder")
        self.mode = mode
        self.encoder = encoder
        self.context_dim = int(context_dim)

    def forward(
        self,
        *,
        context: Tensor,
        precomputed_z: Tensor | None = None,
        raw_omics: Tensor | None = None,
    ) -> Tensor:
        if context.shape[-1] != self.context_dim:
            raise ValueError(
                f"Invalid context dimension: expected {self.context_dim}, got {context.shape[-1]}"
            )
        if self.mode == "frozen_latent":
            if precomputed_z is None:
                raise ValueError("frozen_latent mode requires precomputed_z")
            z = precomputed_z
        else:
            if raw_omics is None:
                raise ValueError("raw_omics mode requires raw_omics")
            z = self.encoder(raw_omics)
        if z.shape[-1] != 64:
            raise ValueError(f"Expected Z64, received {z.shape[-1]}")
        return torch.cat([z, context], dim=-1)
