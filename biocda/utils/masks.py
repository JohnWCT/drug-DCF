"""Attention mask utilities."""
from __future__ import annotations

import torch


def zero_padding_attention(attention: torch.Tensor, atom_mask: torch.Tensor) -> torch.Tensor:
    """Zero out padded atom positions in [B, H, N] attention."""
    return attention * atom_mask.unsqueeze(1).to(attention.dtype)
