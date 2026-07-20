"""Numerically stable masked softmax for atom attention."""
from __future__ import annotations

import torch
from torch import Tensor


def masked_softmax(logits: Tensor, mask: Tensor, *, dim: int = -1) -> Tensor:
    """Softmax with padding forced to zero; valid positions renormalized.

    Raises ValueError if any batch row has zero valid atoms.
    """
    if logits.shape != mask.shape:
        raise ValueError(f"logits shape {logits.shape} != mask shape {mask.shape}")
    if mask.dtype != torch.bool:
        mask = mask.bool()
    valid_counts = mask.sum(dim=dim)
    if (valid_counts == 0).any():
        raise ValueError("masked_softmax: graph with zero valid atoms")

    masked_logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
    probs = torch.softmax(masked_logits, dim=dim)
    probs = probs * mask.to(probs.dtype)
    denom = probs.sum(dim=dim, keepdim=True).clamp_min(1e-12)
    probs = probs / denom
    return probs
