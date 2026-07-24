"""Smooth-L1 vector reconstruction helper for AADA (Round 25 S1).

Margin scale is per-sample mean over latent dims — never sum(dim=-1).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def vector_smooth_l1(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    *,
    beta: float = 1.0,
) -> torch.Tensor:
    loss = F.smooth_l1_loss(
        reconstruction,
        target,
        beta=beta,
        reduction="none",
    )
    return loss.mean(dim=-1)
