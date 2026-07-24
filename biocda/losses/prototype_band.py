"""Prototype distance-band loss (Round 25 conditional S2b).

Fields: prototype_lower_margin / prototype_upper_margin.
Only enabled when over-overlap evidence gates pass.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class PrototypeBandOutput:
    loss: torch.Tensor
    upper_loss: torch.Tensor
    lower_loss: torch.Tensor
    below_fraction: torch.Tensor
    within_fraction: torch.Tensor
    above_fraction: torch.Tensor


def prototype_distance_band_loss(
    distances: torch.Tensor,
    lower_margin: torch.Tensor,
    upper_margin: torch.Tensor,
    *,
    lower_weight: float = 0.1,
) -> PrototypeBandOutput:
    if distances.numel() == 0:
        raise ValueError("empty distance set is not allowed for band loss")
    if not 0.0 <= lower_weight < 1.0:
        raise ValueError("lower_weight must be in [0, 1)")
    if torch.any(lower_margin >= upper_margin):
        raise ValueError("lower_margin must be smaller than upper_margin")

    upper_penalty = torch.relu(distances - upper_margin)
    lower_penalty = torch.relu(lower_margin - distances)

    below = distances < lower_margin
    above = distances > upper_margin
    within = ~(below | above)

    total = upper_penalty + lower_weight * lower_penalty
    return PrototypeBandOutput(
        loss=total.mean(),
        upper_loss=upper_penalty.mean(),
        lower_loss=lower_penalty.mean(),
        below_fraction=below.float().mean(),
        within_fraction=within.float().mean(),
        above_fraction=above.float().mean(),
    )
