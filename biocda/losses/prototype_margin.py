"""Prototype margin-gated alignment loss (Round 25 S2).

Field name: prototype_upper_margin (NOT reconstruction_margin).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class PrototypeMarginOutput:
    loss: torch.Tensor
    distances: torch.Tensor
    margins: torch.Tensor
    active_mask: torch.Tensor
    active_fraction: torch.Tensor
    mean_distance: torch.Tensor
    mean_margin: torch.Tensor


def cosine_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    return 1.0 - (a * b).sum(dim=-1)


def margin_gated_prototype_loss(
    target_centroids: torch.Tensor,
    source_anchors: torch.Tensor,
    margins: torch.Tensor,
) -> PrototypeMarginOutput:
    """L = mean_c max(0, d(mu_t,c, p_s,c) - delta_c).

    - source_anchors must already be detached by caller (enforced here too).
    - target_centroids retain gradients.
    - empty cancer set must fail (caller must not pass empty tensors).
    """
    if target_centroids.numel() == 0 or source_anchors.numel() == 0:
        raise ValueError("empty cancer set is not allowed for prototype margin loss")
    if target_centroids.shape != source_anchors.shape:
        raise ValueError("target_centroids and source_anchors must have the same shape")
    if margins.shape != target_centroids.shape[:1]:
        raise ValueError("margins must have shape [num_valid_cancers]")

    distances = cosine_distance(target_centroids, source_anchors.detach())
    penalties = torch.relu(distances - margins)
    active_mask = distances > margins

    return PrototypeMarginOutput(
        loss=penalties.mean(),
        distances=distances.detach(),
        margins=margins.detach(),
        active_mask=active_mask.detach(),
        active_fraction=active_mask.float().mean(),
        mean_distance=distances.mean().detach(),
        mean_margin=margins.mean().detach(),
    )
