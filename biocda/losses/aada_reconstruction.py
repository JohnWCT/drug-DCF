"""AADA reconstruction-margin losses (Round 25 S1).

Field name: reconstruction_margin (NOT prototype_upper_margin).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from biocda.losses.smooth_l1_vector import vector_smooth_l1


@dataclass
class AADADiscriminatorOutput:
    loss: torch.Tensor
    source_reconstruction_error: torch.Tensor
    target_reconstruction_error: torch.Tensor
    reconstruction_hinge_active_fraction: torch.Tensor


@dataclass
class AADATargetAdapterOutput:
    loss: torch.Tensor
    target_reconstruction_error: torch.Tensor


def aada_ae_discriminator_loss(
    latent_ae,
    source_latent: torch.Tensor,
    adapted_target_latent: torch.Tensor,
    *,
    reconstruction_margin: float,
    alpha_margin: float,
    beta: float = 1.0,
) -> AADADiscriminatorOutput:
    """Update latent AE only: L_rec(source) + alpha * hinge(margin - L_rec(target))."""
    source_fixed = source_latent.detach()
    target_fixed = adapted_target_latent.detach()

    source_rec = vector_smooth_l1(latent_ae(source_fixed), source_fixed, beta=beta)
    target_rec = vector_smooth_l1(latent_ae(target_fixed), target_fixed, beta=beta)

    hinge = torch.relu(float(reconstruction_margin) - target_rec)
    loss = source_rec.mean() + float(alpha_margin) * hinge.mean()
    return AADADiscriminatorOutput(
        loss=loss,
        source_reconstruction_error=source_rec.mean().detach(),
        target_reconstruction_error=target_rec.mean().detach(),
        reconstruction_hinge_active_fraction=(hinge > 0).float().mean().detach(),
    )


def aada_target_adapter_loss(
    latent_ae,
    adapted_target_latent: torch.Tensor,
    *,
    beta: float = 1.0,
) -> AADATargetAdapterOutput:
    """Adapter update with AE frozen: minimize L_rec(AE(adapted), adapted)."""
    for p in latent_ae.parameters():
        p.requires_grad_(False)
    try:
        target_rec = vector_smooth_l1(
            latent_ae(adapted_target_latent),
            adapted_target_latent,
            beta=beta,
        )
        loss = target_rec.mean()
        return AADATargetAdapterOutput(
            loss=loss,
            target_reconstruction_error=target_rec.mean().detach(),
        )
    finally:
        for p in latent_ae.parameters():
            p.requires_grad_(True)
