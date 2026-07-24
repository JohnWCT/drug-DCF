"""AADA training step helpers for Round 25 S1."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch import nn

from biocda.losses.aada_reconstruction import (
    aada_ae_discriminator_loss,
    aada_target_adapter_loss,
)
from biocda.stage2.latent_autoencoder import LatentAutoencoder
from biocda.stage2.target_adapter import TargetResidualAdapter


def build_aada_components(
    latent_dim: int,
    *,
    device,
    lr: float,
    gan_epoch: int,
) -> Dict:
    ae = LatentAutoencoder(latent_dim).to(device)
    adapter = TargetResidualAdapter(latent_dim).to(device)
    ae_opt = torch.optim.RMSprop(ae.parameters(), lr=lr)
    adapter_opt = torch.optim.RMSprop(adapter.parameters(), lr=lr)
    ae_sched = torch.optim.lr_scheduler.CosineAnnealingLR(ae_opt, max(1, gan_epoch))
    adapter_sched = torch.optim.lr_scheduler.CosineAnnealingLR(adapter_opt, max(1, gan_epoch))
    return {
        "latent_ae": ae,
        "target_adapter": adapter,
        "ae_optimizer": ae_opt,
        "adapter_optimizer": adapter_opt,
        "ae_scheduler": ae_sched,
        "adapter_scheduler": adapter_sched,
    }


def apply_target_adapter(
    target_z: torch.Tensor,
    adapter: Optional[nn.Module],
    *,
    enabled: bool,
) -> torch.Tensor:
    if not enabled or adapter is None:
        return target_z
    return adapter(target_z)


def aada_update_step(
    *,
    latent_ae: nn.Module,
    target_adapter: nn.Module,
    ae_optimizer,
    adapter_optimizer,
    source_z: torch.Tensor,
    target_z_base: torch.Tensor,
    reconstruction_margin: float,
    alpha_margin: float,
    lambda_aada: float,
    beta: float = 1.0,
) -> Tuple[torch.Tensor, dict]:
    """One AADA pair: AE discriminator update, then target-adapter update."""
    # Source path must remain frozen — callers pass detached encoder outs for AE update.
    adapted = target_adapter(target_z_base)

    ae_optimizer.zero_grad(set_to_none=True)
    ae_out = aada_ae_discriminator_loss(
        latent_ae,
        source_z.detach(),
        adapted.detach(),
        reconstruction_margin=reconstruction_margin,
        alpha_margin=alpha_margin,
        beta=beta,
    )
    ae_out.loss.backward()
    ae_optimizer.step()

    adapter_optimizer.zero_grad(set_to_none=True)
    # Recompute adapted with grad for adapter; AE grads disabled inside helper.
    adapted2 = target_adapter(target_z_base)
    ad_out = aada_target_adapter_loss(latent_ae, adapted2, beta=beta)
    (float(lambda_aada) * ad_out.loss).backward()
    adapter_optimizer.step()

    metrics = {
        "loss_aada": float(ad_out.loss.detach().item()),
        "source_reconstruction_error": float(ae_out.source_reconstruction_error.item()),
        "target_reconstruction_error": float(ae_out.target_reconstruction_error.item()),
        "reconstruction_hinge_active_fraction": float(
            ae_out.reconstruction_hinge_active_fraction.item()
        ),
    }
    return ad_out.loss.detach(), metrics
