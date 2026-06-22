"""Reconstruction loss helpers for VAE / AE pretraining."""

from __future__ import annotations

import torch
import torch.nn.functional as F

SUPPORTED_RECONSTRUCTION_LOSS_TYPES = frozenset(
    {"mse", "smooth_l1", "hybrid_mse_smooth_l1"}
)


def compute_reconstruction_loss(
    recon_x: torch.Tensor,
    x: torch.Tensor,
    loss_type: str = "mse",
    smooth_l1_beta: float = 1.0,
    reduction: str = "mean",
    hybrid_alpha: float = 0.5,
) -> torch.Tensor:
    loss_type = str(loss_type).lower()

    if loss_type == "mse":
        return F.mse_loss(recon_x, x, reduction=reduction)

    if loss_type == "smooth_l1":
        return F.smooth_l1_loss(
            recon_x,
            x,
            beta=float(smooth_l1_beta),
            reduction=reduction,
        )

    if loss_type == "hybrid_mse_smooth_l1":
        mse = F.mse_loss(recon_x, x, reduction=reduction)
        smooth = F.smooth_l1_loss(
            recon_x,
            x,
            beta=float(smooth_l1_beta),
            reduction=reduction,
        )
        return float(hybrid_alpha) * mse + (1.0 - float(hybrid_alpha)) * smooth

    raise ValueError(f"Unsupported reconstruction_loss_type: {loss_type}")


def reconstruction_loss_defaults() -> dict:
    return {
        "reconstruction_loss_type": "mse",
        "smooth_l1_beta": 1.0,
        "reconstruction_loss_reduction": "mean",
        "reconstruction_loss_scale": 1.0,
        "hybrid_reconstruction_alpha": 0.5,
    }


def reconstruction_loss_kwargs(params: dict) -> dict:
    defaults = reconstruction_loss_defaults()
    return {
        "reconstruction_loss_type": params.get(
            "reconstruction_loss_type", defaults["reconstruction_loss_type"]
        ),
        "smooth_l1_beta": float(params.get("smooth_l1_beta", defaults["smooth_l1_beta"])),
        "reconstruction_loss_reduction": params.get(
            "reconstruction_loss_reduction", defaults["reconstruction_loss_reduction"]
        ),
        "reconstruction_loss_scale": float(
            params.get("reconstruction_loss_scale", defaults["reconstruction_loss_scale"])
        ),
        "hybrid_reconstruction_alpha": float(
            params.get("hybrid_reconstruction_alpha", defaults["hybrid_reconstruction_alpha"])
        ),
    }
