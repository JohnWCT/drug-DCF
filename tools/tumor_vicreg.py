"""VICReg-style tumor latent anti-collapse regularization (Round 6E)."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_vicreg_var_cov_loss(
    z,
    var_target=1.0,
    eps=1e-4,
):
    """
    Variance + covariance regularization on latent batch.

    Returns:
        var_loss, cov_loss, metrics dict
    """
    if z.size(0) < 2:
        zero = z.sum() * 0.0
        return zero, zero, {
            "tumor_vicreg_var_loss": 0.0,
            "tumor_vicreg_cov_loss": 0.0,
            "tumor_vicreg_mean_std": 0.0,
            "tumor_vicreg_min_std": 0.0,
            "tumor_vicreg_cov_offdiag_mean_abs": 0.0,
            "tumor_vicreg_valid": False,
        }

    z = z - z.mean(dim=0, keepdim=True)
    std = torch.sqrt(z.var(dim=0, unbiased=False) + eps)
    var_loss = torch.mean(F.relu(float(var_target) - std))

    n, d = z.shape
    cov = (z.t() @ z) / max(1, n - 1)
    off_diag = cov - torch.diag(torch.diag(cov))
    cov_loss = (off_diag.pow(2).sum()) / d

    metrics = {
        "tumor_vicreg_var_loss": float(var_loss.detach().item()),
        "tumor_vicreg_cov_loss": float(cov_loss.detach().item()),
        "tumor_vicreg_mean_std": float(std.mean().detach().item()),
        "tumor_vicreg_min_std": float(std.min().detach().item()),
        "tumor_vicreg_cov_offdiag_mean_abs": float(off_diag.abs().mean().detach().item()),
        "tumor_vicreg_valid": True,
    }
    return var_loss, cov_loss, metrics
