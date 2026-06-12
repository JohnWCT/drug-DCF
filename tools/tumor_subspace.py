"""Tumor / transfer latent subspace split (Round 6C)."""

from __future__ import annotations

import torch


def split_tumor_transfer_latent(z, tumor_dim: int):
    """Split shared latent into tumor and transfer subspaces."""
    tumor_dim = int(tumor_dim)
    if tumor_dim <= 0 or tumor_dim >= z.size(-1):
        return z, z.new_zeros(z.size(0), 0)
    return z[:, :tumor_dim], z[:, tumor_dim:]


def resolve_subspace_training_params(param: dict) -> dict:
    latent_size = int(param.get("latent_size", 32))
    use_tumor_subspace = bool(param.get("use_tumor_subspace", False))
    tumor_dim = int(param.get("tumor_dim", min(16, latent_size // 2)))
    if use_tumor_subspace:
        tumor_dim = max(1, min(tumor_dim, latent_size - 1))
    else:
        tumor_dim = latent_size
    transfer_dim = latent_size - tumor_dim if use_tumor_subspace else latent_size
    return {
        "use_tumor_subspace": use_tumor_subspace,
        "tumor_dim": tumor_dim,
        "transfer_dim": transfer_dim,
        "latent_size": latent_size,
        "classifier_latent_view": str(param.get("classifier_latent_view", "shared")),
        "alignment_latent_view": str(param.get("alignment_latent_view", "shared")),
        "topology_latent_view": str(param.get("topology_latent_view", "shared")),
        "lambda_subspace_ortho": float(param.get("lambda_subspace_ortho", 0.0)),
    }


def select_latent_view(z: torch.Tensor, view: str, subspace_cfg: dict) -> torch.Tensor:
    """Select shared-latent slice for a given view name."""
    view = str(view).lower()
    if not subspace_cfg.get("use_tumor_subspace", False) or view == "shared":
        return z
    z_tumor, z_transfer = split_tumor_transfer_latent(z, subspace_cfg["tumor_dim"])
    if view == "tumor":
        return z_tumor
    if view == "transfer":
        return z_transfer
    raise ValueError(f"Unsupported latent view={view}")


def alignment_discriminator_input(z_shared, z_private, subspace_cfg: dict) -> torch.Tensor:
    """Build discriminator input from shared + private latents."""
    view = str(subspace_cfg.get("alignment_latent_view", "shared")).lower()
    if not subspace_cfg.get("use_tumor_subspace", False) or view == "shared":
        return torch.cat((z_shared, z_private), dim=1)
    z_s = select_latent_view(z_shared, view, subspace_cfg)
    z_p = select_latent_view(z_private, view, subspace_cfg)
    return torch.cat((z_s, z_p), dim=1)


def classifier_input_dim(subspace_cfg: dict) -> int:
    view = str(subspace_cfg.get("classifier_latent_view", "shared")).lower()
    if not subspace_cfg.get("use_tumor_subspace", False) or view == "shared":
        return int(subspace_cfg["latent_size"])
    if view == "tumor":
        return int(subspace_cfg["tumor_dim"])
    if view == "transfer":
        return int(subspace_cfg["transfer_dim"])
    return int(subspace_cfg["latent_size"])


def discriminator_input_dim(subspace_cfg: dict) -> int:
    view = str(subspace_cfg.get("alignment_latent_view", "shared")).lower()
    if not subspace_cfg.get("use_tumor_subspace", False) or view == "shared":
        return int(subspace_cfg["latent_size"]) * 2
    if view == "tumor":
        return int(subspace_cfg["tumor_dim"]) * 2
    if view == "transfer":
        return int(subspace_cfg["transfer_dim"]) * 2
    return int(subspace_cfg["latent_size"]) * 2


def compute_subspace_orthogonality_loss(z_tumor, z_transfer):
    """
    Decorrelate tumor and transfer subspaces via cross-covariance penalty.
    """
    if z_transfer.numel() == 0 or z_tumor.size(0) < 2:
        return z_tumor.sum() * 0.0
    z_t = z_tumor - z_tumor.mean(dim=0, keepdim=True)
    z_r = z_transfer - z_transfer.mean(dim=0, keepdim=True)
    cov = (z_t.t() @ z_r) / max(1, z_t.size(0) - 1)
    return cov.pow(2).mean()
