"""Class-wise MMD alignment for conditional source-target latent matching."""

from __future__ import annotations

import torch


def _median_heuristic_gamma(x: torch.Tensor, y: torch.Tensor) -> float:
    combined = torch.cat([x, y], dim=0)
    if combined.size(0) < 2:
        return 1.0
    with torch.no_grad():
        dists = torch.pdist(combined, p=2)
        if dists.numel() == 0:
            return 1.0
        median = float(dists.median().item())
    if median <= 0:
        return 1.0
    return 1.0 / (median ** 2)


def _rbf_kernel(x: torch.Tensor, y: torch.Tensor, gamma: float) -> torch.Tensor:
    x_norm = (x * x).sum(dim=1, keepdim=True)
    y_norm = (y * y).sum(dim=1, keepdim=True)
    dist = x_norm + y_norm.t() - 2.0 * x @ y.t()
    return torch.exp(-gamma * dist.clamp_min(0.0))


def _class_mmd_unbiased(x: torch.Tensor, y: torch.Tensor, gamma: float) -> torch.Tensor:
    n = x.size(0)
    m = y.size(0)
    if n < 2 or m < 2:
        return x.sum() * 0.0

    k_xx = _rbf_kernel(x, x, gamma)
    k_yy = _rbf_kernel(y, y, gamma)
    k_xy = _rbf_kernel(x, y, gamma)

    xx = (k_xx.sum() - k_xx.diag().sum()) / (n * (n - 1))
    yy = (k_yy.sum() - k_yy.diag().sum()) / (m * (m - 1))
    xy = k_xy.mean()
    return xx + yy - 2.0 * xy


def compute_classwise_mmd(
    z_source,
    y_source,
    z_target,
    y_target,
    num_classes,
    min_samples_per_domain=2,
    kernel="rbf",
    gamma="median",
):
    """
    Average per-class MMD between source and target latents.

    Returns:
        loss: scalar tensor (mean over valid classes)
        metrics: dict
    """
    if kernel != "rbf":
        raise ValueError(f"Unsupported kernel={kernel}")

    y_source = y_source.long()
    y_target = y_target.long()
    class_losses = []
    valid_class_ids = []

    for class_id in range(int(num_classes)):
        s_mask = y_source == class_id
        t_mask = y_target == class_id
        s_count = int(s_mask.sum().item())
        t_count = int(t_mask.sum().item())
        if s_count >= int(min_samples_per_domain) and t_count >= int(min_samples_per_domain):
            x = z_source[s_mask]
            y = z_target[t_mask]
            if gamma == "median":
                g = _median_heuristic_gamma(x, y)
            else:
                g = float(gamma)
            class_losses.append(_class_mmd_unbiased(x, y, g))
            valid_class_ids.append(class_id)

    metrics = {
        "cmmd_loss": 0.0,
        "cmmd_valid_class_count": len(valid_class_ids),
        "cmmd_valid_sample_count": 0,
        "cmmd_mean_class_loss": 0.0,
        "cmmd_valid": False,
    }

    if not class_losses:
        return z_source.sum() * 0.0, metrics

    stacked = torch.stack(class_losses)
    loss = stacked.mean()
    metrics.update(
        {
            "cmmd_loss": float(loss.detach().item()),
            "cmmd_mean_class_loss": float(stacked.mean().detach().item()),
            "cmmd_valid": True,
        }
    )
    return loss, metrics
