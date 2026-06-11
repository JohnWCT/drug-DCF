"""Class-wise alignment: MMD and same-class prototype gap (Round 5)."""

from __future__ import annotations

import torch
import torch.nn.functional as F


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
    valid_sample_count = 0

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
            valid_sample_count += s_count + t_count

    metrics = {
        "cmmd_loss": 0.0,
        "cmmd_valid_class_count": len(valid_class_ids),
        "cmmd_valid_sample_count": valid_sample_count,
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
            "cmmd_valid_sample_count": valid_sample_count,
            "cmmd_valid": True,
        }
    )
    return loss, metrics


def _class_prototype(z: torch.Tensor, detach: bool) -> torch.Tensor:
    proto = z.mean(dim=0)
    return proto.detach() if detach else proto


def compute_classwise_prototype_gap(
    z_source,
    y_source,
    z_target,
    y_target,
    num_classes,
    min_samples_per_domain=2,
    metric="cosine",
    detach_source=True,
    detach_target=False,
    l2_squared=True,
):
    """
    Same-class prototype gap: P_source[c] aligned toward P_target[c].

    Returns:
        loss: scalar tensor (mean over valid classes; zero graph if none)
        metrics: dict of plain floats / bools for CSV logging
    """
    metric = str(metric).lower()
    if metric not in {"cosine", "l2"}:
        raise ValueError(f"Unsupported class_gap metric={metric}")

    y_source = y_source.long()
    y_target = y_target.long()
    class_losses = []
    per_class_vals = []
    valid_source_count = 0
    valid_target_count = 0

    for class_id in range(int(num_classes)):
        s_mask = y_source == class_id
        t_mask = y_target == class_id
        s_count = int(s_mask.sum().item())
        t_count = int(t_mask.sum().item())
        if s_count >= int(min_samples_per_domain) and t_count >= int(min_samples_per_domain):
            p_source = _class_prototype(z_source[s_mask], detach_source)
            p_target = _class_prototype(z_target[t_mask], detach_target)
            if metric == "cosine":
                sim = F.cosine_similarity(p_target.unsqueeze(0), p_source.unsqueeze(0), dim=1)
                loss_c = 1.0 - sim.squeeze()
            else:
                diff = p_target - p_source
                loss_c = (diff * diff).mean() if l2_squared else diff.norm(p=2)
            class_losses.append(loss_c)
            per_class_vals.append(float(loss_c.detach().item()))
            valid_source_count += s_count
            valid_target_count += t_count

    metrics = {
        "class_gap_loss": 0.0,
        "class_gap_valid": False,
        "class_gap_metric": metric,
        "class_gap_l2_squared": bool(l2_squared) if metric == "l2" else False,
        "class_gap_valid_class_count": len(per_class_vals),
        "class_gap_valid_sample_count_source": valid_source_count,
        "class_gap_valid_sample_count_target": valid_target_count,
        "class_gap_mean": 0.0,
        "class_gap_median": 0.0,
        "class_gap_max": 0.0,
        "class_gap_min": 0.0,
        "class_gap_detach_source": bool(detach_source),
        "class_gap_detach_target": bool(detach_target),
    }

    if not class_losses:
        return z_source.sum() * 0.0, metrics

    stacked = torch.stack(class_losses)
    loss = stacked.mean()
    vals = sorted(per_class_vals)
    mid = len(vals) // 2
    median = vals[mid] if len(vals) % 2 == 1 else 0.5 * (vals[mid - 1] + vals[mid])
    metrics.update(
        {
            "class_gap_loss": float(loss.detach().item()),
            "class_gap_valid": True,
            "class_gap_mean": float(sum(vals) / len(vals)),
            "class_gap_median": float(median),
            "class_gap_max": float(max(vals)),
            "class_gap_min": float(min(vals)),
        }
    )
    return loss, metrics
