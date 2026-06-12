"""Tumor prototype topology preservation (Round 6A)."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def compute_class_prototypes(
    z,
    y,
    num_classes,
    min_samples_per_class=2,
    detach=False,
):
    """
    Per-class mean prototypes for classes with enough samples.

    Returns:
        prototypes: Tensor [n_valid_classes, dim]
        valid_classes: Tensor [n_valid_classes] (long)
        counts: Tensor [n_valid_classes]
    """
    y = y.long()
    protos = []
    valid_classes = []
    counts = []
    for class_id in range(int(num_classes)):
        mask = y == class_id
        count = int(mask.sum().item())
        if count >= int(min_samples_per_class):
            proto = z[mask].mean(dim=0)
            if detach:
                proto = proto.detach()
            protos.append(proto)
            valid_classes.append(class_id)
            counts.append(count)
    if not protos:
        empty = z.new_zeros((0, z.size(-1)))
        return empty, z.new_zeros(0, dtype=torch.long), z.new_zeros(0, dtype=torch.long)
    return torch.stack(protos), torch.tensor(valid_classes, device=z.device, dtype=torch.long), torch.tensor(
        counts, device=z.device, dtype=torch.long
    )


def compute_pairwise_distance_matrix(
    prototypes,
    metric="cosine_distance",
    normalize=True,
):
    """
    Pairwise distance matrix for prototype rows.

    metric:
        cosine_distance
        euclidean
        squared_euclidean
    """
    if prototypes.numel() == 0:
        return prototypes.new_zeros((0, 0))
    metric = str(metric).lower()
    n = prototypes.size(0)
    if metric == "cosine_distance":
        normed = F.normalize(prototypes, dim=1, eps=1e-8)
        sim = normed @ normed.t()
        dist = 1.0 - sim
        dist.fill_diagonal_(0.0)
    elif metric == "euclidean":
        dist = torch.cdist(prototypes, prototypes, p=2)
    elif metric == "squared_euclidean":
        dist = torch.cdist(prototypes, prototypes, p=2).pow(2)
    else:
        raise ValueError(f"Unsupported topology metric={metric}")
    if normalize and n > 0:
        max_val = dist.max()
        if float(max_val.item()) > 0:
            dist = dist / max_val
    return dist


def _upper_triangle_flat(mat: torch.Tensor) -> torch.Tensor:
    n = mat.size(0)
    if n < 2:
        return mat.new_zeros(0)
    idx = torch.triu_indices(n, n, offset=1)
    return mat[idx[0], idx[1]]


def _pearson_corr(x: torch.Tensor, y: torch.Tensor) -> float:
    if x.numel() < 2:
        return float("nan")
    x = x.float()
    y = y.float()
    x = x - x.mean()
    y = y - y.mean()
    denom = x.norm() * y.norm()
    if float(denom.item()) <= 0:
        return float("nan")
    return float((x @ y / denom).item())


def _topology_loss_value(pred: torch.Tensor, target: torch.Tensor, loss_type: str) -> torch.Tensor:
    loss_type = str(loss_type).lower()
    if loss_type == "smooth_l1":
        return F.smooth_l1_loss(pred, target)
    if loss_type == "l1":
        return F.l1_loss(pred, target)
    if loss_type == "mse":
        return F.mse_loss(pred, target)
    raise ValueError(f"Unsupported topology_loss_type={loss_type}")


def compute_tumor_topology_loss(
    z_source,
    y_source,
    z_target,
    y_target,
    num_classes,
    min_samples_per_domain=2,
    metric="cosine_distance",
    topology_loss_type="smooth_l1",
    detach_source=True,
    normalize_distance=True,
):
    """
    Compare source and target class-prototype distance matrices on shared valid classes.

    Returns:
        loss: scalar tensor (zero graph if invalid)
        metrics: dict of plain floats / bools
    """
    metric = str(metric).lower()
    p_src, cls_src, _ = compute_class_prototypes(
        z_source, y_source, num_classes, min_samples_per_class=min_samples_per_domain, detach=False
    )
    p_tgt, cls_tgt, _ = compute_class_prototypes(
        z_target, y_target, num_classes, min_samples_per_class=min_samples_per_domain, detach=False
    )

    metrics = {
        "tumor_topology_loss": 0.0,
        "tumor_topology_valid": False,
        "tumor_topology_valid_class_count": 0,
        "tumor_topology_metric": metric,
        "tumor_topology_loss_type": str(topology_loss_type),
        "tumor_topology_source_mean_distance": 0.0,
        "tumor_topology_target_mean_distance": 0.0,
        "tumor_topology_distance_corr": float("nan"),
        "tumor_topology_distance_mae": 0.0,
        "tumor_topology_distance_rmse": 0.0,
        "tumor_topology_detach_source": bool(detach_source),
    }

    if p_src.numel() == 0 or p_tgt.numel() == 0:
        return z_source.sum() * 0.0, metrics

    common = sorted(set(cls_src.tolist()) & set(cls_tgt.tolist()))
    if len(common) < 3:
        return z_source.sum() * 0.0, metrics

    # align prototypes by class id
    idx_src = [cls_src.tolist().index(c) for c in common]
    idx_tgt = [cls_tgt.tolist().index(c) for c in common]
    p_src = p_src[idx_src]
    p_tgt = p_tgt[idx_tgt]

    d_src = compute_pairwise_distance_matrix(p_src, metric=metric, normalize=normalize_distance)
    d_tgt = compute_pairwise_distance_matrix(p_tgt, metric=metric, normalize=normalize_distance)
    if detach_source:
        d_ref = d_src.detach()
    else:
        d_ref = d_src

    loss = _topology_loss_value(d_tgt, d_ref, topology_loss_type)

    tri_src = _upper_triangle_flat(d_src.detach())
    tri_tgt = _upper_triangle_flat(d_tgt.detach())
    tri_diff = tri_tgt - tri_src
    mae = float(tri_diff.abs().mean().item()) if tri_diff.numel() else 0.0
    rmse = float(torch.sqrt((tri_diff.pow(2).mean() + 1e-12)).item()) if tri_diff.numel() else 0.0
    corr = _pearson_corr(tri_tgt, tri_src)

    metrics.update(
        {
            "tumor_topology_loss": float(loss.detach().item()),
            "tumor_topology_valid": True,
            "tumor_topology_valid_class_count": len(common),
            "tumor_topology_source_mean_distance": float(tri_src.mean().item()) if tri_src.numel() else 0.0,
            "tumor_topology_target_mean_distance": float(tri_tgt.mean().item()) if tri_tgt.numel() else 0.0,
            "tumor_topology_distance_corr": corr if not math.isnan(corr) else 0.0,
            "tumor_topology_distance_mae": mae,
            "tumor_topology_distance_rmse": rmse,
        }
    )
    return loss, metrics
