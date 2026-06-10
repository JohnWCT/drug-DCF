"""Batch-based source+target cancer prototype InfoNCE for VAEwC GAN stage."""

from __future__ import annotations

import torch
import torch.nn.functional as F

REQUIRED_METRIC_KEYS = (
    "proto_loss",
    "proto_acc",
    "proto_valid_class_count",
    "proto_valid_sample_count",
    "proto_mean_positive_similarity",
    "proto_mean_negative_similarity",
    "proto_margin",
    "proto_valid",
)


def default_proto_metrics() -> dict:
    """Return zeroed prototype metrics for invalid / skipped batches."""
    return {
        "proto_loss": 0.0,
        "proto_acc": 0.0,
        "proto_valid_class_count": 0,
        "proto_valid_sample_count": 0,
        "proto_mean_positive_similarity": 0.0,
        "proto_mean_negative_similarity": 0.0,
        "proto_margin": 0.0,
        "proto_valid": False,
    }


def _zero_loss(z_source: torch.Tensor) -> torch.Tensor:
    """Scalar zero loss that stays on the autograd graph."""
    return z_source.sum() * 0.0


def compute_batch_prototype_infonce(
    z_source,
    y_source,
    z_target,
    y_target,
    num_classes,
    temperature=0.2,
    min_samples_per_class=1,
):
    """
    Compute batch-based source+target cancer prototype InfoNCE.

    Returns:
        loss: scalar torch.Tensor
        metrics: dict
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")

    z_all = torch.cat([z_source, z_target], dim=0)
    y_all = torch.cat([y_source, y_target], dim=0).long()

    if not torch.isfinite(z_all).all():
        raise FloatingPointError("NaN/Inf detected in latent inputs for prototype InfoNCE")

    prototypes = []
    valid_class_ids = []
    for class_id in range(int(num_classes)):
        mask = y_all == class_id
        count = int(mask.sum().item())
        if count >= int(min_samples_per_class):
            prototypes.append(z_all[mask].mean(dim=0))
            valid_class_ids.append(class_id)

    valid_class_count = len(valid_class_ids)
    metrics = default_proto_metrics()
    metrics["proto_valid_class_count"] = valid_class_count

    if valid_class_count < 2:
        return _zero_loss(z_source), metrics

    proto_matrix = torch.stack(prototypes, dim=0)
    label_map = torch.full((int(num_classes),), -1, dtype=torch.long, device=z_all.device)
    for idx, class_id in enumerate(valid_class_ids):
        label_map[class_id] = idx

    mapped_labels = label_map[y_all]
    valid_sample_mask = mapped_labels >= 0
    valid_sample_count = int(valid_sample_mask.sum().item())
    metrics["proto_valid_sample_count"] = valid_sample_count

    if valid_sample_count < 2:
        return _zero_loss(z_source), metrics

    z_valid = z_all[valid_sample_mask]
    labels_valid = mapped_labels[valid_sample_mask]

    z_norm = F.normalize(z_valid, dim=1)
    p_norm = F.normalize(proto_matrix, dim=1)
    logits = z_norm @ p_norm.t() / float(temperature)

    if not torch.isfinite(logits).all():
        raise FloatingPointError("NaN/Inf detected in prototype InfoNCE logits")

    loss = F.cross_entropy(logits, labels_valid)
    with torch.no_grad():
        sim = z_norm @ p_norm.t()
        preds = logits.argmax(dim=1)
        acc = (preds == labels_valid).float().mean().item()
        pos_sim = sim[torch.arange(sim.size(0), device=sim.device), labels_valid]
        neg_mask = torch.ones_like(sim, dtype=torch.bool)
        neg_mask[torch.arange(sim.size(0), device=sim.device), labels_valid] = False
        neg_sim = sim[neg_mask].view(sim.size(0), -1)
        mean_pos = float(pos_sim.mean().item())
        mean_neg = float(neg_sim.mean().item()) if neg_sim.numel() > 0 else 0.0

    metrics.update(
        {
            "proto_loss": float(loss.detach().item()),
            "proto_acc": acc,
            "proto_mean_positive_similarity": mean_pos,
            "proto_mean_negative_similarity": mean_neg,
            "proto_margin": mean_pos - mean_neg,
            "proto_valid": True,
        }
    )
    return loss, metrics
