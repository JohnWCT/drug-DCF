"""Batch-based prototype InfoNCE for VAEwC GAN stage (combined + cross-domain)."""

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

EXTENDED_METRIC_KEYS = (
    "proto_mode",
    "proto_direction",
    "proto_cross_domain_valid",
    "proto_t2s_loss",
    "proto_s2t_loss",
    "proto_t2s_acc",
    "proto_s2t_acc",
    "proto_t2s_valid_sample_count",
    "proto_s2t_valid_sample_count",
    "proto_source_valid_class_count",
    "proto_target_valid_class_count",
    "proto_detach",
)


def default_proto_metrics(
    mode: str = "combined",
    direction: str = "symmetric",
    detach: bool = True,
) -> dict:
    """Return zeroed prototype metrics for invalid / skipped batches."""
    metrics = {
        "proto_loss": 0.0,
        "proto_acc": 0.0,
        "proto_valid_class_count": 0,
        "proto_valid_sample_count": 0,
        "proto_mean_positive_similarity": 0.0,
        "proto_mean_negative_similarity": 0.0,
        "proto_margin": 0.0,
        "proto_valid": False,
        "proto_mode": mode,
        "proto_direction": direction,
        "proto_cross_domain_valid": False,
        "proto_t2s_loss": 0.0,
        "proto_s2t_loss": 0.0,
        "proto_t2s_acc": 0.0,
        "proto_s2t_acc": 0.0,
        "proto_t2s_valid_sample_count": 0,
        "proto_s2t_valid_sample_count": 0,
        "proto_source_valid_class_count": 0,
        "proto_target_valid_class_count": 0,
        "proto_detach": detach,
    }
    return metrics


def _zero_loss(z_source: torch.Tensor) -> torch.Tensor:
    """Scalar zero loss that stays on the autograd graph."""
    return z_source.sum() * 0.0


def _build_domain_prototypes(
    z: torch.Tensor,
    y: torch.Tensor,
    num_classes: int,
    min_samples_per_domain: int,
    detach: bool,
):
    prototypes = []
    valid_class_ids = []
    for class_id in range(int(num_classes)):
        mask = y == class_id
        count = int(mask.sum().item())
        if count >= int(min_samples_per_domain):
            proto = z[mask].mean(dim=0)
            if detach:
                proto = proto.detach()
            prototypes.append(proto)
            valid_class_ids.append(class_id)
    if not prototypes:
        return None, []
    return torch.stack(prototypes, dim=0), valid_class_ids


def _cross_domain_direction_loss(
    anchors: torch.Tensor,
    anchor_labels: torch.Tensor,
    proto_matrix: torch.Tensor,
    valid_class_ids: list,
    num_classes: int,
    temperature: float,
):
    """Anchors classify against prototypes from the opposite domain."""
    label_map = torch.full((int(num_classes),), -1, dtype=torch.long, device=anchors.device)
    for idx, class_id in enumerate(valid_class_ids):
        label_map[class_id] = idx

    mapped_labels = label_map[anchor_labels.long()]
    valid_sample_mask = mapped_labels >= 0
    valid_sample_count = int(valid_sample_mask.sum().item())
    if valid_sample_count < 1 or proto_matrix.size(0) < 2:
        return _zero_loss(anchors), {
            "loss": 0.0,
            "acc": 0.0,
            "valid_sample_count": valid_sample_count,
            "valid": False,
            "mean_pos": 0.0,
            "mean_neg": 0.0,
            "margin": 0.0,
        }

    z_valid = anchors[valid_sample_mask]
    labels_valid = mapped_labels[valid_sample_mask]

    z_norm = F.normalize(z_valid, dim=1)
    p_norm = F.normalize(proto_matrix, dim=1)
    logits = z_norm @ p_norm.t() / float(temperature)

    if not torch.isfinite(logits).all():
        raise FloatingPointError("NaN/Inf detected in cross-domain prototype InfoNCE logits")

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

    return loss, {
        "loss": float(loss.detach().item()),
        "acc": acc,
        "valid_sample_count": valid_sample_count,
        "valid": True,
        "mean_pos": mean_pos,
        "mean_neg": mean_neg,
        "margin": mean_pos - mean_neg,
    }


def _compute_cross_domain_infonce(
    z_source,
    y_source,
    z_target,
    y_target,
    num_classes,
    temperature,
    min_samples_per_domain,
    direction,
    detach_prototypes,
):
    metrics = default_proto_metrics(mode="cross_domain", direction=direction, detach=detach_prototypes)
    y_source = y_source.long()
    y_target = y_target.long()

    source_proto, source_valid_ids = _build_domain_prototypes(
        z_source, y_source, num_classes, min_samples_per_domain, detach_prototypes
    )
    target_proto, target_valid_ids = _build_domain_prototypes(
        z_target, y_target, num_classes, min_samples_per_domain, detach_prototypes
    )
    metrics["proto_source_valid_class_count"] = len(source_valid_ids)
    metrics["proto_target_valid_class_count"] = len(target_valid_ids)

    losses = []
    accs = []
    pos_sims = []
    neg_sims = []

    if direction in ("target_to_source", "symmetric") and source_proto is not None:
        t2s_loss, t2s = _cross_domain_direction_loss(
            z_target,
            y_target,
            source_proto,
            source_valid_ids,
            num_classes,
            temperature,
        )
        metrics["proto_t2s_loss"] = t2s["loss"]
        metrics["proto_t2s_acc"] = t2s["acc"]
        metrics["proto_t2s_valid_sample_count"] = t2s["valid_sample_count"]
        if t2s["valid"]:
            losses.append(t2s_loss)
            accs.append(t2s["acc"])
            pos_sims.append(t2s["mean_pos"])
            neg_sims.append(t2s["mean_neg"])

    if direction in ("source_to_target", "symmetric") and target_proto is not None:
        s2t_loss, s2t = _cross_domain_direction_loss(
            z_source,
            y_source,
            target_proto,
            target_valid_ids,
            num_classes,
            temperature,
        )
        metrics["proto_s2t_loss"] = s2t["loss"]
        metrics["proto_s2t_acc"] = s2t["acc"]
        metrics["proto_s2t_valid_sample_count"] = s2t["valid_sample_count"]
        if s2t["valid"]:
            losses.append(s2t_loss)
            accs.append(s2t["acc"])
            pos_sims.append(s2t["mean_pos"])
            neg_sims.append(s2t["mean_neg"])

    if not losses:
        return _zero_loss(z_source), metrics

    loss = sum(losses) / len(losses)
    metrics["proto_cross_domain_valid"] = True
    metrics["proto_valid"] = True
    metrics["proto_loss"] = float(loss.detach().item())
    metrics["proto_acc"] = float(sum(accs) / len(accs))
    metrics["proto_valid_class_count"] = len(set(source_valid_ids) & set(target_valid_ids))
    metrics["proto_valid_sample_count"] = (
        metrics["proto_t2s_valid_sample_count"] + metrics["proto_s2t_valid_sample_count"]
    )
    metrics["proto_mean_positive_similarity"] = float(sum(pos_sims) / len(pos_sims))
    metrics["proto_mean_negative_similarity"] = float(sum(neg_sims) / len(neg_sims))
    metrics["proto_margin"] = (
        metrics["proto_mean_positive_similarity"] - metrics["proto_mean_negative_similarity"]
    )
    return loss, metrics


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
    Compute batch-based source+target cancer prototype InfoNCE (combined mode).

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
    metrics = default_proto_metrics(mode="combined", direction="symmetric", detach=False)

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
    metrics["proto_valid_class_count"] = valid_class_count

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


def compute_prototype_infonce(
    z_source,
    y_source,
    z_target,
    y_target,
    num_classes,
    temperature=0.2,
    min_samples_per_class=1,
    min_samples_per_domain=1,
    mode="combined",
    direction="symmetric",
    detach_prototypes=True,
):
    """
    Unified prototype InfoNCE entry point.

    mode="combined": source+target combined prototypes (Round 3 behavior).
    mode="cross_domain": opposite-domain prototypes (Round 4).
    """
    mode = str(mode).lower()
    direction = str(direction).lower()

    if mode == "combined":
        loss, metrics = compute_batch_prototype_infonce(
            z_source,
            y_source,
            z_target,
            y_target,
            num_classes=num_classes,
            temperature=temperature,
            min_samples_per_class=min_samples_per_class,
        )
        metrics["proto_detach"] = bool(detach_prototypes)
        return loss, metrics

    if mode == "cross_domain":
        if direction not in {"target_to_source", "source_to_target", "symmetric"}:
            raise ValueError(f"Unsupported proto_direction={direction}")
        return _compute_cross_domain_infonce(
            z_source,
            y_source,
            z_target,
            y_target,
            num_classes=num_classes,
            temperature=temperature,
            min_samples_per_domain=min_samples_per_domain,
            direction=direction,
            detach_prototypes=detach_prototypes,
        )

    raise ValueError(f"Unsupported proto_mode={mode}")
