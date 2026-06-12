"""Within-domain tumor supervised contrastive learning (Round 6D)."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _within_domain_supcon(z, y, temperature: float, min_samples_per_class: int):
    """Supervised contrastive loss within a single domain."""
    y = y.long()
    n = z.size(0)
    if n < 2:
        return z.sum() * 0.0, False, 0

    z = F.normalize(z, dim=1, eps=1e-8)
    sim = torch.matmul(z, z.t()) / float(temperature)
    sim = sim - torch.max(sim, dim=1, keepdim=True).values.detach()

    valid_classes = []
    losses = []
    for class_id in torch.unique(y):
        mask = y == int(class_id.item())
        count = int(mask.sum().item())
        if count < int(min_samples_per_class):
            continue
        valid_classes.append(int(class_id.item()))
        anchor_idx = mask.nonzero(as_tuple=False).squeeze(1)
        for i in anchor_idx:
            pos_mask = mask.clone()
            pos_mask[i] = False
            if int(pos_mask.sum().item()) == 0:
                continue
            neg_mask = ~mask
            logits = sim[i]
            pos_logits = logits[pos_mask]
            denom_logits = logits[neg_mask | pos_mask]
            if denom_logits.numel() == 0:
                continue
            log_prob = pos_logits - torch.logsumexp(denom_logits, dim=0)
            losses.append(-log_prob.mean())

    if not losses:
        return z.sum() * 0.0, False, len(valid_classes)
    return torch.stack(losses).mean(), True, len(valid_classes)


def compute_within_domain_supcon_loss(
    z_source,
    y_source,
    z_target,
    y_target,
    temperature=1.0,
    min_samples_per_class=2,
    domain_weights=(0.5, 0.5),
):
    """
    Supervised contrastive loss separately within source and target domains.
    """
    w_src, w_tgt = float(domain_weights[0]), float(domain_weights[1])
    loss_s, valid_s, n_cls_s = _within_domain_supcon(z_source, y_source, temperature, min_samples_per_class)
    loss_t, valid_t, n_cls_t = _within_domain_supcon(z_target, y_target, temperature, min_samples_per_class)

    parts = []
    weights = []
    if valid_s:
        parts.append(loss_s)
        weights.append(w_src)
    if valid_t:
        parts.append(loss_t)
        weights.append(w_tgt)

    if not parts:
        loss = z_source.sum() * 0.0
    else:
        wsum = sum(weights)
        loss = sum(p * (w / wsum) for p, w in zip(parts, weights))

    metrics = {
        "tumor_supcon_loss": float(loss.detach().item()),
        "tumor_supcon_source_loss": float(loss_s.detach().item()) if valid_s else 0.0,
        "tumor_supcon_target_loss": float(loss_t.detach().item()) if valid_t else 0.0,
        "tumor_supcon_source_valid": bool(valid_s),
        "tumor_supcon_target_valid": bool(valid_t),
        "tumor_supcon_temperature": float(temperature),
        "tumor_supcon_valid_class_count_source": int(n_cls_s),
        "tumor_supcon_valid_class_count_target": int(n_cls_t),
        "tumor_supcon_valid": bool(valid_s or valid_t),
    }
    return loss, metrics
