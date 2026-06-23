"""Source-anchor EMA prototype alignment (Round 12)."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F

SUPPORTED_PROTO_ALIGN_METRICS = frozenset({"cosine", "euclidean"})


def resolve_source_anchor_proto_training_params(params: dict) -> dict:
    enabled = bool(params.get("source_anchor_proto_enabled", False))
    metric = str(params.get("proto_align_metric", "cosine")).lower()
    momentum = float(params.get("proto_ema_momentum", 0.95))
    min_count = int(params.get("proto_align_min_count", 2))
    if metric not in SUPPORTED_PROTO_ALIGN_METRICS:
        raise ValueError(
            f"Unsupported proto_align_metric={metric!r}; "
            f"expected one of {sorted(SUPPORTED_PROTO_ALIGN_METRICS)}"
        )
    if not (0.0 <= momentum < 1.0):
        raise ValueError(f"proto_ema_momentum must be in [0, 1), got {momentum}")
    if min_count < 1:
        raise ValueError(f"proto_align_min_count must be >= 1, got {min_count}")
    return {
        "source_anchor_proto_enabled": enabled,
        "lambda_proto_align": float(params.get("lambda_proto_align", 0.0)),
        "proto_align_metric": metric,
        "proto_align_start_epoch": int(params.get("proto_align_start_epoch", 20)),
        "proto_align_full_epoch": int(params.get("proto_align_full_epoch", 90)),
        "proto_ema_momentum": momentum,
        "proto_align_min_count": min_count,
        "proto_align_normalize": bool(params.get("proto_align_normalize", True)),
        "proto_align_update_source_ema": bool(params.get("proto_align_update_source_ema", True)),
    }


def get_proto_align_lambda_eff(
    epoch: int,
    lambda_proto_align: float,
    start_epoch: int,
    full_epoch: int,
) -> float:
    lam = float(lambda_proto_align)
    if lam <= 0.0:
        return 0.0
    if epoch < int(start_epoch):
        return 0.0
    if epoch >= int(full_epoch):
        return lam
    ramp = (epoch - int(start_epoch)) / max(1, int(full_epoch) - int(start_epoch))
    return lam * ramp


class SourceAnchorEMAPrototypes:
    """Maintains source-domain EMA prototypes by cancer type (no gradients)."""

    def __init__(
        self,
        num_cancer_types: int,
        latent_size: int,
        momentum: float = 0.95,
        normalize: bool = True,
        device=None,
    ):
        if device is None:
            device = torch.device("cpu")
        self.num_cancer_types = int(num_cancer_types)
        self.latent_size = int(latent_size)
        self.momentum = float(momentum)
        self.normalize = bool(normalize)
        self.device = device
        self.prototypes = torch.zeros(self.num_cancer_types, self.latent_size, device=device)
        self.initialized = torch.zeros(self.num_cancer_types, dtype=torch.bool, device=device)
        self.counts = torch.zeros(self.num_cancer_types, dtype=torch.long, device=device)

    @torch.no_grad()
    def update(self, source_z, source_cancer_ids, min_count: int = 2) -> None:
        source_z = source_z.detach()
        source_cancer_ids = source_cancer_ids.long().detach()
        for class_id in range(self.num_cancer_types):
            mask = source_cancer_ids == class_id
            n = int(mask.sum().item())
            if n < int(min_count):
                continue
            batch_mean = source_z[mask].mean(dim=0)
            if self.normalize:
                batch_mean = F.normalize(batch_mean.unsqueeze(0), dim=1).squeeze(0)
            if not bool(self.initialized[class_id].item()):
                self.prototypes[class_id] = batch_mean
                self.initialized[class_id] = True
            else:
                m = self.momentum
                updated = m * self.prototypes[class_id] + (1.0 - m) * batch_mean
                if self.normalize:
                    updated = F.normalize(updated.unsqueeze(0), dim=1).squeeze(0)
                self.prototypes[class_id] = updated
            self.counts[class_id] += n

    def get(self, cancer_ids):
        return self.prototypes[cancer_ids.long()].detach()

    def initialized_count(self) -> int:
        return int(self.initialized.sum().item())


def compute_batch_prototypes(
    z: torch.Tensor,
    cancer_ids: torch.Tensor,
    num_cancer_types: int,
    min_count: int = 2,
) -> Dict[int, torch.Tensor]:
    cancer_ids = cancer_ids.long()
    out: Dict[int, torch.Tensor] = {}
    for class_id in range(int(num_cancer_types)):
        mask = cancer_ids == class_id
        if int(mask.sum().item()) >= int(min_count):
            out[class_id] = z[mask].mean(dim=0)
    return out


def compute_source_anchor_alignment_loss(
    target_z: torch.Tensor,
    target_cancer_ids: torch.Tensor,
    source_anchor: SourceAnchorEMAPrototypes,
    metric: str = "cosine",
    min_count: int = 2,
) -> Tuple[torch.Tensor, dict]:
    metric = str(metric).lower()
    if metric not in SUPPORTED_PROTO_ALIGN_METRICS:
        raise ValueError(f"Unsupported proto_align metric={metric}")

    target_cancer_ids = target_cancer_ids.long()
    class_losses = []
    distances = []
    skip_count = 0

    for class_id in range(source_anchor.num_cancer_types):
        if not bool(source_anchor.initialized[class_id].item()):
            skip_count += 1
            continue
        t_mask = target_cancer_ids == class_id
        if int(t_mask.sum().item()) < int(min_count):
            skip_count += 1
            continue
        target_proto = target_z[t_mask].mean(dim=0)
        anchor = source_anchor.prototypes[class_id].detach()
        if source_anchor.normalize and metric == "cosine":
            target_proto = F.normalize(target_proto.unsqueeze(0), dim=1).squeeze(0)
        if metric == "cosine":
            sim = F.cosine_similarity(target_proto.unsqueeze(0), anchor.unsqueeze(0), dim=1)
            loss_c = 1.0 - sim.squeeze()
            distances.append(float(loss_c.detach().item()))
        else:
            diff = target_proto - anchor
            loss_c = (diff * diff).sum()
            distances.append(float(diff.norm(p=2).detach().item()))
        class_losses.append(loss_c)

    metrics = {
        "proto_align_loss": 0.0,
        "proto_align_num_cancers": len(class_losses),
        "proto_align_skip_count": skip_count,
        "proto_align_metric": metric,
        "mean_target_to_source_anchor_distance": 0.0,
    }

    if not class_losses:
        return target_z.sum() * 0.0, metrics

    stacked = torch.stack(class_losses)
    loss = stacked.mean()
    metrics.update(
        {
            "proto_align_loss": float(loss.detach().item()),
            "proto_align_num_cancers": len(class_losses),
            "mean_target_to_source_anchor_distance": float(sum(distances) / len(distances)),
        }
    )
    return loss, metrics


def source_anchor_proto_metrics_payload(proto_cfg: dict, gan_logs: dict) -> dict:
    payload = {
        "source_anchor_proto_enabled": bool(proto_cfg.get("source_anchor_proto_enabled", False)),
        "lambda_proto_align": float(proto_cfg.get("lambda_proto_align", 0.0)),
        "proto_align_metric": proto_cfg.get("proto_align_metric", "cosine"),
        "proto_align_start_epoch": int(proto_cfg.get("proto_align_start_epoch", 20)),
        "proto_align_full_epoch": int(proto_cfg.get("proto_align_full_epoch", 90)),
        "proto_ema_momentum": float(proto_cfg.get("proto_ema_momentum", 0.95)),
        "proto_align_min_count": int(proto_cfg.get("proto_align_min_count", 2)),
        "proto_align_normalize": bool(proto_cfg.get("proto_align_normalize", True)),
    }
    payload.update(gan_logs)
    return payload
