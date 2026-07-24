"""Prototype alignment helpers bridging legacy Stage2 and Round 25 losses."""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from biocda.losses.prototype_margin import margin_gated_prototype_loss
from tools.source_anchor_prototypes import (
    SourceAnchorEMAPrototypes,
    compute_source_anchor_alignment_loss,
)


def compute_margin_gated_alignment_loss(
    target_z: torch.Tensor,
    target_cancer_ids: torch.Tensor,
    source_anchor: SourceAnchorEMAPrototypes,
    margins_by_cancer: torch.Tensor,
    *,
    min_count: int = 2,
) -> Tuple[torch.Tensor, dict]:
    """Build valid cancer centroids then apply margin-gated loss.

    Empty valid set raises (Round 25 contract) — never returns silent 0.
    """
    target_cancer_ids = target_cancer_ids.long()
    centroids: List[torch.Tensor] = []
    anchors: List[torch.Tensor] = []
    margins: List[torch.Tensor] = []
    cancer_ids: List[int] = []

    for class_id in range(source_anchor.num_cancer_types):
        if not bool(source_anchor.initialized[class_id].item()):
            continue
        t_mask = target_cancer_ids == class_id
        if int(t_mask.sum().item()) < int(min_count):
            continue
        target_proto = target_z[t_mask].mean(dim=0)
        if source_anchor.normalize:
            target_proto = F.normalize(target_proto.unsqueeze(0), dim=1).squeeze(0)
        centroids.append(target_proto)
        anchors.append(source_anchor.prototypes[class_id].detach())
        margins.append(margins_by_cancer[class_id])
        cancer_ids.append(class_id)

    if not centroids:
        raise RuntimeError(
            "margin-gated prototype alignment: empty valid cancer set "
            "(no initialized source anchors with sufficient target counts)"
        )

    out = margin_gated_prototype_loss(
        torch.stack(centroids, dim=0),
        torch.stack(anchors, dim=0),
        torch.stack(margins, dim=0),
    )
    metrics = {
        "proto_align_loss": float(out.loss.detach().item()),
        "proto_align_num_cancers": len(cancer_ids),
        "proto_align_metric": "cosine",
        "proto_align_mode": "margin_gated",
        "prototype_hinge_active_fraction": float(out.active_fraction.item()),
        "mean_target_to_source_anchor_distance": float(out.mean_distance.item()),
        "mean_prototype_upper_margin": float(out.mean_margin.item()),
        "proto_align_cancer_ids": cancer_ids,
    }
    return out.loss, metrics


def dispatch_prototype_alignment_loss(
    mode: str,
    target_z: torch.Tensor,
    target_cancer_ids: torch.Tensor,
    source_anchor: SourceAnchorEMAPrototypes,
    *,
    margins_by_cancer: torch.Tensor | None = None,
    min_count: int = 2,
    metric: str = "cosine",
) -> Tuple[torch.Tensor, dict]:
    mode = str(mode).lower()
    if mode in ("always_on", "always-on", "s0"):
        loss, metrics = compute_source_anchor_alignment_loss(
            target_z,
            target_cancer_ids,
            source_anchor,
            metric=metric,
            min_count=min_count,
        )
        metrics = dict(metrics)
        metrics["proto_align_mode"] = "always_on"
        return loss, metrics
    if mode in ("margin_gated", "margin-gated", "s2"):
        if margins_by_cancer is None:
            raise ValueError("margins_by_cancer required for margin_gated mode")
        return compute_margin_gated_alignment_loss(
            target_z,
            target_cancer_ids,
            source_anchor,
            margins_by_cancer,
            min_count=min_count,
        )
    raise ValueError(f"unsupported prototype alignment mode={mode!r}")
