"""Gradient norm monitoring for XA optimization diagnostics."""
from __future__ import annotations

from typing import Dict

from torch import nn


def collect_grad_norms(model: nn.Module) -> Dict[str, float]:
    groups = {
        "sample_projector": "sample_projector",
        "atom_projection": "cross_attention.atom_projection",
        "cross_attention": "cross_attention.layers",
        "gin_final_layer": "drug_encoder.gin.convs.4",
        "response_head": "response_head",
    }
    out: Dict[str, float] = {}
    for label, prefix in groups.items():
        total = 0.0
        count = 0
        for name, p in model.named_parameters():
            if not name.startswith(prefix):
                continue
            if p.grad is None:
                continue
            total += float(p.grad.detach().norm().item())
            count += 1
        out[f"{label}_grad_norm"] = total
        out[f"{label}_grad_param_count"] = float(count)
    return out
