"""Attention health diagnostics."""
from __future__ import annotations

import math
from typing import Any, Dict

import torch
from torch import Tensor


def _entropy(probs: Tensor, mask: Tensor, dim: int = -1) -> Tensor:
    p = probs.clamp_min(1e-12)
    ent = -(p * p.log()).sum(dim=dim)
    valid = mask.sum(dim=dim).clamp_min(1).float()
    return ent / valid.log().clamp_min(1e-12)


def attention_health_summary(
    attention: Tensor,
    mask: Tensor,
) -> Dict[str, Any]:
    """Compute entropy, effective atoms, max attention, head diversity."""
    # attention: [B, H, N], mask: [B, N]
    if attention.ndim != 3:
        raise ValueError("attention must be [B, H, N]")
    batch_size, num_heads, max_atoms = attention.shape
    m = mask.unsqueeze(1).expand_as(attention)
    valid_attn = attention * m.to(attention.dtype)
    n_atoms = mask.sum(dim=-1).float().clamp_min(1.0)

    ent = -(valid_attn.clamp_min(1e-12) * valid_attn.clamp_min(1e-12).log()).sum(dim=-1)
    norm_ent = ent / torch.log(n_atoms).unsqueeze(-1).clamp_min(1e-12)

    max_attn = valid_attn.max(dim=-1).values
    eff_atoms = torch.exp(ent)

    head_sims = []
    if num_heads > 1:
        flat = valid_attn.reshape(batch_size, num_heads, -1)
        for h1 in range(num_heads):
            for h2 in range(h1 + 1, num_heads):
                a = flat[:, h1]
                b = flat[:, h2]
                cos = torch.nn.functional.cosine_similarity(a, b, dim=-1)
                head_sims.append(float(cos.mean()))

    return {
        "mean_normalized_entropy": float(norm_ent.mean()),
        "median_normalized_entropy": float(norm_ent.median()),
        "p05_normalized_entropy": float(torch.quantile(norm_ent.flatten(), 0.05)),
        "p95_normalized_entropy": float(torch.quantile(norm_ent.flatten(), 0.95)),
        "mean_max_atom_attention": float(max_attn.mean()),
        "mean_effective_atoms": float(eff_atoms.mean()),
        "mean_effective_atom_fraction": float((eff_atoms / n_atoms.unsqueeze(-1)).mean()),
        "mean_head_cosine_similarity": float(sum(head_sims) / len(head_sims)) if head_sims else 0.0,
        "num_heads": num_heads,
        "max_atoms": max_atoms,
    }
