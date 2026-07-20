"""Query sensitivity diagnostics."""
from __future__ import annotations

from typing import Any, Dict, List

import torch
from torch import Tensor


def _js_distance(p: Tensor, q: Tensor, eps: float = 1e-12) -> Tensor:
    m = 0.5 * (p + q)
    kl_pm = (p * (p / m.clamp_min(eps)).log()).sum(dim=-1)
    kl_qm = (q * (q / m.clamp_min(eps)).log()).sum(dim=-1)
    return 0.5 * (kl_pm + kl_qm)


def compare_attention(a: Tensor, b: Tensor, mask: Tensor) -> Dict[str, float]:
    m = mask.unsqueeze(1).to(a.dtype)
    pa = (a * m) / a.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    pb = (b * m) / b.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    l1 = (pa - pb).abs().sum(dim=-1).mean()
    js = _js_distance(pa.clamp_min(1e-12), pb.clamp_min(1e-12)).mean()
    return {"attention_l1_distance": float(l1), "attention_js_distance": float(js)}


def query_shuffle_delta(
    attention_before: Tensor,
    attention_after: Tensor,
    logits_before: Tensor,
    logits_after: Tensor,
    mask: Tensor,
) -> Dict[str, float]:
    attn = compare_attention(attention_before, attention_after, mask)
    pred_delta = float((logits_before - logits_after).abs().mean())
    return {**attn, "prediction_l1_delta": pred_delta}
