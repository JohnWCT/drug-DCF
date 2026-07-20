"""Biological context utilization diagnostics."""
from __future__ import annotations

from typing import Any, Dict

import torch
from torch import Tensor

from biocda.diagnostics.query_sensitivity import compare_attention, _js_distance


def context_intervention_summary(
    logits_original: Tensor,
    logits_intervened: Tensor,
    attention_original: Tensor,
    attention_intervened: Tensor,
    mask: Tensor,
) -> Dict[str, float]:
    attn = compare_attention(attention_original, attention_intervened, mask)
    return {
        **attn,
        "prediction_delta": float((logits_original - logits_intervened).abs().mean()),
    }
