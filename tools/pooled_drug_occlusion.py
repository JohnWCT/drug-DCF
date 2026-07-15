#!/usr/bin/env python3
"""Input-only perturbation adapter for pooled drug models (no attention claims)."""
from __future__ import annotations

from typing import Callable, Iterable

import torch


def pooled_input_occlusion(
    predict: Callable[[torch.Tensor], torch.Tensor],
    drug_inputs: torch.Tensor,
    feature_groups: Iterable[Iterable[int]],
) -> list[dict]:
    baseline = predict(drug_inputs)
    rows = []
    for group_id, indices in enumerate(feature_groups):
        perturbed = drug_inputs.clone()
        selected = sorted(set(map(int, indices)))
        perturbed[..., selected] = 0
        value = predict(perturbed)
        rows.append({
            "group_id": group_id,
            "feature_indices": selected,
            "prediction_delta": (baseline - value).detach().cpu(),
            "method": "input_perturbation",
            "has_attention": False,
        })
    return rows
